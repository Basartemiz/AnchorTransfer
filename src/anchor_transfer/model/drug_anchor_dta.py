"""Drug-Anchor DTA — drug-side anchor transfer model.

Given (anchor_drug, query_drug, protein) predicts:
  1. Binary binding classification (does query_drug bind the protein?)
  2. Regression pKi affinity score

The anchor drug is a known strong binder of the target protein.
Knowledge transfer: "drug X binds this protein with pKi=9, what about drug Y?"

Architecture mirrors AnchorTransferDTAv2 but swaps the anchor axis:
  V2:         (anchor_protein, query_protein, drug) → pKi
  DrugAnchor: (anchor_drug, query_drug, protein) → pKi
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_transfer.model.anchor_transfer import CHARISOSMISET, SMILES_MAX_LEN, encode_smiles


class SMILESEncoder(nn.Module):
    """Parallel CNN encoder for SMILES strings (shared for anchor and query drug)."""

    def __init__(self, vocab_size: int = 52, embed_dim: int = 128,
                 num_filters: int = 32, output_dim: int = 256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, smiles_indices: torch.Tensor) -> torch.Tensor:
        x = self.embed(smiles_indices).permute(0, 2, 1)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(outs, dim=1)
        return self.proj(x)


class PairwiseInteraction(nn.Module):
    """Pairwise interaction MLP: [x, y] → MLP → ℝ^D."""

    def __init__(self, dim_x: int, dim_y: int, out_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim_x + dim_y, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([x, y], dim=-1))


class DrugAnchorDTA(nn.Module):
    """Drug-anchor transfer model with triple pairwise interactions."""

    def __init__(
        self,
        esm2_dim: int = 480,
        proj_dim: int = 256,
        dropout: float = 0.3,
        smiles_vocab: int = 52,
        smiles_embed_dim: int = 128,
        smiles_filters: int = 32,
    ):
        super().__init__()

        # Protein projection (ESM-2 → proj_dim)
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # Shared drug encoder for both anchor and query drug
        self.drug_encoder = SMILESEncoder(
            vocab_size=smiles_vocab,
            embed_dim=smiles_embed_dim,
            num_filters=smiles_filters,
            output_dim=proj_dim,
        )

        # Triple pairwise interactions
        self.interact_anchor_query = PairwiseInteraction(proj_dim, proj_dim, proj_dim, dropout)  # anchor_drug ↔ query_drug
        self.interact_anchor_prot = PairwiseInteraction(proj_dim, proj_dim, proj_dim, dropout)   # anchor_drug ↔ protein
        self.interact_query_prot = PairwiseInteraction(proj_dim, proj_dim, proj_dim, dropout)    # query_drug ↔ protein

        # Fusion: [anchor_drug, query_drug, protein, h_aq, h_ap, h_qp] = 6 * proj_dim
        fused_dim = 6 * proj_dim

        # Dual prediction heads
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        anchor_drug_indices: torch.Tensor,
        query_drug_indices: torch.Tensor,
        protein_esm2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        anchor_drug_indices : (B, max_len) int — encoded SMILES of known binder drug
        query_drug_indices : (B, max_len) int — encoded SMILES of query drug
        protein_esm2 : (B, esm2_dim) — ESM-2 embedding of target protein

        Returns
        -------
        dict with 'binding_logit', 'binding_prob', 'pki_pred'
        """
        ad = self.drug_encoder(anchor_drug_indices)  # (B, D)
        qd = self.drug_encoder(query_drug_indices)   # (B, D)
        p = self.protein_proj(protein_esm2)          # (B, D)

        h_aq = self.interact_anchor_query(ad, qd)   # anchor_drug ↔ query_drug
        h_ap = self.interact_anchor_prot(ad, p)      # anchor_drug ↔ protein
        h_qp = self.interact_query_prot(qd, p)       # query_drug ↔ protein

        z = torch.cat([ad, qd, p, h_aq, h_ap, h_qp], dim=-1)  # (B, 6D)

        binding_logit = self.classifier(z).squeeze(-1)
        pki_pred = self.regressor(z).squeeze(-1)

        return {
            "binding_logit": binding_logit,
            "binding_prob": torch.sigmoid(binding_logit),
            "pki_pred": pki_pred,
        }

    def compute_loss(
        self,
        anchor_drug_indices: torch.Tensor,
        query_drug_indices: torch.Tensor,
        protein_esm2: torch.Tensor,
        pki_targets: torch.Tensor,
        binding_labels: torch.Tensor | None = None,
        binding_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss: BCE (binary) + α * MSE (regression)."""
        out = self.forward(anchor_drug_indices, query_drug_indices, protein_esm2)

        mse_loss = F.mse_loss(out["pki_pred"], pki_targets)

        if binding_mask is not None and binding_mask.any():
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"][binding_mask],
                binding_labels[binding_mask].float(),
            )
        elif binding_labels is not None and binding_mask is None:
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"], binding_labels.float(),
            )
        else:
            bce_loss = torch.tensor(0.0, device=pki_targets.device)

        total_loss = bce_loss + alpha * mse_loss
        out["loss"] = total_loss
        out["bce_loss"] = bce_loss
        out["mse_loss"] = mse_loss
        return out
