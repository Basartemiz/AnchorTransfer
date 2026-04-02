"""Anchor Transfer DTA v2 — Triple Cross-Attention model.

Given (anchor_protein, query_protein, drug) predicts:
  1. Binary binding classification (does query bind the drug?)
  2. Regression pKi affinity score

Architecture:
  Stage 1: Shared protein projection (ESM-2 → D) + SMILES CNN (→ D)
  Stage 2: Triple bidirectional cross-attention (anchor↔drug, query↔drug, anchor↔query)
  Stage 3: Fusion [a' ∥ q' ∥ d' ∥ h_ad ∥ h_qd ∥ h_aq] → 6D
  Stage 4: Dual MLP heads → binary + regression
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export encode_smiles from v1 for compatibility
from idr_gat.model.anchor_transfer import CHARISOSMISET, SMILES_MAX_LEN, encode_smiles


class SMILESEncoder(nn.Module):
    """Parallel CNN encoder for SMILES strings."""

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
    """Pairwise interaction between two embeddings.

    Given x, y ∈ ℝ^D:
      concat: [x, y] → MLP → ℝ^D
      output = MLP([x, y])  (no residual to avoid accumulation)
    """

    def __init__(self, dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Args: x (B, D), y (B, D). Returns: (B, D)."""
        return self.mlp(torch.cat([x, y], dim=-1))  # (B, D)


class AnchorTransferDTAv2(nn.Module):
    """Advanced anchor-based transfer model with triple cross-attention."""

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

        # Stage 1: Modality projections
        # Shared protein projection (anchor and query use same weights)
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )

        # Drug encoder (SMILES CNN → proj_dim)
        self.drug_encoder = SMILESEncoder(
            vocab_size=smiles_vocab,
            embed_dim=smiles_embed_dim,
            num_filters=smiles_filters,
            output_dim=proj_dim,
        )

        # Stage 2: Triple pairwise interactions
        self.interact_anchor_drug = PairwiseInteraction(proj_dim, dropout)
        self.interact_query_drug = PairwiseInteraction(proj_dim, dropout)
        self.interact_anchor_query = PairwiseInteraction(proj_dim, dropout)

        # Stage 3: Fusion dimension = 6 * proj_dim
        fused_dim = 6 * proj_dim  # [a', q', d', h_ad, h_qd, h_aq]

        # Stage 4: Dual prediction heads
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(
        self,
        anchor_esm2: torch.Tensor,
        query_esm2: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        anchor_esm2 : (B, esm2_dim) ESM-2 embedding of known binder
        query_esm2 : (B, esm2_dim) ESM-2 embedding of query protein
        drug_indices : (B, max_len) int — encoded SMILES characters

        Returns
        -------
        dict with 'binding_logit', 'binding_prob', 'pki_pred'
        """
        # Stage 1: Project to shared space
        a = self.protein_proj(anchor_esm2)   # (B, D)
        q = self.protein_proj(query_esm2)    # (B, D)
        d = self.drug_encoder(drug_indices)  # (B, D)

        # Stage 2: Triple pairwise interactions
        h_ad = self.interact_anchor_drug(a, d)   # anchor ↔ drug
        h_qd = self.interact_query_drug(q, d)    # query ↔ drug
        h_aq = self.interact_anchor_query(a, q)  # anchor ↔ query

        # Stage 3: Fusion
        z = torch.cat([a, q, d, h_ad, h_qd, h_aq], dim=-1)  # (B, 6D)

        # Stage 4: Dual heads
        binding_logit = self.classifier(z).squeeze(-1)
        pki_pred = self.regressor(z).squeeze(-1)

        return {
            "binding_logit": binding_logit,
            "binding_prob": torch.sigmoid(binding_logit),
            "pki_pred": pki_pred,
        }

    def compute_loss(
        self,
        anchor_esm2: torch.Tensor,
        query_esm2: torch.Tensor,
        drug_indices: torch.Tensor,
        pki_targets: torch.Tensor,
        binding_labels: torch.Tensor | None = None,
        binding_mask: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task loss: BCE (binary) + α * MSE (regression)."""
        out = self.forward(anchor_esm2, query_esm2, drug_indices)

        mse_loss = F.mse_loss(out["pki_pred"], pki_targets)

        if binding_mask is not None and binding_mask.any():
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"][binding_mask],
                binding_labels[binding_mask].float(),
            )
        elif binding_labels is not None and binding_mask is None:
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"],
                binding_labels.float(),
            )
        else:
            bce_loss = torch.tensor(0.0, device=pki_targets.device)

        total_loss = bce_loss + alpha * mse_loss

        out["loss"] = total_loss
        out["bce_loss"] = bce_loss
        out["mse_loss"] = mse_loss
        return out
