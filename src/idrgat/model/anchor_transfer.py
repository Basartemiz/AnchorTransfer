"""Anchor Transfer DTA model.

Given (anchor_protein, query_protein, drug) predicts:
  1. Binary binding classification (does query bind the drug?)
  2. Regression pKi affinity score

The anchor is a known binder of the drug. At inference, the anchor is
found via Foldseek structural search against a whole-protein graph.

Architecture:
  - Shared Linear(esm2_dim, 256) projection for both proteins
  - SMILES CNN drug encoder → 128-dim
  - Concat [proj(anchor), proj(query), drug_emb] = 640-dim
  - Binary head: MLP(640 → 512 → 256 → 1) + sigmoid
  - Regression head: MLP(640 → 512 → 256 → 1)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# SMILES character encoding (matches DeepDTA)
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
    "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "[": 50,
    "T": 17, "]": 51, "V": 18, "Y": 19, "c": 20, "e": 21, "l": 22,
    "n": 23, "o": 24, "r": 25, "s": 26, "t": 27, "u": 28,
}
SMILES_MAX_LEN = 100


def encode_smiles(smi: str, max_len: int = SMILES_MAX_LEN) -> list[int]:
    """Encode a SMILES string as a list of integers."""
    return [CHARISOSMISET.get(c, 0) for c in smi[:max_len]] + [0] * max(0, max_len - len(smi))


class SMILESEncoder(nn.Module):
    """DeepDTA-style parallel CNN encoder for SMILES strings."""

    def __init__(self, vocab_size: int = 52, embed_dim: int = 128,
                 num_filters: int = 32, output_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, smiles_indices: torch.Tensor) -> torch.Tensor:
        """Args: smiles_indices (B, max_len) int. Returns: (B, output_dim)."""
        x = self.embed(smiles_indices).permute(0, 2, 1)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(outs, dim=1)
        return self.proj(x)


class AnchorTransferDTA(nn.Module):
    """Anchor-based transfer model for drug-target affinity prediction."""

    def __init__(
        self,
        esm2_dim: int = 480,
        protein_proj_dim: int = 256,
        drug_output_dim: int = 128,
        head_dropout: float = 0.3,
    ):
        super().__init__()
        self.esm2_dim = esm2_dim

        # Shared protein projection
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, protein_proj_dim),
            nn.ReLU(),
            nn.Dropout(head_dropout),
        )

        # Drug encoder (SMILES CNN)
        self.drug_encoder = SMILESEncoder(output_dim=drug_output_dim)

        # Fused dimension: anchor_proj + query_proj + drug_emb
        fused_dim = protein_proj_dim * 2 + drug_output_dim  # 640

        # Binary classification head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 1),
        )

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(head_dropout),
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
        dict with keys:
            'binding_logit': (B,) raw logit for binary classification
            'binding_prob': (B,) sigmoid probability
            'pki_pred': (B,) predicted pKi
        """
        anchor_proj = self.protein_proj(anchor_esm2)
        query_proj = self.protein_proj(query_esm2)
        drug_emb = self.drug_encoder(drug_indices)

        fused = torch.cat([anchor_proj, query_proj, drug_emb], dim=-1)

        binding_logit = self.classifier(fused).squeeze(-1)
        pki_pred = self.regressor(fused).squeeze(-1)

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
        """Compute multi-task loss.

        Parameters
        ----------
        pki_targets : (B,) ground-truth pKi values
        binding_labels : (B,) binary labels (1=binder, 0=non-binder)
        binding_mask : (B,) bool mask — True for samples with binary labels
                       (excludes middle-zone samples from BCE)
        alpha : float, weight for MSE loss relative to BCE

        Returns
        -------
        dict with 'loss', 'bce_loss', 'mse_loss', plus forward outputs
        """
        out = self.forward(anchor_esm2, query_esm2, drug_indices)

        # MSE on all samples
        mse_loss = F.mse_loss(out["pki_pred"], pki_targets)

        # BCE only on samples with binary labels (skip if no valid labels in batch)
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
