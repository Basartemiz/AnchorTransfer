"""CoNCISE model adapted for DTA regression.

Based on the original Concise architecture (Erden et al., RECOMB 2025) — the
paper-reproducible version — with the Sigmoid/Cosine prediction head replaced
by a regression MLP for pKi prediction.

Uses the backbone's forward directly. The only change is the final head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from concise.model.concise import Concise


class ConciseDTA(nn.Module):
    """Concise backbone (paper version) with regression head for pKi."""

    def __init__(
        self,
        esm_dim: int = 480,
        drug_layers: list[list[int]] | None = None,
        proj_dim: int = 256,
        nheads: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        if drug_layers is None:
            drug_layers = [[32], [32], [32]]

        self.backbone = Concise(
            drug_layers=drug_layers,
            ligand_dim=2048,
            residue_dim=esm_dim,
            drug_dim=proj_dim,
            proj_dim=proj_dim,
            nheads=nheads,
            activation="gelu",
            cosine_prediction=False,
        )

        # Replace the backbone's sigmoid classifier with regression head
        n_drug_codes = len(drug_layers)
        fused_dim = n_drug_codes * proj_dim + proj_dim
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, drug_fp: torch.Tensor, prot_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        drug_fp : (B, 2048) float — Morgan fingerprint
        prot_emb : (B, 50, esm_dim) float — ESM-2 embedding (repeated)

        Returns
        -------
        (B,) predicted pKi
        """
        out = self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)
        return out["binding"]
