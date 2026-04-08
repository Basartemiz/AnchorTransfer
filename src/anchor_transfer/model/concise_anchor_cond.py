"""ConciseAnchor-Cond: Conditional drug encoding with CoNCISE cross-attention.

Approach B: anchor protein conditions the drug encoding so drug codes
are anchor-dependent. Uses CoNCISE's original cross-attention mechanism.

Architecture:
  1. Pool anchor: Raygun → project → pool → (256,)
  2. Conditional drug: [drug_fp, anchor_pooled] → MLP → 3 codes × 256d
  3. Self-attention + cross-attention (CoNCISE backbone, no residual on drug)
  4. Compare post-cross-attention drug embeddings: [anc, qry, |diff|, prod]
  5. Fuse → regress → pKi
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from concise.model.concise import Concise


class ConciseAnchorCond(nn.Module):
    """Conditional drug encoding + CoNCISE cross-attention."""

    def __init__(
        self,
        drug_layers: list[list[int]] | None = None,
        ligand_dim: int = 2048,
        residue_dim: int = 1280,
        proj_dim: int = 256,
        nheads: int = 32,
        dropout: float = 0.2,
    ):
        super().__init__()
        if drug_layers is None:
            drug_layers = [[32], [32], [32]]

        self.backbone = Concise(
            drug_layers=drug_layers,
            ligand_dim=ligand_dim,
            residue_dim=residue_dim,
            drug_dim=proj_dim,
            proj_dim=proj_dim,
            nheads=nheads,
            activation="gelu",
            cosine_prediction=False,
        )

        n_drug_codes = len(drug_layers)
        self.n_codes = n_drug_codes

        # Anchor pooler: project + pool Raygun embeddings
        self.anchor_project = nn.Linear(residue_dim, proj_dim)
        self.anchor_norm = nn.LayerNorm(proj_dim)

        # Conditional drug encoder: [drug_fp, anchor_pooled] → codes
        self.cond_encoder = nn.Sequential(
            nn.Linear(ligand_dim + proj_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim * n_drug_codes),
        )
        self.cond_norm = nn.LayerNorm(proj_dim)

        # Comparison + protein contexts → regression
        drug_flat_dim = n_drug_codes * proj_dim
        fused_dim = drug_flat_dim * 4 + proj_dim * 2
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.regressor[-1].bias, 6.5)

    def _pool_anchor(self, anchor_emb: torch.Tensor) -> torch.Tensor:
        """Pool Raygun anchor embeddings to a single vector."""
        h = self.anchor_norm(F.gelu(self.anchor_project(anchor_emb)))  # (B, 50, d)
        wt = torch.softmax(10.0 * h, dim=1)
        return (h * wt).sum(dim=1)  # (B, d)

    def _encode_drug_conditioned(self, drug_fp: torch.Tensor, anchor_pooled: torch.Tensor) -> torch.Tensor:
        """Anchor-conditioned drug encoding."""
        x = torch.cat([drug_fp, anchor_pooled], dim=-1)
        h = self.cond_encoder(x)
        B = h.size(0)
        return self.cond_norm(h.view(B, self.n_codes, -1))  # (B, n_codes, d)

    def _run_attention(self, d_emb: torch.Tensor, prot_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CoNCISE self-attention + cross-attention. No residual on drug side."""
        bb = self.backbone
        r_emb = bb.r_project(prot_emb)

        # Self-attention
        d_d_mixed, _ = bb.d_to_d_attention(d_emb)
        r_r_mixed, _ = bb.r_to_r_attention(r_emb)
        d_emb = d_emb + d_d_mixed
        r_emb = r_emb + r_r_mixed

        # Cross-attention — no residual on drug side
        r_d_mixed, _ = bb.r_to_d_attention(r_emb, d_emb, d_emb)
        d_r_mixed, _ = bb.d_to_r_attention(d_emb, r_emb, r_emb)
        d_emb = d_r_mixed  # fully protein-conditioned
        r_emb = r_emb + r_d_mixed

        # Pool protein
        r_wt = torch.softmax(10.0 * r_emb, dim=1)
        r_pooled = (r_emb * r_wt).sum(dim=1)

        B = d_emb.size(0)
        d_flat = d_emb.reshape(B, -1)
        return d_flat, r_pooled

    def forward(self, drug_fp: torch.Tensor, anchor_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        # Pool anchor for conditioning
        anc_pooled = self._pool_anchor(anchor_emb)

        # Conditional drug encoding (anchor-dependent codes)
        drug_codes = self._encode_drug_conditioned(drug_fp, anc_pooled)

        # Run cross-attention with anchor and query
        anc_drug_flat, anc_prot = self._run_attention(drug_codes, anchor_emb)
        qry_drug_flat, qry_prot = self._run_attention(drug_codes, query_emb)

        # Compare
        diff = torch.abs(anc_drug_flat - qry_drug_flat)
        product = anc_drug_flat * qry_drug_flat
        fused = torch.cat([anc_drug_flat, qry_drug_flat, diff, product, anc_prot, qry_prot], dim=-1)

        return self.regressor(fused).squeeze(-1)
