"""ConciseAnchor-Bilinear: Raygun backbone + bilinear anchor comparison.

Approach C: Replace CoNCISE's cross-attention with bilinear attention
(like AnchorDrugBAN) but keep Raygun protein embeddings.
Normal drug encoding (not conditioned on anchor).

Architecture:
  1. Drug: Morgan FP → MLP → 3 codes × 256d (standard, not conditional)
  2. Proteins: Raygun → project → per-residue embeddings (50 × 256d)
  3. Shared bilinear W: drug_codes × residues → per-code binding patterns
  4. Compare: [anc_binding, qry_binding, |diff|, product]
  5. Fuse per-code → pool → regress → pKi
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConciseAnchorBilinear(nn.Module):
    """Raygun backbone with bilinear anchor comparison."""

    def __init__(
        self,
        ligand_dim: int = 2048,
        residue_dim: int = 1280,
        proj_dim: int = 256,
        n_codes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Drug encoder: FP → codes (no anchor conditioning)
        self.drug_encoder = nn.Sequential(
            nn.Linear(ligand_dim, 512),
            nn.GELU(),
            nn.Linear(512, proj_dim * n_codes),
        )
        self.drug_norm = nn.LayerNorm(proj_dim)
        self.n_codes = n_codes

        # Protein encoder: project Raygun residues
        self.protein_project = nn.Linear(residue_dim, proj_dim)
        self.protein_norm = nn.LayerNorm(proj_dim)

        # Shared bilinear weight for code-residue interaction
        self.bilinear = nn.Parameter(torch.empty(proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.bilinear)

        # Per-code comparison: 4d → d
        self.code_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
        )

        # Final regression: pooled codes + protein pools
        fused_dim = n_codes * proj_dim + 2 * proj_dim
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.regressor[-1].bias, 6.5)

    def _encode_drug(self, drug_fp: torch.Tensor) -> torch.Tensor:
        h = self.drug_encoder(drug_fp)
        B = h.size(0)
        return self.drug_norm(h.view(B, self.n_codes, -1))

    def _encode_protein(self, prot_emb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.protein_norm(F.gelu(self.protein_project(prot_emb)))  # (B, 50, d)
        wt = torch.softmax(10.0 * h, dim=1)
        pooled = (h * wt).sum(dim=1)
        return h, pooled

    def _bilinear_binding(self, drug_codes: torch.Tensor, prot_residues: torch.Tensor) -> torch.Tensor:
        """Per-code binding pattern via bilinear attention over residues."""
        scores = torch.einsum("bid,dh,bjh->bij", drug_codes, self.bilinear, prot_residues)
        scores = scores / math.sqrt(drug_codes.size(-1))
        attn = torch.softmax(scores, dim=2)  # (B, n_codes, 50)
        return torch.bmm(attn, prot_residues)  # (B, n_codes, d)

    def forward(self, drug_fp: torch.Tensor, anchor_emb: torch.Tensor, query_emb: torch.Tensor) -> torch.Tensor:
        # Encode drug (same for both)
        drug_codes = self._encode_drug(drug_fp)  # (B, n_codes, d)

        # Encode proteins
        anc_res, anc_pooled = self._encode_protein(anchor_emb)
        qry_res, qry_pooled = self._encode_protein(query_emb)

        # Bilinear binding patterns (shared W)
        anc_binding = self._bilinear_binding(drug_codes, anc_res)
        qry_binding = self._bilinear_binding(drug_codes, qry_res)

        # Per-code comparison
        diff = torch.abs(anc_binding - qry_binding)
        product = anc_binding * qry_binding
        comparison = torch.cat([anc_binding, qry_binding, diff, product], dim=-1)
        fused_codes = self.code_fusion(comparison)  # (B, n_codes, d)

        # Pool + protein contexts → regress
        B = drug_codes.size(0)
        code_flat = fused_codes.reshape(B, -1)
        full = torch.cat([code_flat, anc_pooled, qry_pooled], dim=-1)

        return self.regressor(full).squeeze(-1)
