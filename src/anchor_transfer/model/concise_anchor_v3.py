"""ConciseAnchor v3: Conditional drug encoding + bilinear interaction.

Combines two mechanisms to maximize anchor information flow:
  B) Conditional drug encoding: anchor protein conditions the drug FSQ
     quantization so drug codes are anchor-dependent
  C) Bilinear interaction: shared bilinear weight W computes per-code
     binding patterns for anchor and query (replaces cross-attention)

Architecture:
  1. Pool anchor protein: Raygun (50×1280) → project → pool → (256,)
  2. Conditional drug encoding: [drug_fp, anchor_pooled] → MLP → FSQ → codes
  3. Project drug codes + both proteins to shared dim
  4. Bilinear attention: drug_codes @ W @ protein^T → per-code binding patterns
  5. Compare: [anc_binding, qry_binding, |diff|, product] per code
  6. Fuse + pool → regress → pKi
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RaygunPooler(nn.Module):
    """Project and pool Raygun embeddings (50×1280) → (proj_dim,)."""

    def __init__(self, residue_dim: int = 1280, proj_dim: int = 256):
        super().__init__()
        self.project = nn.Linear(residue_dim, proj_dim)
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (projected_residues (B,50,d), pooled (B,d))."""
        h = self.norm(F.gelu(self.project(x)))  # (B, 50, d)
        wt = torch.softmax(10.0 * h, dim=1)
        pooled = (h * wt).sum(dim=1)  # (B, d)
        return h, pooled


class ConditionalDrugEncoder(nn.Module):
    """Drug FP → anchor-conditioned embedding codes.

    Takes [drug_fp, anchor_pooled] as input so the drug representation
    is anchor-dependent from the start.
    """

    def __init__(self, ligand_dim: int = 2048, cond_dim: int = 256,
                 hidden_dim: int = 256, n_codes: int = 3):
        super().__init__()
        self.n_codes = n_codes
        # Conditional input: drug FP + anchor pooled
        self.encoder = nn.Sequential(
            nn.Linear(ligand_dim + cond_dim, 512),
            nn.GELU(),
            nn.Linear(512, hidden_dim * n_codes),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, drug_fp: torch.Tensor, anchor_pooled: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_codes, hidden_dim)."""
        x = torch.cat([drug_fp, anchor_pooled], dim=-1)  # (B, 2048+256)
        h = self.encoder(x)  # (B, n_codes * hidden_dim)
        B = h.size(0)
        h = h.view(B, self.n_codes, -1)  # (B, n_codes, hidden_dim)
        return self.norm(h)


def masked_softmax(logits: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.softmax(logits, dim=dim)


class ConciseAnchorV3(nn.Module):
    """Conditional drug encoding + bilinear anchor comparison."""

    def __init__(
        self,
        ligand_dim: int = 2048,
        residue_dim: int = 1280,
        proj_dim: int = 256,
        n_codes: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.protein_encoder = RaygunPooler(residue_dim, proj_dim)
        self.drug_encoder = ConditionalDrugEncoder(
            ligand_dim=ligand_dim, cond_dim=proj_dim,
            hidden_dim=proj_dim, n_codes=n_codes,
        )

        # Shared bilinear weight for drug-code × protein-residue interaction
        self.bilinear = nn.Parameter(torch.empty(proj_dim, proj_dim))
        nn.init.xavier_uniform_(self.bilinear)

        # Per-code comparison fusion: 4 * proj_dim → proj_dim
        self.code_fusion = nn.Sequential(
            nn.Linear(proj_dim * 4, proj_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim * 2, proj_dim),
            nn.GELU(),
        )

        # Final regression: pooled codes + protein contexts
        # n_codes * proj_dim (pooled comparison) + 2 * proj_dim (protein pools)
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

    def _bilinear_binding(
        self, drug_codes: torch.Tensor, prot_residues: torch.Tensor
    ) -> torch.Tensor:
        """Compute per-code binding patterns via bilinear attention.

        Args:
            drug_codes: (B, n_codes, d)
            prot_residues: (B, 50, d)

        Returns:
            (B, n_codes, d) — per-code protein binding context
        """
        # Bilinear scores: (B, n_codes, 50)
        scores = torch.einsum("bid,dh,bjh->bij", drug_codes, self.bilinear, prot_residues)
        scores = scores / math.sqrt(drug_codes.size(-1))

        # Attention over residues for each code
        attn = torch.softmax(scores, dim=2)  # (B, n_codes, 50)

        # Per-code binding context
        binding = torch.bmm(attn, prot_residues)  # (B, n_codes, d)
        return binding

    def forward(
        self,
        drug_fp: torch.Tensor,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        drug_fp : (B, 2048) — Morgan fingerprint
        anchor_emb : (B, 50, 1280) — Raygun embedding for anchor
        query_emb : (B, 50, 1280) — Raygun embedding for query

        Returns
        -------
        (B,) predicted pKi
        """
        # Encode proteins
        anc_residues, anc_pooled = self.protein_encoder(anchor_emb)
        qry_residues, qry_pooled = self.protein_encoder(query_emb)

        # Conditional drug encoding (anchor-dependent)
        drug_codes = self.drug_encoder(drug_fp, anc_pooled)  # (B, n_codes, d)

        # Bilinear binding patterns (shared W)
        anc_binding = self._bilinear_binding(drug_codes, anc_residues)  # (B, n_codes, d)
        qry_binding = self._bilinear_binding(drug_codes, qry_residues)  # (B, n_codes, d)

        # Per-code comparison
        diff = torch.abs(anc_binding - qry_binding)
        product = anc_binding * qry_binding
        comparison = torch.cat([anc_binding, qry_binding, diff, product], dim=-1)  # (B, n_codes, 4d)

        # Fuse per-code
        fused_codes = self.code_fusion(comparison)  # (B, n_codes, d)

        # Pool codes + protein contexts
        B = drug_codes.size(0)
        code_flat = fused_codes.reshape(B, -1)  # (B, n_codes * d)
        full = torch.cat([code_flat, anc_pooled, qry_pooled], dim=-1)

        return self.regressor(full).squeeze(-1)
