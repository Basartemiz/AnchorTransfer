"""ConciseAnchor: Anchor transfer learning with CoNCISE backbone.

Combines CoNCISE's FSQ drug quantization + cross-attention with anchor
transfer. Compares post-cross-attention drug embeddings between anchor
and query proteins to predict binding affinity.

Architecture:
  1. Drug: Morgan FP (2048d) → FSQ → 3 codes × drug_dim
  2. Project drug + proteins to shared dim
  3. Self-attention on drug codes and protein residues (shared)
  4. Cross-attention drug↔anchor and drug↔query (shared weights)
  5. Compare post-cross-attention drug embeddings: [anc, qry, |diff|, prod]
  6. Pool protein contexts, concat all → MLP → pKi
"""

from __future__ import annotations

import torch
import torch.nn as nn
from concise.model.concise import Concise


class ConciseAnchor(nn.Module):
    """CoNCISE backbone with anchor transfer comparison."""

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

        # Use CoNCISE backbone for shared encoders + attention
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
        drug_flat_dim = n_drug_codes * proj_dim  # 3 * 256 = 768

        # Comparison + protein context → regression
        # Drug comparison: 4 * 768 = 3072
        # Protein contexts: 2 * 256 = 512
        # Total: 3584
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

    def _encode_interaction(
        self, drug_fp: torch.Tensor, prot_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run CoNCISE pipeline up through cross-attention.

        Returns the post-cross-attention drug codes (flattened) and
        pooled protein representation.
        """
        bb = self.backbone

        # Drug encoding (FSQ) — returns dict with 'codes', 'emb', 'points'
        d_out = bb.d_encoder(drug_fp)
        d_emb = d_out['emb']  # (B, n_codes, drug_dim)

        # Project to shared space
        d_emb = bb.d_project(d_emb)        # (B, n_codes, proj_dim)
        r_emb = bb.r_project(prot_emb)     # (B, 50, proj_dim)

        # Self-attention (returns (output, attn_weights) tuples)
        d_d_mixed, _ = bb.d_to_d_attention(d_emb)
        r_r_mixed, _ = bb.r_to_r_attention(r_emb)
        d_emb = d_emb + d_d_mixed
        r_emb = r_emb + r_r_mixed

        # Cross-attention (drug ↔ protein)
        # No residual on drug side — force fully protein-conditioned representation
        r_d_mixed, _ = bb.r_to_d_attention(r_emb, d_emb, d_emb)
        d_r_mixed, _ = bb.d_to_r_attention(d_emb, r_emb, r_emb)
        d_emb = d_r_mixed          # NOT d_emb + d_r_mixed
        r_emb = r_emb + r_d_mixed

        # Pool protein: softmax-weighted sum
        r_wt = torch.softmax(10.0 * r_emb, dim=1)
        r_pooled = (r_emb * r_wt).sum(dim=1)  # (B, proj_dim)

        # Flatten drug codes
        B = d_emb.size(0)
        d_flat = d_emb.reshape(B, -1)  # (B, n_codes * proj_dim)

        return d_flat, r_pooled

    def forward(
        self,
        drug_fp: torch.Tensor,
        anchor_emb: torch.Tensor,
        query_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        drug_fp : (B, 2048) float — Morgan fingerprint
        anchor_emb : (B, 50, 1280) float — Raygun embedding for anchor protein
        query_emb : (B, 50, 1280) float — Raygun embedding for query protein

        Returns
        -------
        (B,) predicted pKi
        """
        # Run shared pipeline for both proteins
        anc_drug_flat, anc_prot = self._encode_interaction(drug_fp, anchor_emb)
        qry_drug_flat, qry_prot = self._encode_interaction(drug_fp, query_emb)

        # Compare drug interaction patterns
        diff = torch.abs(anc_drug_flat - qry_drug_flat)
        product = anc_drug_flat * qry_drug_flat

        # Fuse everything
        fused = torch.cat([
            anc_drug_flat, qry_drug_flat, diff, product,
            anc_prot, qry_prot,
        ], dim=-1)

        return self.regressor(fused).squeeze(-1)
