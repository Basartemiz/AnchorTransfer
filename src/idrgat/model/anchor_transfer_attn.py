"""Anchor Transfer DTA with Pairwise Cross-Attention.

Same triplet formulation as V2 (anchor_protein, query_protein, drug) → pKi,
but replaces pairwise MLPs with cross-attention between each pair.

Each pairwise interaction uses multi-head cross-attention:
  - anchor attends to drug, drug attends to anchor
  - query attends to drug, drug attends to query
  - anchor attends to query, query attends to anchor
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from idr_gat.model.anchor_transfer import CHARISOSMISET, SMILES_MAX_LEN, encode_smiles


class SMILESEncoder(nn.Module):
    def __init__(self, vocab_size=52, embed_dim=128, num_filters=32, output_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]])
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=1))


class PairwiseCrossAttention(nn.Module):
    """Bidirectional cross-attention between two embeddings."""

    def __init__(self, dim=256, num_heads=4, dropout=0.3):
        super().__init__()
        self.attn_xy = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_yx = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, y):
        # x, y are (B, D) — unsqueeze to (B, 1, D) for attention
        x_seq = x.unsqueeze(1)
        y_seq = y.unsqueeze(1)

        # x attends to y
        x_att, _ = self.attn_xy(x_seq, y_seq, y_seq)
        x_out = self.norm_x(x_seq + x_att).squeeze(1)

        # y attends to x
        y_att, _ = self.attn_yx(y_seq, x_seq, x_seq)
        y_out = self.norm_y(y_seq + y_att).squeeze(1)

        # Combine
        return self.ffn(torch.cat([x_out, y_out], dim=-1))


class AnchorTransferAttn(nn.Module):
    """Anchor transfer model with pairwise cross-attention."""

    def __init__(self, esm2_dim=480, proj_dim=256, num_heads=4, dropout=0.3,
                 smiles_vocab=52, smiles_embed_dim=128, smiles_filters=32):
        super().__init__()

        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, proj_dim), nn.ReLU(), nn.LayerNorm(proj_dim),
        )
        self.drug_encoder = SMILESEncoder(smiles_vocab, smiles_embed_dim, smiles_filters, proj_dim)

        self.attn_anchor_drug = PairwiseCrossAttention(proj_dim, num_heads, dropout)
        self.attn_query_drug = PairwiseCrossAttention(proj_dim, num_heads, dropout)
        self.attn_anchor_query = PairwiseCrossAttention(proj_dim, num_heads, dropout)

        fused_dim = 6 * proj_dim

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

    def forward(self, anchor_esm2, query_esm2, drug_indices):
        a = self.protein_proj(anchor_esm2)
        q = self.protein_proj(query_esm2)
        d = self.drug_encoder(drug_indices)

        h_ad = self.attn_anchor_drug(a, d)
        h_qd = self.attn_query_drug(q, d)
        h_aq = self.attn_anchor_query(a, q)

        z = torch.cat([a, q, d, h_ad, h_qd, h_aq], dim=-1)

        binding_logit = self.classifier(z).squeeze(-1)
        pki_pred = self.regressor(z).squeeze(-1)

        return {
            "binding_logit": binding_logit,
            "binding_prob": torch.sigmoid(binding_logit),
            "pki_pred": pki_pred,
        }

    def compute_loss(self, anchor_esm2, query_esm2, drug_indices,
                     pki_targets, binding_labels=None, binding_mask=None, alpha=1.0):
        out = self.forward(anchor_esm2, query_esm2, drug_indices)
        mse_loss = F.mse_loss(out["pki_pred"], pki_targets)

        if binding_mask is not None and binding_mask.any():
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"][binding_mask], binding_labels[binding_mask].float())
        elif binding_labels is not None and binding_mask is None:
            bce_loss = F.binary_cross_entropy_with_logits(
                out["binding_logit"], binding_labels.float())
        else:
            bce_loss = torch.tensor(0.0, device=pki_targets.device)

        out["loss"] = bce_loss + alpha * mse_loss
        out["bce_loss"] = bce_loss
        out["mse_loss"] = mse_loss
        return out
