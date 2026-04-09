"""Anchor Transfer DTA with parameter-matched latent cross-attention.

This keeps the V2 triplet formulation:
  (anchor_protein, query_protein, drug) -> binding logit + pKi

Relative to the original V2 model, the shallow pairwise MLP interactions are
replaced by latent-token cross-attention blocks. Each entity embedding is first
expanded into a small set of learned latent tokens, and attention is performed
across those tokens so the interaction stage has meaningful structure despite
the upstream inputs being single vectors.

To keep the comparison fair, the interaction budget is kept close to V2 by
sharing one latent attention module across the two protein-drug relations and
using a second module for the protein-protein relation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from anchor_transfer.model.anchor_transfer import CHARISOSMISET, SMILES_MAX_LEN, encode_smiles


class SMILESEncoder(nn.Module):
    """Parallel CNN encoder for SMILES strings."""

    def __init__(
        self,
        vocab_size: int = 52,
        embed_dim: int = 128,
        num_filters: int = 32,
        output_dim: int = 256,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, kernel) for kernel in (4, 6, 8)]
        )
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, smiles_indices: torch.Tensor) -> torch.Tensor:
        x = self.embed(smiles_indices).permute(0, 2, 1)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=1))


class LatentCrossAttentionInteraction(nn.Module):
    """Cross-attention interaction over a small learned token set."""

    def __init__(
        self,
        dim: int = 256,
        token_dim: int = 72,
        num_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim

        self.x_tokens = nn.Linear(dim, num_tokens * token_dim)
        self.y_tokens = nn.Linear(dim, num_tokens * token_dim)

        self.attn_xy = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attn_yx = nn.MultiheadAttention(
            token_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm_x = nn.LayerNorm(token_dim)
        self.norm_y = nn.LayerNorm(token_dim)
        self.dropout = nn.Dropout(dropout)

        self.fuse = nn.Sequential(
            nn.Linear(token_dim * 2, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.ReLU(),
        )

    def _to_tokens(self, proj: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        return proj(x).view(x.size(0), self.num_tokens, self.token_dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x_tok = self._to_tokens(self.x_tokens, x)
        y_tok = self._to_tokens(self.y_tokens, y)

        x_att, _ = self.attn_xy(x_tok, y_tok, y_tok, need_weights=False)
        y_att, _ = self.attn_yx(y_tok, x_tok, x_tok, need_weights=False)

        x_out = self.norm_x(x_tok + self.dropout(x_att))
        y_out = self.norm_y(y_tok + self.dropout(y_att))

        pooled = torch.cat([x_out.mean(dim=1), y_out.mean(dim=1)], dim=-1)
        return self.fuse(pooled)


class AnchorTransferLatentAttn(nn.Module):
    """V2-style anchor transfer model with parameter-matched attention blocks."""

    def __init__(
        self,
        esm2_dim: int = 480,
        proj_dim: int = 256,
        dropout: float = 0.3,
        smiles_vocab: int = 52,
        smiles_embed_dim: int = 128,
        smiles_filters: int = 32,
        token_dim: int = 72,
        num_tokens: int = 4,
        num_heads: int = 4,
    ):
        super().__init__()

        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, proj_dim),
            nn.ReLU(),
            nn.LayerNorm(proj_dim),
        )
        self.drug_encoder = SMILESEncoder(
            vocab_size=smiles_vocab,
            embed_dim=smiles_embed_dim,
            num_filters=smiles_filters,
            output_dim=proj_dim,
        )

        # Share the protein-drug attention block between anchor-drug and
        # query-drug so the total parameter budget stays close to V2.
        self.protein_drug_interaction = LatentCrossAttentionInteraction(
            dim=proj_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.protein_pair_interaction = LatentCrossAttentionInteraction(
            dim=proj_dim,
            token_dim=token_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
            dropout=dropout,
        )

        fused_dim = 6 * proj_dim

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
        a = self.protein_proj(anchor_esm2)
        q = self.protein_proj(query_esm2)
        d = self.drug_encoder(drug_indices)

        h_ad = self.protein_drug_interaction(a, d)
        h_qd = self.protein_drug_interaction(q, d)
        h_aq = self.protein_pair_interaction(a, q)

        z = torch.cat([a, q, d, h_ad, h_qd, h_aq], dim=-1)
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

        out["loss"] = bce_loss + alpha * mse_loss
        out["bce_loss"] = bce_loss
        out["mse_loss"] = mse_loss
        return out
