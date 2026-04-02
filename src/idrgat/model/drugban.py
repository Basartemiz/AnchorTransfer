"""DrugBAN-style pairwise baseline for DTA regression.

This keeps the original high-level idea:
  1. drug graph encoder -> per-atom embeddings
  2. protein CNN encoder -> per-residue embeddings
  3. bilinear attention over atom/residue pairs
  4. fused representation -> pKi regression

The implementation is intentionally lightweight so it fits the project's
existing benchmark harness without introducing a separate training stack.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch

try:
    from idr_gat.model.drug_encoder import ATOM_FEATURE_DIM
except ImportError:
    # Older remote checkout still exposes the graph encoder under the
    # GraphDTA-specific module name.
    ATOM_FEATURE_DIM = 78


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Softmax that respects padding masks and stays numerically stable."""
    logits = logits.masked_fill(~mask, -1e4)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class ProteinCNNEncoder(nn.Module):
    """Character-level protein encoder that preserves residue-wise features."""

    def __init__(self, vocab_size: int = 26, embed_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mask = tokens.ne(0)
        x = self.embedding(tokens).transpose(1, 2)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x)).transpose(1, 2)
        return self.norm(x), mask


class DrugGraphEncoder(nn.Module):
    """GIN encoder that returns per-atom embeddings instead of pooled graphs."""

    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        mlp = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        dense_x, mask = to_dense_batch(x, batch)
        return self.norm(dense_x), mask


class DrugBANModel(nn.Module):
    """DrugBAN-style bilinear attention model adapted for pKi regression."""

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.drug_encoder = DrugGraphEncoder(hidden_dim=hidden_dim)
        self.protein_encoder = ProteinCNNEncoder(hidden_dim=hidden_dim)

        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear)

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.regressor = nn.Linear(256, 1)

    def forward(self, graph_batch, protein_tokens: torch.Tensor) -> torch.Tensor:
        drug_nodes, drug_mask = self.drug_encoder(graph_batch)
        prot_nodes, prot_mask = self.protein_encoder(protein_tokens)

        scores = torch.einsum("bid,dh,bjh->bij", drug_nodes, self.bilinear, prot_nodes)
        scores = scores / math.sqrt(drug_nodes.size(-1))

        pair_mask = drug_mask.unsqueeze(2) & prot_mask.unsqueeze(1)
        scores = scores.masked_fill(~pair_mask, -1e4)

        drug_logits = scores.max(dim=2).values
        prot_logits = scores.max(dim=1).values

        drug_attn = masked_softmax(drug_logits, drug_mask, dim=1)
        prot_attn = masked_softmax(prot_logits, prot_mask, dim=1)

        drug_ctx = torch.einsum("bn,bnd->bd", drug_attn, drug_nodes)
        prot_ctx = torch.einsum("bm,bmd->bd", prot_attn, prot_nodes)

        fused = torch.cat(
            [drug_ctx, prot_ctx, torch.abs(drug_ctx - prot_ctx), drug_ctx * prot_ctx],
            dim=1,
        )
        fused = self.fusion(fused)
        return self.regressor(fused).squeeze(-1)
