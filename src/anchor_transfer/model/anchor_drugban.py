"""Anchor-DrugBAN: Bilinear anchor comparison for DTA.

Combines DrugBAN's atom-residue bilinear attention with anchor transfer.
Instead of comparing pooled vectors, compares atom-level binding patterns
between anchor and query proteins using a shared bilinear weight.

Architecture:
  1. Drug: GIN → per-atom embeddings (A, d)
  2. Anchor protein: CNN → per-residue embeddings (R_a, d)
  3. Query protein: CNN → per-residue embeddings (R_q, d)
  4. Shared bilinear attention W: atoms × residues → binding pattern
     - anchor_binding = softmax(atoms @ W @ R_a^T) @ R_a → (A, d)
     - query_binding  = softmax(atoms @ W @ R_q^T) @ R_q → (A, d)
  5. Per-atom comparison: [anchor_binding, query_binding, |diff|, product]
  6. Atom pooling → MLP → pKi
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv
from torch_geometric.utils import to_dense_batch

try:
    from anchor_transfer.model.drug_encoder import ATOM_FEATURE_DIM
except ImportError:
    ATOM_FEATURE_DIM = 9


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    logits = logits.masked_fill(~mask, -1e4)
    probs = torch.softmax(logits, dim=dim)
    probs = probs * mask.to(probs.dtype)
    return probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-8)


class ProteinCNNEncoder(nn.Module):
    """Character-level protein encoder preserving per-residue features."""

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
    """GIN encoder returning per-atom embeddings."""

    def __init__(self, atom_feature_dim: int = ATOM_FEATURE_DIM,
                 hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        mlp = nn.Sequential(nn.Linear(atom_feature_dim, hidden_dim), nn.ReLU(),
                            nn.Linear(hidden_dim, hidden_dim))
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, graph_batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, edge_index)))
        dense_x, mask = to_dense_batch(x, batch)
        return self.norm(dense_x), mask


class AnchorDrugBAN(nn.Module):
    """Bilinear anchor comparison model for DTA prediction.

    Uses shared bilinear attention to compute per-atom binding patterns
    for both anchor and query proteins, then compares them.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2, binary: bool = False):
        super().__init__()
        self.binary = binary
        self.drug_encoder = DrugGraphEncoder(hidden_dim=hidden_dim)
        self.protein_encoder = ProteinCNNEncoder(hidden_dim=hidden_dim)

        # Shared bilinear weight for atom-residue interaction
        self.bilinear = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        nn.init.xavier_uniform_(self.bilinear)

        # Per-atom comparison fusion: 4 * hidden_dim (anchor, query, |diff|, product)
        self.atom_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )

        # Final regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _bilinear_attention(
        self, drug_nodes: torch.Tensor, drug_mask: torch.Tensor,
        prot_nodes: torch.Tensor, prot_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute drug-side binding pattern via bilinear attention.

        Args:
            drug_nodes: (B, A, d) per-atom drug embeddings
            drug_mask: (B, A) atom padding mask
            prot_nodes: (B, R, d) per-residue protein embeddings
            prot_mask: (B, R) residue padding mask

        Returns:
            (B, A, d) per-atom binding context from the protein
        """
        # Bilinear scores: (B, A, R) = drug @ W @ prot^T
        scores = torch.einsum("bid,dh,bjh->bij", drug_nodes, self.bilinear, prot_nodes)
        scores = scores / math.sqrt(drug_nodes.size(-1))

        # Mask invalid pairs
        pair_mask = drug_mask.unsqueeze(2) & prot_mask.unsqueeze(1)  # (B, A, R)

        # Attention over residues for each atom → per-atom binding context
        attn = masked_softmax(scores, pair_mask, dim=2)  # (B, A, R)
        binding_ctx = torch.bmm(attn, prot_nodes)  # (B, A, d)

        return binding_ctx

    def forward(
        self,
        graph_batch,
        anchor_tokens: torch.Tensor,
        query_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        graph_batch : PyG Batch — molecular graphs for the drug
        anchor_tokens : (B, max_len) int — encoded anchor protein sequences
        query_tokens : (B, max_len) int — encoded query protein sequences

        Returns
        -------
        (B,) predicted pKi
        """
        # Encode drug → per-atom
        drug_nodes, drug_mask = self.drug_encoder(graph_batch)  # (B, A, d), (B, A)

        # Encode proteins → per-residue (shared encoder)
        anchor_nodes, anchor_mask = self.protein_encoder(anchor_tokens)  # (B, R_a, d)
        query_nodes, query_mask = self.protein_encoder(query_tokens)    # (B, R_q, d)

        # Shared bilinear attention: per-atom binding patterns
        anchor_binding = self._bilinear_attention(
            drug_nodes, drug_mask, anchor_nodes, anchor_mask
        )  # (B, A, d) — "how each atom binds the anchor"
        query_binding = self._bilinear_attention(
            drug_nodes, drug_mask, query_nodes, query_mask
        )  # (B, A, d) — "how each atom binds the query"

        # Per-atom comparison
        diff = torch.abs(anchor_binding - query_binding)
        product = anchor_binding * query_binding
        atom_features = torch.cat(
            [anchor_binding, query_binding, diff, product], dim=-1
        )  # (B, A, 4d)

        # Fuse per-atom features
        atom_fused = self.atom_fusion(atom_features)  # (B, A, d)

        # Pool atoms (masked mean)
        atom_fused = atom_fused * drug_mask.unsqueeze(-1).float()
        pooled = atom_fused.sum(dim=1) / drug_mask.sum(dim=1, keepdim=True).clamp_min(1)  # (B, d)

        # Predict pKi
        return self.regressor(pooled).squeeze(-1)
