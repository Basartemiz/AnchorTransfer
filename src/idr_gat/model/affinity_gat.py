"""AffinityGAT: full-graph GATv2 + SMILES CNN + cross-attention for DTA prediction."""
from __future__ import annotations

import torch
# Disable cuDNN for Conv1d compatibility on some GPU configs
torch.backends.cudnn.enabled = False
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv

# SMILES character encoding (matches DeepDTA fair checkpoint)
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
    return [CHARISOSMISET.get(c, 0) for c in smi[:max_len]] + [0] * max(0, max_len - len(smi))


class SMILESEncoder(nn.Module):
    """DeepDTA-style parallel CNN encoder for SMILES strings."""

    def __init__(self, vocab_size: int = 52, embed_dim: int = 128,
                 num_filters: int = 32, output_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, smiles_indices: torch.Tensor) -> torch.Tensor:
        """Args: smiles_indices (B, max_len) int. Returns: (B, output_dim)."""
        x = self.embed(smiles_indices).permute(0, 2, 1)  # (B, embed, L)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]  # 3 × (B, 32)
        x = torch.cat(outs, dim=1)  # (B, 96)
        return self.proj(x)  # (B, output_dim)


class CrossAttention(nn.Module):
    """Bidirectional cross-attention between protein and drug embeddings."""

    def __init__(self, dim: int = 512, n_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.drug_to_prot = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.prot_to_drug = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, prot_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        """Args: prot_emb (B, dim), drug_emb (B, dim). Returns: (B, dim)."""
        p = prot_emb.unsqueeze(1)
        d = drug_emb.unsqueeze(1)
        d2p, _ = self.drug_to_prot(d, p, p)
        p2d, _ = self.prot_to_drug(p, d, d)
        out = d2p.squeeze(1) + p2d.squeeze(1)
        return self.norm(out)


class AffinityGAT(nn.Module):
    """Full-graph GATv2 + SMILES CNN + cross-attention for affinity prediction.

    Forward pass:
      1. Build node features from 3Di + ESM-2 embeddings
      2. GATv2 message passing on full graph (4 layers)
      3. Read out anchor node embedding
      4. Encode drug SMILES via CNN
      5. Cross-attention between protein and drug
      6. MLP → predicted pKi
    """

    def __init__(
        self,
        threedi_vocab_size: int = 21,
        threedi_embed_dim: int = 64,
        esm2_input_dim: int = 480,
        esm2_proj_dim: int = 256,
        hidden_dim: int = 512,
        gat_layers: int = 4,
        gat_heads: int = 8,
        dropout: float = 0.3,
        smiles_vocab: int = 52,
        smiles_embed_dim: int = 128,
        smiles_filters: int = 32,
        cross_attn_heads: int = 4,
    ):
        super().__init__()
        self.threedi_embed = nn.Embedding(threedi_vocab_size, threedi_embed_dim, padding_idx=20)
        self.esm2_proj = nn.Sequential(
            nn.Linear(esm2_input_dim, esm2_proj_dim),
            nn.LayerNorm(esm2_proj_dim),
            nn.GELU(),
        )
        gat_input_dim = threedi_embed_dim + esm2_proj_dim

        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        for i in range(gat_layers):
            in_dim = gat_input_dim if i == 0 else hidden_dim
            self.gat_convs.append(
                GATv2Conv(in_dim, hidden_dim // gat_heads, heads=gat_heads,
                          edge_dim=1, concat=True, dropout=dropout)
            )
            self.gat_norms.append(nn.LayerNorm(hidden_dim))
        self.gat_dropout = nn.Dropout(dropout)

        self.drug_encoder = SMILESEncoder(
            vocab_size=smiles_vocab, embed_dim=smiles_embed_dim,
            num_filters=smiles_filters, output_dim=hidden_dim,
        )

        self.cross_attn = CrossAttention(dim=hidden_dim, n_heads=cross_attn_heads, dropout=dropout)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def build_node_features(self, data) -> torch.Tensor:
        """Build per-node features from 3Di tokens + ESM-2 embeddings."""
        x_3di = data.x_3di
        seq_lens = data.x_seq_lens
        embedded = self.threedi_embed(x_3di)
        mask = torch.arange(x_3di.size(1), device=x_3di.device).unsqueeze(0) < seq_lens.unsqueeze(1)
        embedded = embedded * mask.unsqueeze(-1).float()
        threedi_feat = embedded.sum(dim=1) / seq_lens.clamp(min=1).unsqueeze(-1).float()

        esm2_feat = self.esm2_proj(data.x_esm2)

        return torch.cat([threedi_feat, esm2_feat], dim=-1)

    def encode_graph(self, data) -> torch.Tensor:
        """Run GATv2 on full graph, return per-node embeddings.

        Uses gradient checkpointing to reduce GPU memory for large graphs.
        """
        from torch.utils.checkpoint import checkpoint

        x = self.build_node_features(data)
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        for conv, norm in zip(self.gat_convs, self.gat_norms):
            def _layer_fn(x_in, _ei=edge_index, _ea=edge_attr, _conv=conv, _norm=norm):
                x_out = _conv(x_in, _ei, _ea)
                x_out = _norm(x_out)
                x_out = F.elu(x_out)
                return x_out

            if self.training and x.requires_grad:
                x_new = checkpoint(_layer_fn, x, use_reentrant=False)
            else:
                x_new = _layer_fn(x)

            x_new = self.gat_dropout(x_new)
            if x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        return x

    def forward(
        self,
        graph_data,
        anchor_indices: torch.Tensor,
        smiles_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            graph_data: PyG Data with full graph (can be on CPU — moved to GPU internally)
            anchor_indices: (B,) int — which node to read out per sample
            smiles_indices: (B, SMILES_MAX_LEN) int — encoded SMILES
        Returns:
            (B,) predicted pKi values
        """
        device = smiles_indices.device
        # Move graph to model's device for forward pass
        graph_on_device = graph_data.to(device) if graph_data.x_3di.device != device else graph_data
        node_embs = self.encode_graph(graph_on_device)
        prot_emb = node_embs[anchor_indices]
        drug_emb = self.drug_encoder(smiles_indices)
        interaction = self.cross_attn(prot_emb, drug_emb)
        return self.head(interaction).squeeze(-1)

    def compute_loss(
        self,
        graph_data,
        anchor_indices: torch.Tensor,
        smiles_indices: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        preds = self.forward(graph_data, anchor_indices, smiles_indices)
        return F.mse_loss(preds, targets)
