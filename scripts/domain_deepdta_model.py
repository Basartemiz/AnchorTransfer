"""Domain-adapted DeepDTA architecture for shorter domain sequences.

Changes from standard DeepDTA:
- DOMAIN_MAX_LEN = 500 (was 1000)
- Protein CNN kernels: [3, 4, 6] (was [8, 8, 8])
- Drug CNN unchanged
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DOMAIN_MAX_LEN = 500
SMILES_MAX_LEN = 100


class DomainDeepDTA(nn.Module):
    """DeepDTA adapted for domain-length protein sequences."""

    def __init__(self, smiles_vocab=66, protein_vocab=26, embed_dim=128, num_filters=32):
        super().__init__()
        self.smiles_embed = nn.Embedding(smiles_vocab, embed_dim, padding_idx=0)
        self.protein_embed = nn.Embedding(protein_vocab, embed_dim, padding_idx=0)

        # Drug CNN: same as original (kernel 8, stacked)
        self.smiles_conv1 = nn.Conv1d(embed_dim, num_filters, 8)
        self.smiles_conv2 = nn.Conv1d(num_filters, num_filters * 2, 8)
        self.smiles_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, 8)

        # Protein CNN: smaller kernels for shorter domain sequences
        self.protein_conv1 = nn.Conv1d(embed_dim, num_filters, 3)
        self.protein_conv2 = nn.Conv1d(num_filters, num_filters * 2, 4)
        self.protein_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, 6)

        self.fc1 = nn.Linear(num_filters * 3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, smiles_seq, protein_seq):
        smiles = self.smiles_embed(smiles_seq).permute(0, 2, 1)
        smiles = F.relu(self.smiles_conv1(smiles))
        smiles = F.relu(self.smiles_conv2(smiles))
        smiles = F.relu(self.smiles_conv3(smiles))
        smiles = smiles.max(dim=2)[0]

        protein = self.protein_embed(protein_seq).permute(0, 2, 1)
        protein = F.relu(self.protein_conv1(protein))
        protein = F.relu(self.protein_conv2(protein))
        protein = F.relu(self.protein_conv3(protein))
        protein = protein.max(dim=2)[0]

        x = torch.cat([smiles, protein], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.out(x).squeeze(-1)


def encode_domain(seq: str, max_len: int = DOMAIN_MAX_LEN) -> list[int]:
    """Encode domain sequence using DeepDTA CHARPROTSET, padded to DOMAIN_MAX_LEN."""
    CHARPROTSET = {
        "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
        "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
        "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
        "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
    }
    return [CHARPROTSET.get(c, 0) for c in seq[:max_len]] + [0] * max(0, max_len - len(seq))
