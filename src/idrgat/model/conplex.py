"""ConPlex-style contrastive coembedding model for drug-target interaction.

Adapted from Singh et al., PNAS 2023.
Uses ESM-2 35M protein embeddings + SMILES CNN drug encoding
projected to shared 1024-dim space with cosine distance prediction.

Architecture:
  - Protein: Linear(esm2_dim→512) → ReLU → Linear(512→1024) → L2-norm
  - Drug: SMILESEncoder(→256) → Linear(256→512) → ReLU → Linear(512→1024) → L2-norm
  - Prediction: 1 - cosine_distance(protein_emb, drug_emb)
  - Loss: BCE + triplet contrastive (alternating phases)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """DeepDTA-style parallel CNN for SMILES."""

    def __init__(self, vocab_size=52, embed_dim=128, num_filters=32, output_dim=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])
        self.proj = nn.Linear(num_filters * 3, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x).permute(0, 2, 1)
        outs = [F.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        return self.proj(torch.cat(outs, dim=1))


class ConPlex(nn.Module):
    """ConPlex-style contrastive coembedding for DTI prediction."""

    def __init__(
        self,
        esm2_dim: int = 480,
        shared_dim: int = 1024,
        drug_hidden: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.shared_dim = shared_dim

        # Protein projector: ESM-2 → shared space
        self.protein_proj = nn.Sequential(
            nn.Linear(esm2_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, shared_dim),
        )

        # Drug encoder + projector
        self.drug_encoder = SMILESEncoder(output_dim=drug_hidden)
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, shared_dim),
        )

    def encode_protein(self, esm2_emb: torch.Tensor) -> torch.Tensor:
        """Project protein ESM-2 embedding to shared space (L2-normalized)."""
        h = self.protein_proj(esm2_emb)
        return F.normalize(h, p=2, dim=-1)

    def encode_drug(self, smiles_indices: torch.Tensor) -> torch.Tensor:
        """Encode and project drug SMILES to shared space (L2-normalized)."""
        h = self.drug_encoder(smiles_indices)
        h = self.drug_proj(h)
        return F.normalize(h, p=2, dim=-1)

    def forward(
        self,
        protein_esm2: torch.Tensor,
        drug_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass — returns cosine similarity as binding score."""
        p_emb = self.encode_protein(protein_esm2)
        d_emb = self.encode_drug(drug_indices)
        # Cosine similarity (since both are L2-normalized, just dot product)
        sim = (p_emb * d_emb).sum(dim=-1)
        return {
            "score": sim,  # [-1, 1], higher = more likely to bind
            "protein_emb": p_emb,
            "drug_emb": d_emb,
        }

    def compute_loss(
        self,
        protein_esm2: torch.Tensor,
        drug_indices: torch.Tensor,
        labels: torch.Tensor,
        phase: str = "bce",
        neg_drug_indices: torch.Tensor | None = None,
        margin: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Compute loss for training.

        phase='bce': Binary cross-entropy on cosine similarity
        phase='contrastive': Triplet loss with negative drugs
        """
        out = self.forward(protein_esm2, drug_indices)

        if phase == "bce":
            # Scale similarity to [0, 1] for BCE
            prob = (out["score"] + 1) / 2  # cosine sim [-1,1] → [0,1]
            bce_loss = F.binary_cross_entropy(prob, labels.float())
            out["loss"] = bce_loss
            out["prob"] = prob

        elif phase == "contrastive":
            assert neg_drug_indices is not None
            neg_out = self.forward(protein_esm2, neg_drug_indices)
            pos_dist = 1 - out["score"]  # cosine distance
            neg_dist = 1 - neg_out["score"]
            triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()
            out["loss"] = triplet_loss
            out["prob"] = (out["score"] + 1) / 2

        return out
