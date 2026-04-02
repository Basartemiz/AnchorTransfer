"""ESM-DTA — DeepDTA with ESM-2 protein embeddings.

Same architecture as DeepDTA but replaces learned protein character embeddings
+ Conv1d stack with frozen ESM-2 35M embeddings → Linear projection.

This isolates the effect of protein representation quality:
  DeepDTA:  char_embed(seq) → Conv1d×3 → 96-dim
  ESM-DTA:  ESM-2(seq) → Linear → 128-dim

Same drug encoder, same prediction head (adjusted for 128-dim protein).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EsmDTAModel(nn.Module):
    """DeepDTA with ESM-2 protein encoder."""

    def __init__(self, esm2_dim: int = 480, prot_proj_dim: int = 128):
        super().__init__()
        # Drug encoder: same as DeepDTA
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.sc1 = nn.Conv1d(128, 32, 8)
        self.sc2 = nn.Conv1d(32, 64, 8)
        self.sc3 = nn.Conv1d(64, 96, 8)

        # Protein encoder: ESM-2 → Linear (replaces DeepDTA's embed + Conv1d×3)
        self.protein_proj = nn.Linear(esm2_dim, prot_proj_dim)

        # Prediction head: same structure as DeepDTA
        self.fc1 = nn.Linear(96 + prot_proj_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.do = nn.Dropout(0.1)

    def forward(self, drug_indices: torch.Tensor, protein_esm2: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        drug_indices : (B, max_smi_len) int — encoded SMILES characters
        protein_esm2 : (B, esm2_dim) float — ESM-2 embedding

        Returns
        -------
        (B,) predicted pKi
        """
        # Drug encoding (same as DeepDTA)
        s = self.smiles_embed(drug_indices).permute(0, 2, 1)
        s = F.relu(self.sc1(s))
        s = F.relu(self.sc2(s))
        s = F.relu(self.sc3(s))
        s = s.max(2)[0]  # (B, 96)

        # Protein encoding (ESM-2 → projection)
        p = F.relu(self.protein_proj(protein_esm2))  # (B, 128)

        # Prediction
        x = torch.cat([s, p], 1)  # (B, 224)
        x = self.do(F.relu(self.fc1(x)))
        x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x)))
        return self.out(x).squeeze(-1)
