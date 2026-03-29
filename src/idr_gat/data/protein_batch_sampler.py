"""Protein-centric batch sampler for multi-task affinity training."""
from __future__ import annotations

import math
import random
from collections import defaultdict


class ProteinBatchSampler:
    """Yields batches of protein-drug items grouped by protein.

    Each batch contains `proteins_per_batch` proteins. For each protein,
    samples pos/hard_neg/mid drugs based on pKi thresholds.

    Items in each batch get a 'label' field added:
      1 = positive (pKi >= pos_threshold)
      0 = hard negative (pKi <= neg_threshold)
     -1 = middle zone (excluded from InfoNCE, included in MSE)
    """

    def __init__(
        self,
        items: list[dict],
        proteins_per_batch: int = 4,
        pos_per_protein: int = 8,
        hard_neg_per_protein: int = 8,
        mid_per_protein: int = 4,
        pos_threshold: float = 7.0,
        neg_threshold: float = 5.0,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.proteins_per_batch = proteins_per_batch
        self.pos_per_protein = pos_per_protein
        self.hard_neg_per_protein = hard_neg_per_protein
        self.mid_per_protein = mid_per_protein
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        self.protein_pools: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: {"pos": [], "neg": [], "mid": []}
        )
        for item in items:
            uid = item["uniprot_id"]
            pki = item["pki"]
            if pki >= pos_threshold:
                self.protein_pools[uid]["pos"].append(item)
            elif pki <= neg_threshold:
                self.protein_pools[uid]["neg"].append(item)
            else:
                self.protein_pools[uid]["mid"].append(item)

        self.valid_proteins = [
            uid for uid, pools in self.protein_pools.items()
            if len(pools["pos"]) >= 1 and len(pools["neg"]) >= 1
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.valid_proteins) / self.proteins_per_batch)

    def _sample_k(self, pool: list[dict], k: int) -> list[dict]:
        if len(pool) <= k:
            return list(pool)
        return self.rng.sample(pool, k)

    def __iter__(self):
        proteins = list(self.valid_proteins)
        if self.shuffle:
            self.rng.shuffle(proteins)

        for start in range(0, len(proteins), self.proteins_per_batch):
            batch_proteins = proteins[start:start + self.proteins_per_batch]
            batch = []
            for uid in batch_proteins:
                pools = self.protein_pools[uid]
                for item in self._sample_k(pools["pos"], self.pos_per_protein):
                    batch.append({**item, "label": 1})
                for item in self._sample_k(pools["neg"], self.hard_neg_per_protein):
                    batch.append({**item, "label": 0})
                for item in self._sample_k(pools["mid"], self.mid_per_protein):
                    batch.append({**item, "label": -1})
            yield batch
