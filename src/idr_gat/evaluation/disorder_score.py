"""Disorder intensity score for IDPs.

Combines 3Di structural alphabet entropy (conformational variability)
with AlphaFold pLDDT (predicted disorder) into a single [0,1] score.

Higher score = more disordered = more likely induced-fit binding.
Lower score = more residual structure = more likely conformational selection.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MAX_ENTROPY = math.log2(len(THREEDI_ALPHABET))


def threedi_entropy(seqs: list[str]) -> float:
    """Compute mean per-residue Shannon entropy across 3Di sequences.

    Each sequence represents one conformation's 3Di encoding.
    Returns normalized entropy in [0, 1].
    """
    if not seqs or len(seqs) < 2:
        return 0.0

    seq_len = min(len(s) for s in seqs)
    if seq_len == 0:
        return 0.0

    entropies = []
    for pos in range(seq_len):
        counts = Counter(s[pos] for s in seqs if pos < len(s))
        total = sum(counts.values())
        if total <= 1:
            entropies.append(0.0)
            continue
        h = 0.0
        for c, n in counts.items():
            p = n / total
            if p > 0:
                h -= p * math.log2(p)
        entropies.append(h / MAX_ENTROPY)

    return float(np.mean(entropies))


def compute_disorder_score(
    threedi_entropy: float,
    mean_plddt: float,
    w_entropy: float = 0.5,
    w_plddt: float = 0.5,
) -> float:
    """Compute composite disorder score.

    Args:
        threedi_entropy: normalized 3Di entropy [0, 1]
        mean_plddt: mean pLDDT [0, 1] (will be inverted)
    Returns:
        score in [0, 1], higher = more disordered
    """
    return w_entropy * threedi_entropy + w_plddt * (1.0 - mean_plddt)


def bin_by_disorder(
    scores: dict[str, float],
    n_bins: int = 4,
) -> dict[str, list[str]]:
    """Bin proteins into quantile groups by disorder score.

    Returns: {"Q1 (lowest)": [uid, ...], "Q2": [...], ...}
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    uids = [uid for uid, _ in sorted_items]
    n = len(uids)

    bins = {}
    for i in range(n_bins):
        start = i * n // n_bins
        end = (i + 1) * n // n_bins
        if i == 0:
            label = "Q1 (lowest)"
        elif i == n_bins - 1:
            label = f"Q{n_bins} (highest)"
        else:
            label = f"Q{i+1}"
        bins[label] = uids[start:end]

    return bins
