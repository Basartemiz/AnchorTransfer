"""MMseqs2-based OOD train/val/test split for proteins."""
from __future__ import annotations

import random
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def mmseqs_cluster(sequences: dict[str, str], identity: float, coverage: float = 0.8) -> dict[str, str]:
    """Run MMseqs2 easy-cluster and return {member_id: representative_id}.

    Proteins in the same cluster share at least `identity` fraction sequence
    identity over at least `coverage` fraction of the longer sequence.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        fasta = tmp / "in.fasta"
        with open(fasta, "w") as f:
            for uid, seq in sequences.items():
                f.write(f">{uid}\n{seq}\n")
        out_prefix = str(tmp / "out")
        work_dir = str(tmp / "work")
        # --threads 1 makes clustering deterministic across runs (same seed=42
        # + same threads → same splits). Without this MMseqs is multi-threaded
        # and cluster assignment can vary between runs.
        subprocess.run(
            ["mmseqs", "easy-cluster", str(fasta), out_prefix, work_dir,
             "--min-seq-id", str(identity), "-c", str(coverage), "--cov-mode", "0",
             "--threads", "1"],
            check=True, capture_output=True, text=True,
        )
        member_to_rep: dict[str, str] = {}
        with open(f"{out_prefix}_cluster.tsv") as f:
            for line in f:
                rep, member = line.strip().split("\t")
                member_to_rep[member] = rep
    return member_to_rep


def ood_split(
    sequences: dict[str, str],
    test_frac: float = 0.1,
    val_frac: float = 0.1,
    test_identity: float = 0.3,
    val_identity: float = 0.5,
    seed: int = 42,
) -> tuple[set[str], set[str], set[str]]:
    """Two-tier OOD split by MMseqs clustering.

    Test proteins sit in clusters disjoint from train at `test_identity` (≤30%
    max identity to train). Val proteins sit in clusters disjoint from train
    at `val_identity` (≤50% max identity to train).

    Returns (train_ids, val_ids, test_ids).
    """
    rng = random.Random(seed)
    total = len(sequences)

    test_m2r = mmseqs_cluster(sequences, identity=test_identity)
    test_clusters: dict[str, list[str]] = defaultdict(list)
    for member, rep in test_m2r.items():
        test_clusters[rep].append(member)
    reps = list(test_clusters.keys())
    rng.shuffle(reps)
    test_ids: set[str] = set()
    target_test = int(total * test_frac)
    for rep in reps:
        if len(test_ids) >= target_test:
            break
        test_ids.update(test_clusters[rep])

    remaining = {uid: seq for uid, seq in sequences.items() if uid not in test_ids}
    val_m2r = mmseqs_cluster(remaining, identity=val_identity)
    val_clusters: dict[str, list[str]] = defaultdict(list)
    for member, rep in val_m2r.items():
        val_clusters[rep].append(member)
    reps = list(val_clusters.keys())
    rng.shuffle(reps)
    val_ids: set[str] = set()
    target_val = int(total * val_frac)
    for rep in reps:
        if len(val_ids) >= target_val:
            break
        val_ids.update(val_clusters[rep])

    train_ids = set(sequences) - test_ids - val_ids
    return train_ids, val_ids, test_ids
