"""PyTorch datasets for DrugBAN paper binary classification."""
from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data
from tqdm import tqdm

log = logging.getLogger(__name__)

# Amino acid character encoding (same as DrugBAN paper)
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21,
    "V": 22, "Y": 23, "X": 24, "Z": 25,
}
MAX_PROT_LEN = 1200  # Match DrugBAN paper setting


def encode_protein(seq: str, max_len: int = MAX_PROT_LEN) -> list[int]:
    """Encode protein sequence as integer list, padded/truncated to max_len."""
    encoded = [CHARPROTSET.get(c, 0) for c in seq[:max_len]]
    return encoded + [0] * (max_len - len(encoded))


def build_graph_cache(
    smiles_list: list[str], existing_cache: dict | None = None
) -> dict[str, Data]:
    """Build PyG graph cache for a list of SMILES strings.

    Args:
        smiles_list: unique SMILES to convert.
        existing_cache: optional pre-existing cache to extend.

    Returns:
        Dict mapping SMILES -> PyG Data.
    """
    from anchor_transfer.model.drug_encoder import smiles_to_graph

    cache = dict(existing_cache or {})
    to_build = [s for s in smiles_list if s not in cache]
    if not to_build:
        return cache

    for smi in tqdm(to_build, desc="Building graphs"):
        try:
            g = smiles_to_graph(smi)
            if g is not None:
                cache[smi] = g
        except Exception:
            pass

    log.info(f"  Graph cache: {len(cache)} total")
    return cache


def load_split(
    data_dir: str, dataset: str, split: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train/val/test DataFrames for a given dataset and split type.

    Args:
        data_dir: root data directory (e.g. "data/drugban_paper").
        dataset: "bindingdb", "biosnap", or "human".
        split: "random", "cold", or "cluster".

    Returns:
        (train_df, val_df, test_df) with columns [SMILES, Protein, Y].
    """
    base = Path(data_dir) / dataset / split

    if split == "cluster":
        source_train = pd.read_csv(base / "source_train.csv")
        # target_train is UNLABELED in the paper's protocol — only used by CDAN
        # for adversarial alignment. Vanilla baselines train on source only.
        target_test = pd.read_csv(base / "target_test.csv")

        # Carve 10% of source_train as validation
        val_df = source_train.sample(frac=0.1, random_state=42)
        train_df = source_train.drop(val_df.index)
        test_df = target_test
    else:
        train_df = pd.read_csv(base / "train.csv")
        val_df = pd.read_csv(base / "val.csv")
        test_df = pd.read_csv(base / "test.csv")

    # Keep only required columns and cast labels
    cols = ["SMILES", "Protein", "Y"]
    train_df = train_df[cols].copy()
    val_df = val_df[cols].copy()
    test_df = test_df[cols].copy()
    train_df["Y"] = train_df["Y"].astype(int)
    val_df["Y"] = val_df["Y"].astype(int)
    test_df["Y"] = test_df["Y"].astype(int)

    log.info(
        f"Loaded {dataset}/{split}: "
        f"train={len(train_df)}, val={len(val_df)}, test={len(test_df)}"
    )
    return train_df, val_df, test_df


class BinaryDTIDataset(Dataset):
    """Dataset for vanilla DrugBAN binary classification."""

    def __init__(self, df: pd.DataFrame, graph_cache: dict[str, Data]):
        self.rows = []
        skipped = 0
        for _, row in df.iterrows():
            smi, prot_seq, label = row["SMILES"], row["Protein"], row["Y"]
            if smi not in graph_cache:
                skipped += 1
                continue
            self.rows.append((smi, prot_seq, label))
        if skipped:
            log.info(
                f"  BinaryDTIDataset: skipped {skipped} (no graph), kept {len(self.rows)}"
            )
        self.graph_cache = graph_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        smi, prot_seq, label = self.rows[idx]
        return {
            "graph": self.graph_cache[smi].clone(),
            "protein": torch.tensor(encode_protein(prot_seq), dtype=torch.long),
            "label": float(label),
        }


class OracleAnchorDTIDataset(Dataset):
    """Oracle anchor dataset: uses TRUE positive interactions as anchors.

    For each (drug_q, protein_q), the anchor protein is one that drug_q
    ACTUALLY binds (Y=1) in the full dataset (train+val+test).
    No self-anchors: anchor_protein != query_protein.
    Samples without a valid oracle anchor are skipped.
    """

    def __init__(self, df: pd.DataFrame, graph_cache: dict[str, Data],
                 all_positives_df: pd.DataFrame):
        """
        Args:
            df: split DataFrame [SMILES, Protein, Y]
            graph_cache: SMILES -> PyG Data
            all_positives_df: ALL positive interactions across all splits
                              (train+val+test with Y=1) [SMILES, Protein]
        """
        self.graph_cache = graph_cache

        # Build drug -> set of binding proteins from ALL data
        drug_to_all_prots: dict[str, set[str]] = defaultdict(set)
        for _, r in all_positives_df.iterrows():
            drug_to_all_prots[r["SMILES"]].add(r["Protein"])

        self.rows = []
        skipped = 0
        for _, row in df.iterrows():
            smi, prot_seq, label = row["SMILES"], row["Protein"], row["Y"]
            if smi not in graph_cache:
                skipped += 1
                continue
            # Oracle: find a protein drug_q truly binds, different from query
            anchor_prot = None
            for p in drug_to_all_prots.get(smi, set()):
                if p != prot_seq:
                    anchor_prot = p
                    break
            if anchor_prot is None:
                skipped += 1
                continue
            self.rows.append((smi, prot_seq, anchor_prot, label))

        self.kept_pairs = set((smi, prot) for smi, prot, _, _ in self.rows)
        log.info(f"  OracleAnchorDTIDataset: {len(self.rows)} kept, {skipped} skipped")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        smi, query_seq, anchor_seq, label = self.rows[idx]
        return {
            "graph": self.graph_cache[smi].clone(),
            "anchor_prot": torch.tensor(encode_protein(anchor_seq), dtype=torch.long),
            "query_prot": torch.tensor(encode_protein(query_seq), dtype=torch.long),
            "label": float(label),
        }


class SubsetBinaryDTIDataset(BinaryDTIDataset):
    """BinaryDTIDataset filtered to a specific set of (SMILES, Protein) pairs."""

    def __init__(self, df: pd.DataFrame, graph_cache: dict[str, Data],
                 keep_pairs: set[tuple[str, str]]):
        # Filter df to only kept pairs before building
        mask = df.apply(lambda r: (r["SMILES"], r["Protein"]) in keep_pairs, axis=1)
        super().__init__(df[mask], graph_cache)
        log.info(f"  SubsetBinaryDTIDataset: {len(self.rows)} (from {len(df)} total)")


class AnchorBinaryDTIDataset(Dataset):
    """Dataset for AnchorDrugBAN binary classification.

    Each sample includes the query protein plus an anchor protein
    (known positive binder retrieved via Tanimoto similarity).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        graph_cache: dict[str, Data],
        anchor_index,
        cache_dir: str | None = None,
        cache_key: str = "",
    ):
        self.graph_cache = graph_cache
        self.anchor_index = anchor_index

        # Batch-resolve all anchors — with disk cache to avoid recomputation across seeds
        valid = df["SMILES"].isin(graph_cache)
        df_valid = df[valid]

        anchor_cache = self._load_or_resolve(
            df_valid, anchor_index, graph_cache, cache_dir, cache_key
        )

        # Build rows using cached anchors
        self.rows = []
        skipped_graph = int((~valid).sum())
        skipped_anchor = 0
        for smi, prot_seq, label in zip(
            df_valid["SMILES"], df_valid["Protein"], df_valid["Y"]
        ):
            anchor_smi, anchor_prot_seq = anchor_cache[(smi, prot_seq)]
            if anchor_smi is None:
                skipped_anchor += 1
                continue
            self.rows.append((smi, prot_seq, anchor_prot_seq, label))

        # Track which (smi, prot) pairs were kept for subset evaluation
        self.kept_pairs = set((smi, prot) for smi, prot, _, _ in self.rows)

        log.info(
            f"  AnchorBinaryDTIDataset: {len(self.rows)} kept, "
            f"skipped {skipped_graph} (graph) + {skipped_anchor} (anchor)"
        )

    @staticmethod
    def _load_or_resolve(df_valid, anchor_index, graph_cache, cache_dir, cache_key):
        """Resolve anchors with disk caching keyed by (dataset, split, subset)."""
        if cache_dir and cache_key:
            cache_path = Path(cache_dir) / f"anchor_cache_{cache_key}.json"
            if cache_path.exists():
                log.info(f"  Loading cached anchors from {cache_path}")
                raw = json.loads(cache_path.read_text())
                return {
                    tuple(k.split("|||")): (v[0], v[1])
                    for k, v in raw.items()
                }

        anchor_cache = anchor_index.resolve_batch(
            df_valid["SMILES"].tolist(),
            df_valid["Protein"].tolist(),
            graph_cache,
        )

        if cache_dir and cache_key:
            cache_path = Path(cache_dir) / f"anchor_cache_{cache_key}.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            serializable = {
                f"{k[0]}|||{k[1]}": [v[0], v[1]]
                for k, v in anchor_cache.items()
            }
            cache_path.write_text(json.dumps(serializable))
            log.info(f"  Saved anchor cache to {cache_path}")

        return anchor_cache

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        smi, query_seq, anchor_seq, label = self.rows[idx]
        return {
            "graph": self.graph_cache[smi].clone(),
            "anchor_prot": torch.tensor(encode_protein(anchor_seq), dtype=torch.long),
            "query_prot": torch.tensor(encode_protein(query_seq), dtype=torch.long),
            "label": float(label),
        }


def collate_binary(batch: list[dict]) -> dict:
    """Collate function for BinaryDTIDataset."""
    return {
        "graph": Batch.from_data_list([b["graph"] for b in batch]),
        "protein": torch.stack([b["protein"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }


def collate_anchor_binary(batch: list[dict]) -> dict:
    """Collate function for AnchorBinaryDTIDataset."""
    return {
        "graph": Batch.from_data_list([b["graph"] for b in batch]),
        "anchor_prot": torch.stack([b["anchor_prot"] for b in batch]),
        "query_prot": torch.stack([b["query_prot"] for b in batch]),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float32),
    }
