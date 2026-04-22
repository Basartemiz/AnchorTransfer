"""DrugBAN-style dataset: PyG molecular graphs + CHARPROTSET residue tokens.

Mirrors the experiment1 API (`AnchorTransferDataset`, `collate_fn`,
`compute_drug_anchors*`), but each sample returns a PyG `Data` drug graph
plus two CHARPROTSET-encoded protein tensors (anchor and query) instead of
ESM embeddings.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

# Match DrugBAN paper hyperparameters.
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7,
    "I": 8, "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14,
    "Q": 15, "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21,
    "V": 22, "Y": 23, "X": 24, "Z": 25,
}
MAX_PROT_LEN = 1200

CACHE_DIR = Path(__file__).parent / "caches"


def encode_protein(seq: str, max_len: int = MAX_PROT_LEN) -> torch.Tensor:
    """CHARPROTSET encoding padded/truncated to max_len (DrugBAN paper convention)."""
    ids = [CHARPROTSET.get(c.upper(), 0) for c in seq[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)


def smiles_to_pyg_graph(smi: str):
    """Build a PyG Data graph from a SMILES string via the project's 9-feat atom featurizer."""
    from anchor_transfer.model.drug_encoder import smiles_to_graph

    try:
        return smiles_to_graph(smi)
    except Exception:
        return None


def build_graph_cache(smiles_list: list[str]) -> dict[str, object]:
    """Build SMILES → PyG Data map, skipping invalid molecules."""
    cache: dict[str, object] = {}
    for smi in set(smiles_list):
        g = smiles_to_pyg_graph(smi)
        if g is not None:
            cache[smi] = g
    return cache


def build_protein_token_cache(sequences: dict[str, str]) -> dict[str, torch.Tensor]:
    """Uniprot_id → residue-index tensor."""
    return {uid: encode_protein(seq) for uid, seq in sequences.items()}


def compute_drug_anchors(
    interactions_df: pd.DataFrame,
    train_uniprots: set[str],
    pki_threshold: float = 7.0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Per-drug strongest / second-strongest train binder (anchor selection).

    Returns (drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki).
    """
    df = interactions_df[interactions_df["uniprot_id"].isin(train_uniprots)]
    df = df.sort_values("pki", ascending=False)
    drug_to_anchor: dict[str, str] = {}
    drug_to_second: dict[str, str] = {}
    drug_to_anchor_pki: dict[str, float] = {}
    drug_to_second_pki: dict[str, float] = {}
    for smi, group in df.groupby("ligand_smiles", sort=False):
        rows = group.reset_index(drop=True)
        pki = float(rows.iloc[0]["pki"])
        if pki < pki_threshold:
            continue
        drug_to_anchor[smi] = rows.iloc[0]["uniprot_id"]
        drug_to_anchor_pki[smi] = pki
        if len(rows) > 1:
            drug_to_second[smi] = rows.iloc[1]["uniprot_id"]
            drug_to_second_pki[smi] = float(rows.iloc[1]["pki"])
    return drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki


def compute_drug_anchors_tanimoto(
    interactions_df: pd.DataFrame,
    train_uniprots: set[str],
    eval_drugs: set[str],
    tanimoto_threshold: float = 0.5,
    pki_threshold: float = 7.0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Tanimoto-based anchor retrieval for eval drugs not in training."""
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    RDLogger.DisableLog("rdApp.*")

    df = interactions_df[interactions_df["uniprot_id"].isin(train_uniprots)]
    df = df.sort_values("pki", ascending=False)

    train_info: dict[str, tuple[str, float, str | None, float | None]] = {}
    for smi, group in df.groupby("ligand_smiles", sort=False):
        rows = group.reset_index(drop=True)
        anchor_uid = rows.iloc[0]["uniprot_id"]
        anchor_pki = float(rows.iloc[0]["pki"])
        second_uid = rows.iloc[1]["uniprot_id"] if len(rows) > 1 else None
        second_pki = float(rows.iloc[1]["pki"]) if len(rows) > 1 else None
        train_info[smi] = (anchor_uid, anchor_pki, second_uid, second_pki)

    def _fps_to_array(fps):
        arr = np.zeros((len(fps), 1024), dtype=np.uint8)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        return arr.astype(np.float32)

    train_smi_list: list[str] = []
    train_fp_list: list[object] = []
    for smi in train_info:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        train_smi_list.append(smi)
        train_fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    train_arr = _fps_to_array(train_fp_list)
    train_norm = train_arr.sum(axis=1)

    eval_smi_valid: list[str] = []
    eval_fp_list: list[object] = []
    for smi in eval_drugs:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        eval_smi_valid.append(smi)
        eval_fp_list.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))

    drug_to_anchor: dict[str, str] = {}
    drug_to_second: dict[str, str] = {}
    drug_to_anchor_pki: dict[str, float] = {}
    drug_to_second_pki: dict[str, float] = {}
    if not eval_fp_list:
        return drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki

    eval_arr = _fps_to_array(eval_fp_list)
    eval_norm = eval_arr.sum(axis=1)

    chunk = 512
    for start in range(0, len(eval_fp_list), chunk):
        end = min(start + chunk, len(eval_fp_list))
        inter = eval_arr[start:end] @ train_arr.T
        union = eval_norm[start:end, None] + train_norm[None, :] - inter
        sims = inter / np.maximum(union, 1.0)
        best_idx = sims.argmax(axis=1)
        best_sim = sims[np.arange(end - start), best_idx]

        for i in range(end - start):
            if best_sim[i] < tanimoto_threshold:
                continue
            anchor_uid, anchor_pki, second_uid, second_pki = train_info[train_smi_list[int(best_idx[i])]]
            if anchor_pki < pki_threshold:
                continue
            smi = eval_smi_valid[start + i]
            drug_to_anchor[smi] = anchor_uid
            drug_to_anchor_pki[smi] = anchor_pki
            if second_uid is not None:
                drug_to_second[smi] = second_uid
                drug_to_second_pki[smi] = second_pki

    return drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki


def compute_drug_anchors_oracle(
    test_interactions_df: pd.DataFrame,
    pki_threshold: float = 7.0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Oracle anchor: strongest TEST binder per drug (upper-bound baseline)."""
    df = test_interactions_df.sort_values("pki", ascending=False)
    drug_to_anchor: dict[str, str] = {}
    drug_to_second: dict[str, str] = {}
    drug_to_anchor_pki: dict[str, float] = {}
    drug_to_second_pki: dict[str, float] = {}
    for smi, group in df.groupby("ligand_smiles", sort=False):
        rows = group.reset_index(drop=True)
        pki = float(rows.iloc[0]["pki"])
        if pki < pki_threshold:
            continue
        drug_to_anchor[smi] = rows.iloc[0]["uniprot_id"]
        drug_to_anchor_pki[smi] = pki
        if len(rows) > 1:
            drug_to_second[smi] = rows.iloc[1]["uniprot_id"]
            drug_to_second_pki[smi] = float(rows.iloc[1]["pki"])
    return drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki


class AnchorTransferDataset(Dataset):
    """Per-interaction samples with strongest-binder anchors, DGL-backed.

    Each sample returns a DGL drug graph (with virtual-node padding to
    MAX_DRUG_NODES) plus two CHARPROTSET-encoded protein tensors (anchor and
    query). Drops rows without a known anchor_pki or without a buildable graph.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        protein_tokens: dict[str, torch.Tensor],
        split_uniprots: set[str],
        drug_to_anchor: dict[str, str],
        drug_to_second: dict[str, str],
        drug_to_anchor_pki: dict[str, float],
        drug_to_second_pki: dict[str, float] | None = None,
        graph_cache: dict[str, object] | None = None,
    ):
        if drug_to_second_pki is None:
            drug_to_second_pki = {}
        valid = set(protein_tokens.keys()) & set(split_uniprots)
        df = interactions_df[interactions_df["uniprot_id"].isin(valid)]

        # Build / reuse graph cache for just the SMILES we need.
        needed_smi = set(df["ligand_smiles"].astype(str))
        if graph_cache is None:
            graph_cache = build_graph_cache(list(needed_smi))
        self.graph_cache = graph_cache
        self.protein_tokens = protein_tokens

        samples: list[tuple[str, str, str, float, float]] = []
        for _, row in df.iterrows():
            smi = row["ligand_smiles"]
            if smi not in drug_to_anchor:
                continue
            if smi not in self.graph_cache:
                continue
            query = row["uniprot_id"]
            anchor = drug_to_anchor[smi]
            anchor_pki = drug_to_anchor_pki.get(smi)
            if anchor == query:
                anchor = drug_to_second.get(smi)
                anchor_pki = drug_to_second_pki.get(smi)
                if anchor is None or anchor_pki is None:
                    continue
            if anchor_pki is None:
                continue
            if anchor not in protein_tokens:
                continue
            samples.append((query, anchor, smi, float(row["pki"]), float(anchor_pki)))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        query, anchor, smi, pki, anchor_pki = self.samples[idx]
        return {
            "drug_graph": self.graph_cache[smi],
            "anchor_tokens": self.protein_tokens[anchor],
            "query_tokens": self.protein_tokens[query],
            "pki": pki,
            "anchor_pki": anchor_pki,
            "protein_id": query,
            "drug_id": smi,
        }


def collate_fn(batch: list[dict]) -> dict[str, object]:
    from torch_geometric.data import Batch

    return {
        "drug_graph": Batch.from_data_list([b["drug_graph"] for b in batch]),
        "anchor_tokens": torch.stack([b["anchor_tokens"] for b in batch]),
        "query_tokens": torch.stack([b["query_tokens"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
        "anchor_pki": torch.tensor([b["anchor_pki"] for b in batch], dtype=torch.float),
        "protein_id": [b["protein_id"] for b in batch],
        "drug_id": [b["drug_id"] for b in batch],
    }
