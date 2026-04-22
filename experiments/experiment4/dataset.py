"""CoNCISE-style dataset: Morgan fingerprints + Raygun-compressed ESM-2 embeddings.

Each sample returns a 2048-bit Morgan FP (float tensor) plus two Raygun
embeddings (50, 1280) — one for the anchor protein and one for the query.
Caches SMILES → FP and uniprot → Raygun emb on disk so subsequent runs skip
the expensive ESM-2 + Raygun precompute.

Anchor helpers (`compute_drug_anchors*`) are identical to the experiment2
versions — they operate on the interactions dataframe, not on featurization.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

CACHE_DIR = Path(__file__).parent / "caches"
MORGAN_FP_PATH = CACHE_DIR / "morgan_fp.pkl"
RAYGUN_CACHE_PATH = CACHE_DIR / "raygun_embeddings.pt"

MORGAN_BITS = 2048
RAYGUN_TOKENS = 50
RAYGUN_DIM = 1280


def _compute_morgan(smi: str) -> np.ndarray | None:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=MORGAN_BITS)
    return np.array(fp, dtype=np.float32)


def build_morgan_cache(smiles_list: list[str]) -> dict[str, torch.Tensor]:
    """SMILES → (2048,) float tensor Morgan FP, persisted to disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, torch.Tensor] = {}
    if MORGAN_FP_PATH.exists():
        with open(MORGAN_FP_PATH, "rb") as f:
            cache = pickle.load(f)

    needed = [s for s in set(smiles_list) if s not in cache]
    if needed:
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        for smi in needed:
            fp = _compute_morgan(smi)
            if fp is not None:
                cache[smi] = torch.from_numpy(fp)
        with open(MORGAN_FP_PATH, "wb") as f:
            pickle.dump(cache, f)
    return cache


def build_raygun_cache(
    sequences: dict[str, str],
    device: str | None = None,
) -> dict[str, torch.Tensor]:
    """Uniprot → (50, 1280) Raygun-compressed ESM-2 embedding.

    Loads the persisted cache if present; computes only missing entries.
    Requires the `fair-esm` and `raygun` packages + a CUDA device for any
    new uniprots. Existing entries are returned as-is.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache: dict[str, torch.Tensor] = {}
    if RAYGUN_CACHE_PATH.exists():
        cache = torch.load(RAYGUN_CACHE_PATH, map_location="cpu", weights_only=False)

    # Raygun's reduction layer uses windowed einops that underflow on
    # sequences shorter than the 50-token target, so skip them.
    missing = [uid for uid in sequences if uid not in cache and len(sequences[uid]) >= 50]
    if not missing:
        return cache

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    import esm

    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device).eval()

    esm_embeddings: dict[str, torch.Tensor] = {}
    batch_size = 8
    with torch.no_grad():
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            batch = [(uid, sequences[uid][:1022]) for uid in chunk]
            _, _, tokens = bc(batch)
            out = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
            for j, (uid, seq) in enumerate(batch):
                esm_embeddings[uid] = out["representations"][33][j:j + 1, 1:len(seq) + 1, :].cpu()

    del esm_model
    torch.cuda.empty_cache()

    raymodel, _, _ = torch.hub.load(
        "rohitsinghlab/raygun", "pretrained_uniref50_95000_750M", trust_repo=True
    )
    raymodel = raymodel.to(device).eval()
    with torch.no_grad():
        for uid, emb in esm_embeddings.items():
            ray_enc = raymodel.encoder(emb.to(device)).squeeze(0).cpu()
            cache[uid] = ray_enc

    del raymodel
    torch.cuda.empty_cache()
    torch.save(cache, RAYGUN_CACHE_PATH)
    return cache


def compute_drug_anchors(
    interactions_df: pd.DataFrame,
    train_uniprots: set[str],
    pki_threshold: float = 7.0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Per-drug strongest / second-strongest train binder."""
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
    """Per-interaction samples: Morgan FP + anchor/query Raygun embeddings."""

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        protein_embeddings: dict[str, torch.Tensor],
        split_uniprots: set[str],
        drug_to_anchor: dict[str, str],
        drug_to_second: dict[str, str],
        drug_to_anchor_pki: dict[str, float],
        drug_to_second_pki: dict[str, float] | None = None,
        morgan_cache: dict[str, torch.Tensor] | None = None,
    ):
        if drug_to_second_pki is None:
            drug_to_second_pki = {}
        valid = set(protein_embeddings.keys()) & set(split_uniprots)
        df = interactions_df[interactions_df["uniprot_id"].isin(valid)]

        needed_smi = set(df["ligand_smiles"].astype(str))
        if morgan_cache is None:
            morgan_cache = build_morgan_cache(list(needed_smi))
        self.morgan_cache = morgan_cache
        self.protein_embeddings = protein_embeddings

        samples: list[tuple[str, str, str, float, float]] = []
        for _, row in df.iterrows():
            smi = row["ligand_smiles"]
            if smi not in drug_to_anchor:
                continue
            if smi not in self.morgan_cache:
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
            if anchor not in protein_embeddings:
                continue
            samples.append((query, anchor, smi, float(row["pki"]), float(anchor_pki)))

        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        query, anchor, smi, pki, anchor_pki = self.samples[idx]
        return {
            "drug_fp": self.morgan_cache[smi],
            "anchor_emb": self.protein_embeddings[anchor],
            "query_emb": self.protein_embeddings[query],
            "pki": pki,
            "anchor_pki": anchor_pki,
            "protein_id": query,
            "drug_id": smi,
        }


def collate_fn(batch: list[dict]) -> dict[str, object]:
    return {
        "drug_fp": torch.stack([b["drug_fp"] for b in batch]),
        "anchor_emb": torch.stack([b["anchor_emb"] for b in batch]),
        "query_emb": torch.stack([b["query_emb"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
        "anchor_pki": torch.tensor([b["anchor_pki"] for b in batch], dtype=torch.float),
        "protein_id": [b["protein_id"] for b in batch],
        "drug_id": [b["drug_id"] for b in batch],
    }
