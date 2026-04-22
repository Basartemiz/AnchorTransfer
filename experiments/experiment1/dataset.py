"""Feature extraction, anchor computation, and PyTorch Dataset for experiment 1."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from anchor_transfer.data.esm_encoder import encode_sequences
from anchor_transfer.model.conplex import encode_smiles

CACHE_DIR = Path(__file__).parent / "caches"
ESM650M_CACHE_PATH = CACHE_DIR / "esm650m_embeddings.pt"


def get_esm2_embeddings(
    uid_to_sequence: dict[str, str],
    max_seq_len: int = 1022,
) -> dict[str, torch.Tensor]:
    """Return ESM-2 650M mean-pooled embeddings keyed by uniprot_id.

    Sequences longer than `max_seq_len` are skipped entirely (not truncated,
    not embedded) — those uniprot_ids will be absent from the returned dict.
    Default 1022 matches ESM-2 650M's position-embedding limit.

    Missing (sequence, embedding) pairs are computed on demand and cached in
    experiments/experiment1/caches/esm650m_embeddings.pt keyed by sequence
    string, so repeated calls with overlapping sequence sets skip recomputation.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    too_long = [uid for uid, seq in uid_to_sequence.items() if len(seq) > max_seq_len]
    if too_long:
        print(f"get_esm2_embeddings: skipping {len(too_long)} sequences longer than {max_seq_len} residues")
    usable = {uid: seq for uid, seq in uid_to_sequence.items() if len(seq) <= max_seq_len}

    cache: dict[str, torch.Tensor] = {}
    if ESM650M_CACHE_PATH.exists():
        cache = torch.load(ESM650M_CACHE_PATH, map_location="cpu", weights_only=False)

    missing_seqs = {s for s in usable.values() if s not in cache}
    if missing_seqs:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        labeled = {f"seq_{i}": s for i, s in enumerate(missing_seqs)}
        new_embs = encode_sequences(
            labeled,
            model_name="esm2_t33_650M_UR50D",
            device=device,
            batch_size=4,
            max_seq_len=max_seq_len,
        )
        for label, seq in labeled.items():
            cache[seq] = torch.from_numpy(new_embs[label])
        torch.save(cache, ESM650M_CACHE_PATH)

    return {uid: cache[seq] for uid, seq in usable.items()}


def get_drug_indices(smiles_list: list[str]) -> torch.Tensor:
    """Tokenize SMILES with CHARISOSMISET into the int tensor consumed by the models."""
    return torch.tensor([encode_smiles(smi) for smi in smiles_list], dtype=torch.long)


def compute_drug_anchors(
    interactions_df: pd.DataFrame,
    train_uniprots: set[str],
    pki_threshold: float = 7.0,
) -> tuple[dict[str, str], dict[str, str], dict[str, float], dict[str, float]]:
    """Per-drug strongest / second-strongest binder from TRAIN proteins only.

    Using train-only prevents val/test queries from being used as anchors, which
    would leak held-out protein identities back into training signal.

    Drugs are dropped when their strongest training binder has pKi < pki_threshold
    (paper convention: only use confirmed strong binders as anchors).

    Returns (drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki).
    Both pKi dicts are needed because the model uses anchor_pki as a residual base
    and the dataset may fall back to the second anchor when primary == query.
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
    """Paper-style Tanimoto anchor retrieval for eval-time drugs.

    For each eval drug, find the most similar training drug by Morgan/Tanimoto
    (radius 2, 1024 bits) and use its strongest DTC-train binder as the anchor.
    `drug_to_second` holds the second-strongest binder of the SAME matched drug
    as a fallback when the primary anchor equals the query protein.

    Eval drugs are dropped (not added to either dict) when:
      - the drug has an invalid SMILES, or
      - best Tanimoto similarity to any training drug is < `tanimoto_threshold`, or
      - the chosen anchor's pKi is < `pki_threshold` (paper: strong binder ≥ 7).
    """
    import numpy as np
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs

    # Silence rdkit's per-call deprecation warning to avoid multi-GB log spam.
    RDLogger.DisableLog("rdApp.*")

    df = interactions_df[interactions_df["uniprot_id"].isin(train_uniprots)]
    df = df.sort_values("pki", ascending=False)

    # Precompute per-train-drug info: (anchor_uid, anchor_pki, second_uid, second_pki).
    train_info: dict[str, tuple[str, float, str | None, float | None]] = {}
    for smi, group in df.groupby("ligand_smiles", sort=False):
        rows = group.reset_index(drop=True)
        anchor_uid = rows.iloc[0]["uniprot_id"]
        anchor_pki = float(rows.iloc[0]["pki"])
        second_uid = rows.iloc[1]["uniprot_id"] if len(rows) > 1 else None
        second_pki = float(rows.iloc[1]["pki"]) if len(rows) > 1 else None
        train_info[smi] = (anchor_uid, anchor_pki, second_uid, second_pki)

    def _fps_to_array(fps: list[object]) -> "np.ndarray":
        arr = np.zeros((len(fps), 1024), dtype=np.uint8)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        return arr.astype(np.float32)

    # Stack training fingerprints into a (N_train, 1024) float32 matrix for BLAS.
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

    # Stack eval fingerprints similarly (invalid SMILES skipped).
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

    # Vectorized Tanimoto in chunks: sim = a·b / (|a| + |b| - a·b), via one GEMM.
    chunk = 512
    for start in range(0, len(eval_fp_list), chunk):
        end = min(start + chunk, len(eval_fp_list))
        inter = eval_arr[start:end] @ train_arr.T                       # (B, N_train)
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
    """Oracle anchor: strongest TEST binder per drug — upper-bound baseline.

    For each drug, pick the test-set protein that binds it most strongly as
    the anchor. This leaks held-out labels (anchor pKi closely tracks target
    pKi for most queries) and quantifies the performance ceiling when every
    anchor is a confirmed strong binder for its drug.

    Drugs are dropped when the strongest test binder has pKi < pki_threshold.
    The dataset layer handles the anchor==query case via drug_to_second.
    """
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
    """Per-interaction samples with strongest-binder anchors.

    Drops rows whose drug has no train-side anchor, and rows where the sole
    anchor is the query itself (matches paper protocol).
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        esm2_embeddings: dict[str, torch.Tensor],
        split_uniprots: set[str],
        drug_to_anchor: dict[str, str],
        drug_to_second: dict[str, str],
        drug_to_anchor_pki: dict[str, float],
        drug_to_second_pki: dict[str, float] | None = None,
    ):
        """Anchor pKi is a required model input (residual base) so samples without
        a known anchor_pki are dropped. Second-anchor fallback (when primary ==
        query) uses `drug_to_second_pki` if provided, else drops the sample.
        """
        if drug_to_second_pki is None:
            drug_to_second_pki = {}
        valid = set(esm2_embeddings.keys()) & set(split_uniprots)
        df = interactions_df[interactions_df["uniprot_id"].isin(valid)]

        samples: list[tuple[str, str, str, float, float]] = []
        for _, row in df.iterrows():
            smi = row["ligand_smiles"]
            if smi not in drug_to_anchor:
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
            if anchor not in esm2_embeddings:
                continue
            samples.append((query, anchor, smi, float(row["pki"]), float(anchor_pki)))

        self.samples = samples
        self.esm2 = esm2_embeddings

        unique_smi = list({s[2] for s in samples})
        self.smi_to_idx = {s: i for i, s in enumerate(unique_smi)}
        self.encoded_smi = (
            torch.tensor([encode_smiles(s) for s in unique_smi], dtype=torch.long)
            if unique_smi
            else torch.zeros((0, 100), dtype=torch.long)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        query, anchor, smi, pki, anchor_pki = self.samples[idx]
        return {
            "anchor_esm2": self.esm2[anchor],
            "query_esm2": self.esm2[query],
            "drug_indices": self.encoded_smi[self.smi_to_idx[smi]],
            "pki": pki,
            "anchor_pki": anchor_pki,
            "protein_id": query,
            "drug_id": smi,
        }


def collate_fn(batch: list[dict]) -> dict[str, object]:
    return {
        "anchor_protein_esm2": torch.stack([b["anchor_esm2"] for b in batch]),
        "protein_esm2": torch.stack([b["query_esm2"] for b in batch]),
        "drug_indices": torch.stack([b["drug_indices"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
        "anchor_pki": torch.tensor([b["anchor_pki"] for b in batch], dtype=torch.float),
        "protein_id": [b["protein_id"] for b in batch],
        "drug_id": [b["drug_id"] for b in batch],
    }
