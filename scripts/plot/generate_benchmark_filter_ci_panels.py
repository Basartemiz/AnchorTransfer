import json
import math
import os
import random
import argparse
from pathlib import Path
from multiprocessing import get_context

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import colormaps
from matplotlib.patches import Patch
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, inchi

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2, encode_smiles
from anchor_transfer.model.conplex import ConPlex
from anchor_transfer.model.esm_dta import EsmDTAModel


CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
    "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "[": 50,
    "T": 17, "]": 51, "V": 18, "Y": 19, "c": 20, "e": 21, "l": 22,
    "n": 23, "o": 24, "r": 25, "s": 26, "t": 27, "u": 28,
}
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}

MODELS = ["V2_oracle", "V2_weakest", "V2_random", "DeepDTA", "ConPlex", "ESM-DTA"]
QUARTILES = ["Q1", "Q2", "Q3", "Q4"]
TANI_BINS = ["<0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", "0.95-1.0"]
PROTOCOLS = ["filtered", "unfiltered"]
MIN_ANCHOR_PKI = 7.0
DEFAULT_CHEM_WORKERS = min(48, max(1, (os.cpu_count() or 1)))
MODEL_COLORS = {
    "V2_oracle": "#2f5c3a",
    "V2_weakest": "#9c8f3f",
    "V2_random": "#6e86b3",
    "DeepDTA": "#c27b39",
    "ConPlex": "#8a5a9e",
    "ESM-DTA": "#b65262",
}


def encode_smi(smi, ml=100):
    return [CHARISOSMISET.get(c, 0) for c in smi[:ml]] + [0] * max(0, ml - len(smi))


def encode_prot(seq, ml=1000):
    return [CHARPROTSET.get(c, 0) for c in seq[:ml]] + [0] * max(0, ml - len(seq))


def ci_fn(yt, yp):
    n = len(yt)
    if n < 2:
        return 0.5
    yt = np.array(yt)
    yp = np.array(yp)
    if n * (n - 1) // 2 > 100000:
        i = np.random.randint(0, n, 100000)
        j = np.random.randint(0, n, 100000)
        mask = i != j
        i, j = i[mask], j[mask]
    else:
        idx = np.triu_indices(n, k=1)
        i, j = idx[0], idx[1]
    dt = yt[i] - yt[j]
    dp = yp[i] - yp[j]
    ties = dt == 0
    denom = (~ties).sum()
    return float(((dt * dp) > 0).sum() / denom) if denom > 0 else 0.5


def canon_smiles(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def full_inchikey(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    try:
        return inchi.MolToInchiKey(mol)
    except Exception:
        return None


def smiles_metadata_row(smi):
    return str(smi), canon_smiles(smi), full_inchikey(smi)


def smiles_reference_row(smi):
    smi = str(smi)
    return smi, canon_smiles(smi), full_inchikey(smi), fp(smi)


def smiles_metadata_map(smiles_values, workers=None):
    unique_smiles = sorted(set(str(smi) for smi in smiles_values))
    if not unique_smiles:
        return {}
    if workers is None:
        workers = int(os.environ.get("CHEM_WORKERS", DEFAULT_CHEM_WORKERS))
    workers = max(1, min(int(workers), len(unique_smiles)))
    if workers == 1 or len(unique_smiles) < 512:
        rows = [smiles_metadata_row(smi) for smi in unique_smiles]
    else:
        with get_context("fork").Pool(processes=workers) as pool:
            rows = pool.map(smiles_metadata_row, unique_smiles, chunksize=max(32, len(unique_smiles) // (workers * 8)))
    return {
        smi: {"canonical_smiles": canonical_smi, "inchikey": ik}
        for smi, canonical_smi, ik in rows
    }


def smiles_reference_map(smiles_values, workers=None):
    unique_smiles = sorted(set(str(smi) for smi in smiles_values))
    if not unique_smiles:
        return []
    if workers is None:
        workers = int(os.environ.get("CHEM_WORKERS", DEFAULT_CHEM_WORKERS))
    workers = max(1, min(int(workers), len(unique_smiles)))
    if workers == 1 or len(unique_smiles) < 512:
        return [smiles_reference_row(smi) for smi in unique_smiles]
    with get_context("fork").Pool(processes=workers) as pool:
        return pool.map(smiles_reference_row, unique_smiles, chunksize=max(32, len(unique_smiles) // (workers * 8)))


def fp(smi):
    mol = Chem.MolFromSmiles(str(smi))
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)


def tani_bin(value):
    if pd.isna(value):
        return np.nan
    if value < 0.4:
        return "<0.4"
    if value < 0.6:
        return "0.4-0.6"
    if value < 0.8:
        return "0.6-0.8"
    if value < 0.95:
        return "0.8-0.95"
    return "0.95-1.0"


def load_embeddings():
    emb = {}
    for rel in [
        "data/processed/esm2_35m_dtc_proteins.pt",
        "data/processed/esm2_35m_dtc_proteins_full.pt",
        "data/processed/esm2_35m_davis.pt",
        "data/processed/esm2_35m_benchmark.pt",
        "data/processed/esm2_35m_glass.pt",
        "data/processed/esm2_35m_metz.pt",
    ]:
        path = Path(rel)
        if not path.exists():
            continue
        data = torch.load(path, map_location="cpu", weights_only=False)
        for k, v in data.items():
            if str(k) not in emb and not torch.isnan(v).any():
                emb[str(k)] = v
    return emb


def build_sequences():
    seqs = {}
    dtc_path = Path("data/processed/dtc_sequences.json")
    if dtc_path.exists():
        seqs.update(json.load(open(dtc_path)))
    davis_path = Path("data/raw/davis/davis_benchmark.csv")
    if davis_path.exists():
        ddf = pd.read_csv(davis_path)
        for _, row in ddf.drop_duplicates("protein_name").iterrows():
            seqs[str(row["protein_name"])] = str(row["protein_sequence"])
    metz_path = Path("data/raw/metz_proteins.csv")
    if metz_path.exists():
        mdf = pd.read_csv(metz_path)
        for _, row in mdf.drop_duplicates("uniprot_id").iterrows():
            seqs[str(row["uniprot_id"])] = str(row["sequence"])
    glass_path = Path("data/raw/glass/protein.json")
    if glass_path.exists():
        glass_meta = json.load(open(glass_path))
        for uid, info in glass_meta.items():
            sequence = info.get("sequence") if isinstance(info, dict) else None
            if sequence:
                seqs[str(uid)] = str(sequence)
    return seqs


def build_dtc_reference(seqs):
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    smiles = sorted(set(dtc["ligand_smiles"].astype(str)))
    fps = []
    canonical = set()
    inchikeys = set()
    for smi, canonical_smi, ik, bitvec in smiles_reference_map(smiles):
        if canonical_smi is not None:
            canonical.add(canonical_smi)
        if ik is not None:
            inchikeys.add(ik)
        if bitvec is not None:
            fps.append(bitvec)
    proteins = set(dtc["uniprot_id"].astype(str))
    seq_overlap = {seqs[uid] for uid in proteins if uid in seqs}
    return {
        "proteins": proteins,
        "sequences": seq_overlap,
        "canonical_smiles": canonical,
        "inchikeys": inchikeys,
        "fps": fps,
    }


def load_benchmarks():
    out = {}
    davis_path = Path("data/raw/davis/davis_benchmark.csv")
    if davis_path.exists():
        out["Davis"] = pd.read_csv(davis_path).rename(
            columns={
                "protein_name": "uniprot_id",
                "drug_smiles": "ligand_smiles",
                "pKd": "pki",
                "protein_sequence": "sequence",
            }
        )
    metz_path = Path("data/raw/metz_benchmark.csv")
    if metz_path.exists():
        out["Metz"] = pd.read_csv(metz_path)
    glass_data = Path("data/raw/glass/glass2_reg_major.csv")
    glass_ligands = Path("data/raw/glass/ligands.tsv")
    if glass_data.exists() and glass_ligands.exists():
        glass = pd.read_csv(glass_data)
        ligands = pd.read_csv(glass_ligands, sep="\t")
        ik_to_smi = dict(zip(ligands["InChIKey"], ligands["SMILES"]))
        glass = glass.rename(columns={"target_uniprot_id": "uniprot_id"})
        glass["ligand_smiles"] = glass["compound_inchikey"].map(ik_to_smi)
        glass = glass.dropna(subset=["ligand_smiles"])
        if "standard_type" in glass.columns:
            glass = glass[glass["standard_type"] == "Ki"].copy()
        glass["pki"] = glass["standard_value"].apply(
            lambda value: -math.log10(float(value) * 1e-9) if float(value) > 0 else 0.0
        )
        glass = glass[glass["pki"] > 0].copy()
        out["GLASS"] = glass
    return out


def apply_protocol_filters(df, protocol, seqs, emb, dtc_ref):
    sdf = df.copy()
    sdf["uniprot_id"] = sdf["uniprot_id"].astype(str)
    sdf["ligand_smiles"] = sdf["ligand_smiles"].astype(str)
    sdf = sdf[sdf["uniprot_id"].isin(emb)].copy()
    sdf = sdf[sdf["uniprot_id"].isin(seqs)].copy()
    chem_meta = smiles_metadata_map(sdf["ligand_smiles"].tolist())
    sdf["canonical_smiles"] = sdf["ligand_smiles"].map(lambda smi: chem_meta.get(str(smi), {}).get("canonical_smiles"))
    sdf["inchikey"] = sdf["ligand_smiles"].map(lambda smi: chem_meta.get(str(smi), {}).get("inchikey"))
    if protocol == "filtered":
        sdf = sdf[~sdf["uniprot_id"].isin(dtc_ref["proteins"])].copy()
        seq_overlap = {
            uid for uid in sdf["uniprot_id"].unique()
            if seqs.get(uid) in dtc_ref["sequences"]
        }
        sdf = sdf[~sdf["uniprot_id"].isin(seq_overlap)].copy()
        sdf = sdf[~sdf["canonical_smiles"].isin(dtc_ref["canonical_smiles"])].copy()
        sdf = sdf[~sdf["inchikey"].isin(dtc_ref["inchikeys"])].copy()
    return sdf


def build_anchor_maps(df, min_anchor_pki=MIN_ANCHOR_PKI):
    strongest_uid = {}
    strongest_pki = {}
    second_uid = {}
    weakest_uid = {}
    all_uids = sorted(set(df["uniprot_id"].astype(str)))
    for smi, group in df.groupby("ligand_smiles"):
        ranked = group[group["pki"] >= min_anchor_pki].sort_values("pki", ascending=False)
        if ranked.empty:
            continue
        uids = list(ranked["uniprot_id"].astype(str))
        pkis = list(ranked["pki"].astype(float))
        strongest_uid[smi] = uids[0]
        strongest_pki[smi] = pkis[0]
        weakest_uid[smi] = uids[-1]
        if len(uids) > 1:
            second_uid[smi] = uids[1]
    return strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids


def anchored_subset(df, strongest_uid, second_uid):
    kept = []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"])
        smi = str(row["ligand_smiles"])
        anc = strongest_uid.get(smi)
        if anc == uid:
            anc = second_uid.get(smi)
        if anc is not None:
            kept.append(idx)
    return df.loc[kept].copy()


def predict_v2(df, emb, model, anchor_fn, device, batch_size=512):
    preds, kept = [], []
    a_batch, q_batch, d_batch, idx_batch = [], [], [], []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"])
        smi = str(row["ligand_smiles"])
        anc = anchor_fn(uid, smi)
        if anc is None or anc not in emb or uid not in emb:
            continue
        a_batch.append(emb[anc])
        q_batch.append(emb[uid])
        d_batch.append(encode_smiles(smi))
        idx_batch.append(idx)
        if len(a_batch) >= batch_size:
            at = torch.stack(a_batch).to(device)
            qt = torch.stack(q_batch).to(device)
            dt = torch.tensor(d_batch, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)["pki_pred"].cpu().numpy().tolist()
            preds.extend(out)
            kept.extend(idx_batch)
            a_batch, q_batch, d_batch, idx_batch = [], [], [], []
    if a_batch:
        at = torch.stack(a_batch).to(device)
        qt = torch.stack(q_batch).to(device)
        dt = torch.tensor(d_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)["pki_pred"].cpu().numpy().tolist()
        preds.extend(out)
        kept.extend(idx_batch)
    out_df = df.loc[kept].copy()
    out_df["pred"] = preds
    return out_df


def predict_deepdta(df, seqs, model, device, batch_size=256):
    preds, kept = [], []
    s_batch, p_batch, idx_batch = [], [], []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"])
        seq = seqs.get(uid)
        if not seq:
            continue
        s_batch.append(encode_smi(str(row["ligand_smiles"])))
        p_batch.append(encode_prot(seq))
        idx_batch.append(idx)
        if len(s_batch) >= batch_size:
            st = torch.tensor(s_batch, dtype=torch.long, device=device)
            pt = torch.tensor(p_batch, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(st, pt).cpu().numpy().tolist()
            preds.extend(out)
            kept.extend(idx_batch)
            s_batch, p_batch, idx_batch = [], [], []
    if s_batch:
        st = torch.tensor(s_batch, dtype=torch.long, device=device)
        pt = torch.tensor(p_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(st, pt).cpu().numpy().tolist()
        preds.extend(out)
        kept.extend(idx_batch)
    out_df = df.loc[kept].copy()
    out_df["pred"] = preds
    return out_df


def predict_conplex(df, emb, model, device, batch_size=256):
    preds, kept = [], []
    p_batch, d_batch, idx_batch = [], [], []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"])
        if uid not in emb:
            continue
        p_batch.append(emb[uid])
        d_batch.append(encode_smi(str(row["ligand_smiles"])))
        idx_batch.append(idx)
        if len(p_batch) >= batch_size:
            pt = torch.stack(p_batch).to(device)
            dt = torch.tensor(d_batch, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(pt, dt)["score"].cpu().numpy().tolist()
            preds.extend(out)
            kept.extend(idx_batch)
            p_batch, d_batch, idx_batch = [], [], []
    if p_batch:
        pt = torch.stack(p_batch).to(device)
        dt = torch.tensor(d_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(pt, dt)["score"].cpu().numpy().tolist()
        preds.extend(out)
        kept.extend(idx_batch)
    out_df = df.loc[kept].copy()
    out_df["pred"] = preds
    return out_df


def predict_esm_dta(df, emb, model, device, batch_size=256):
    preds, kept = [], []
    p_batch, d_batch, idx_batch = [], [], []
    for idx, row in df.iterrows():
        uid = str(row["uniprot_id"])
        if uid not in emb:
            continue
        p_batch.append(emb[uid])
        d_batch.append(encode_smi(str(row["ligand_smiles"])))
        idx_batch.append(idx)
        if len(p_batch) >= batch_size:
            pt = torch.stack(p_batch).to(device)
            dt = torch.tensor(d_batch, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(dt, pt).cpu().numpy().tolist()
            preds.extend(out)
            kept.extend(idx_batch)
            p_batch, d_batch, idx_batch = [], [], []
    if p_batch:
        pt = torch.stack(p_batch).to(device)
        dt = torch.tensor(d_batch, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(dt, pt).cpu().numpy().tolist()
        preds.extend(out)
        kept.extend(idx_batch)
    out_df = df.loc[kept].copy()
    out_df["pred"] = preds
    return out_df


class DeepDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
        self.sc1 = nn.Conv1d(128, 32, 8)
        self.sc2 = nn.Conv1d(32, 64, 8)
        self.sc3 = nn.Conv1d(64, 96, 8)
        self.pc1 = nn.Conv1d(128, 32, 8)
        self.pc2 = nn.Conv1d(32, 64, 8)
        self.pc3 = nn.Conv1d(64, 96, 8)
        self.fc1 = nn.Linear(192, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, smiles_tokens, protein_tokens):
        smiles_tokens = self.smiles_embed(smiles_tokens).permute(0, 2, 1)
        smiles_tokens = F.relu(self.sc1(smiles_tokens))
        smiles_tokens = F.relu(self.sc2(smiles_tokens))
        smiles_tokens = F.relu(self.sc3(smiles_tokens))
        smiles_tokens = smiles_tokens.max(2)[0]
        protein_tokens = self.protein_embed(protein_tokens).permute(0, 2, 1)
        protein_tokens = F.relu(self.pc1(protein_tokens))
        protein_tokens = F.relu(self.pc2(protein_tokens))
        protein_tokens = F.relu(self.pc3(protein_tokens))
        protein_tokens = protein_tokens.max(2)[0]
        x = torch.cat([smiles_tokens, protein_tokens], 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        return self.out(x).squeeze(-1)


def load_model(model, path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def per_protein_metrics(df):
    rows = []
    for uid, group in df.groupby("uniprot_id"):
        rows.append(
            {
                "uniprot_id": uid,
                "n": int(len(group)),
                "ci": ci_fn(group["pki"].values, group["pred"].values),
            }
        )
    return pd.DataFrame(rows)


def build_heatmap_table(pred_df):
    rows = []
    for quartile in QUARTILES:
        for tanimoto_bin in TANI_BINS:
            subset = pred_df[
                (pred_df["anchor_quartile"].astype(str) == quartile)
                & (pred_df["tanimoto_bin"].astype(str) == tanimoto_bin)
            ]
            protein_df = per_protein_metrics(subset) if len(subset) else pd.DataFrame()
            rows.append(
                {
                    "anchor_quartile": quartile,
                    "tanimoto_bin": tanimoto_bin,
                    "n_interactions": int(len(subset)),
                    "n_proteins": int(subset["uniprot_id"].nunique()) if len(subset) else 0,
                    "mean_ci": float(protein_df["ci"].mean()) if len(protein_df) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_quartile_protein_metrics(pred_df):
    rows = []
    for model_name, model_df in pred_df.items():
        for quartile in QUARTILES:
            subset = model_df[model_df["anchor_quartile"].astype(str) == quartile]
            for uid, group in subset.groupby("uniprot_id"):
                rows.append(
                    {
                        "model": model_name,
                        "quartile": quartile,
                        "uniprot_id": uid,
                        "n": int(len(group)),
                        "ci": ci_fn(group["pki"].values, group["pred"].values),
                    }
                )
    return pd.DataFrame(rows)


def save_anchor_artifacts(out_dir, benchmark, protocol, sdf, anchor_df, strongest_uid, second_uid, weakest_uid):
    eligible = sdf[sdf["pki"] >= MIN_ANCHOR_PKI].copy()
    eligible_counts = eligible.groupby("ligand_smiles").size().to_dict()

    cutoffs = anchor_df["anchor_pki"].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).reset_index()
    cutoffs.columns = ["quantile", "anchor_pki"]
    cutoffs["benchmark"] = benchmark
    cutoffs["protocol"] = protocol
    cutoffs.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_anchor_cutoffs.csv", index=False)

    quartile_summary = (
        anchor_df.groupby("anchor_quartile")
        .agg(
            n_interactions=("ligand_smiles", "size"),
            n_drugs=("ligand_smiles", "nunique"),
            n_proteins=("uniprot_id", "nunique"),
            anchor_pki_min=("anchor_pki", "min"),
            anchor_pki_max=("anchor_pki", "max"),
            anchor_pki_mean=("anchor_pki", "mean"),
            anchor_pki_median=("anchor_pki", "median"),
        )
        .reset_index()
    )
    quartile_summary["benchmark"] = benchmark
    quartile_summary["protocol"] = protocol
    quartile_summary.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_anchor_quartile_summary.csv", index=False)

    drug_anchor_df = anchor_df[
        ["ligand_smiles", "anchor_pki", "anchor_quartile", "max_tanimoto_to_dtc", "tanimoto_bin"]
    ].drop_duplicates(subset=["ligand_smiles"]).copy()
    drug_anchor_df["strongest_anchor_uid"] = drug_anchor_df["ligand_smiles"].map(strongest_uid)
    drug_anchor_df["second_anchor_uid"] = drug_anchor_df["ligand_smiles"].map(second_uid)
    drug_anchor_df["weakest_anchor_uid"] = drug_anchor_df["ligand_smiles"].map(weakest_uid)
    drug_anchor_df["n_eligible_anchors"] = drug_anchor_df["ligand_smiles"].map(eligible_counts).fillna(0).astype(int)
    drug_anchor_df["n_anchorable_queries"] = (
        drug_anchor_df["ligand_smiles"].map(anchor_df.groupby("ligand_smiles").size()).fillna(0).astype(int)
    )
    drug_anchor_df["benchmark"] = benchmark
    drug_anchor_df["protocol"] = protocol
    drug_anchor_df = drug_anchor_df[
        [
            "benchmark",
            "protocol",
            "ligand_smiles",
            "strongest_anchor_uid",
            "second_anchor_uid",
            "weakest_anchor_uid",
            "anchor_pki",
            "anchor_quartile",
            "n_eligible_anchors",
            "n_anchorable_queries",
            "max_tanimoto_to_dtc",
            "tanimoto_bin",
        ]
    ].sort_values(["anchor_quartile", "anchor_pki", "ligand_smiles"], ascending=[True, True, True])
    drug_anchor_df.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_anchor_drugs.csv", index=False)

    interaction_anchor_df = anchor_df[
        ["uniprot_id", "ligand_smiles", "pki", "anchor_pki", "anchor_quartile", "max_tanimoto_to_dtc", "tanimoto_bin"]
    ].copy()
    interaction_anchor_df["strongest_anchor_uid"] = interaction_anchor_df["ligand_smiles"].map(strongest_uid)
    interaction_anchor_df["second_anchor_uid"] = interaction_anchor_df["ligand_smiles"].map(second_uid)
    interaction_anchor_df["weakest_anchor_uid"] = interaction_anchor_df["ligand_smiles"].map(weakest_uid)
    interaction_anchor_df["benchmark"] = benchmark
    interaction_anchor_df["protocol"] = protocol
    interaction_anchor_df = interaction_anchor_df[
        [
            "benchmark",
            "protocol",
            "uniprot_id",
            "ligand_smiles",
            "pki",
            "strongest_anchor_uid",
            "second_anchor_uid",
            "weakest_anchor_uid",
            "anchor_pki",
            "anchor_quartile",
            "max_tanimoto_to_dtc",
            "tanimoto_bin",
        ]
    ]
    interaction_anchor_df.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_anchor_interactions.csv", index=False)


def draw_heatmap_grid(benchmark, protocol, tables, out_path):
    vals = []
    for table in tables.values():
        vals.extend(table["mean_ci"].dropna().tolist())
    vmin = min(vals) if vals else 0.0
    vmax = max(vals) if vals else 1.0
    if vmin == vmax:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.8))
    axes = axes.flatten()
    fig.suptitle(f"{benchmark} ({protocol.title()}): Per-Protein CI", fontsize=18, fontweight="bold")
    cmap = colormaps["YlGn"]
    image = None
    for ax, model_name in zip(axes, MODELS):
        table = tables[model_name]
        pivot = table.pivot(index="anchor_quartile", columns="tanimoto_bin", values="mean_ci").reindex(index=QUARTILES, columns=TANI_BINS)
        counts = table.pivot(index="anchor_quartile", columns="tanimoto_bin", values="n_interactions").reindex(index=QUARTILES, columns=TANI_BINS)
        shown = np.ma.masked_invalid(pivot.values.astype(float))
        image = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(model_name, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(TANI_BINS)))
        ax.set_xticklabels(TANI_BINS, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(QUARTILES)))
        ax.set_yticklabels(QUARTILES, fontsize=9)
        for i, quartile in enumerate(QUARTILES):
            for j, tanimoto_bin in enumerate(TANI_BINS):
                value = pivot.loc[quartile, tanimoto_bin]
                n = counts.loc[quartile, tanimoto_bin]
                label = "NA" if pd.isna(value) else f"{value:.2f}\n({int(n)})"
                ax.text(j, i, label, ha="center", va="center", fontsize=7, color="black")
    cax = fig.add_axes([0.92, 0.18, 0.015, 0.64])
    fig.colorbar(image, cax=cax)
    fig.text(0.5, 0.05, "Max Tanimoto to DTC Training Drug Library", ha="center", fontsize=11)
    fig.text(0.03, 0.5, "Anchor Quartile", va="center", rotation="vertical", fontsize=11)
    fig.tight_layout(rect=[0.04, 0.07, 0.9, 0.95])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def draw_quartile_ci_distribution(benchmark, protocol, quartile_df, out_path):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    group_width = 0.78
    box_width = group_width / len(MODELS)
    quartile_positions = np.arange(len(QUARTILES)) * 1.6
    legend_handles = []

    for model_idx, model_name in enumerate(MODELS):
        positions = quartile_positions - group_width / 2 + box_width / 2 + model_idx * box_width
        box_data = []
        for quartile in QUARTILES:
            vals = quartile_df[
                (quartile_df["model"] == model_name)
                & (quartile_df["quartile"] == quartile)
            ]["ci"].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        color = MODEL_COLORS[model_name]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set_color("#333333")
                artist.set_linewidth(1.0)
        legend_handles.append(Patch(facecolor=color, edgecolor="#333333", label=model_name))

    ax.set_xticks(quartile_positions)
    ax.set_xticklabels(QUARTILES)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-Protein CI")
    ax.set_xlabel("Anchor Quartile")
    ax.set_title(f"{benchmark} ({protocol.title()}): Per-Protein CI by Anchor Quartile", fontsize=16, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="Optional benchmark names to run, e.g. Davis Metz GLASS",
    )
    parser.add_argument(
        "--quartile-only",
        action="store_true",
        help="Only write quartile-wise per-protein CI distributions and CSVs.",
    )
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs = build_sequences()
    emb = load_embeddings()
    dtc_ref = build_dtc_reference(seqs)
    benchmarks = load_benchmarks()
    if args.benchmarks:
        wanted = {name.lower() for name in args.benchmarks}
        benchmarks = {name: df for name, df in benchmarks.items() if name.lower() in wanted}

    v2 = load_model(AnchorTransferDTAv2(esm2_dim=480).to(device), "models/v2_dtc/best_model.pt", device)
    deepdta = load_model(DeepDTAModel().to(device), "models/deepdta_dtc/best_model.pt", device)
    conplex = load_model(ConPlex(esm2_dim=480).to(device), "models/conplex_dtc/best_model.pt", device)
    esm_dta = load_model(EsmDTAModel(esm2_dim=480).to(device), "models/esm_dta_dtc/best_model.pt", device)

    out_dir = Path("results/benchmark_filter_ci_panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []

    for benchmark, bench_df in benchmarks.items():
        for protocol in PROTOCOLS:
            sdf = apply_protocol_filters(bench_df, protocol, seqs, emb, dtc_ref)
            strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
            anchor_df = anchored_subset(sdf, strongest_uid, second_uid)
            if anchor_df.empty:
                continue
            anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
            anchor_df["anchor_quartile"] = pd.qcut(
                anchor_df["anchor_pki"],
                4,
                labels=QUARTILES,
                duplicates="drop",
            )

            unique_drugs = sorted(anchor_df["ligand_smiles"].astype(str).unique())
            tani_map = {}
            for smi in unique_drugs:
                qfp = fp(smi)
                if qfp is None:
                    tani_map[smi] = float("nan")
                else:
                    tani_map[smi] = float(max(DataStructs.BulkTanimotoSimilarity(qfp, dtc_ref["fps"])))
            anchor_df["max_tanimoto_to_dtc"] = anchor_df["ligand_smiles"].map(tani_map)
            anchor_df["tanimoto_bin"] = anchor_df["max_tanimoto_to_dtc"].map(tani_bin)
            anchor_df["protocol"] = protocol
            anchor_df["benchmark"] = benchmark
            save_anchor_artifacts(out_dir, benchmark, protocol, sdf, anchor_df, strongest_uid, second_uid, weakest_uid)

            def oracle(uid, smi):
                anc = strongest_uid.get(smi)
                if anc == uid:
                    anc = second_uid.get(smi)
                return anc

            def weakest(uid, smi):
                anc = weakest_uid.get(smi)
                if anc == uid:
                    anc = second_uid.get(smi)
                return anc

            rng = random.Random(seed)

            def rand_anchor(uid, _smi):
                choices = [protein for protein in all_uids if protein != uid]
                return rng.choice(choices) if choices else None

            predictions = {
                "V2_oracle": predict_v2(anchor_df, emb, v2, oracle, device),
                "V2_weakest": predict_v2(anchor_df, emb, v2, weakest, device),
                "V2_random": predict_v2(anchor_df, emb, v2, rand_anchor, device),
                "DeepDTA": predict_deepdta(anchor_df, seqs, deepdta, device),
                "ConPlex": predict_conplex(anchor_df, emb, conplex, device),
                "ESM-DTA": predict_esm_dta(anchor_df, emb, esm_dta, device),
            }

            model_tables = {}
            quartile_frames = {}
            bench_rows = []
            for model_name, pred_df in predictions.items():
                pred_df["anchor_quartile"] = anchor_df.loc[pred_df.index, "anchor_quartile"]
                pred_df["tanimoto_bin"] = anchor_df.loc[pred_df.index, "tanimoto_bin"]
                pred_df["protocol"] = protocol
                pred_df["benchmark"] = benchmark
                prot_df = per_protein_metrics(pred_df)
                quartile_frames[model_name] = pred_df
                if not args.quartile_only:
                    pred_df.to_csv(
                        out_dir / f"{benchmark.lower()}_{protocol}_{model_name.lower()}_predictions.csv",
                        index=False,
                    )
                    heatmap_df = build_heatmap_table(pred_df)
                    heatmap_df["benchmark"] = benchmark
                    heatmap_df["protocol"] = protocol
                    heatmap_df["model"] = model_name
                    model_tables[model_name] = heatmap_df
                    bench_rows.append(
                        {
                            "benchmark": benchmark,
                            "protocol": protocol,
                            "model": model_name,
                            "n_interactions": int(len(pred_df)),
                            "n_proteins": int(pred_df["uniprot_id"].nunique()),
                            "n_drugs": int(pred_df["ligand_smiles"].nunique()),
                            "mean_per_protein_ci": float(prot_df["ci"].mean()) if len(prot_df) else float("nan"),
                        }
                    )

            quartile_ci_df = build_quartile_protein_metrics(predictions)
            quartile_ci_df["benchmark"] = benchmark
            quartile_ci_df["protocol"] = protocol
            quartile_ci_df.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_quartile_per_protein_ci.csv", index=False)

            if not args.quartile_only:
                heatmap_all = pd.concat(list(model_tables.values()), ignore_index=True)
                heatmap_all.to_csv(out_dir / f"{benchmark.lower()}_{protocol}_all_model_heatmap_cells.csv", index=False)
                draw_heatmap_grid(
                    benchmark,
                    protocol,
                    model_tables,
                    out_dir / f"{benchmark.lower()}_{protocol}_heatmap_ci.png",
                )
            draw_quartile_ci_distribution(
                benchmark,
                protocol,
                quartile_ci_df,
                out_dir / f"{benchmark.lower()}_{protocol}_quartile_ci_distribution.png",
            )

            if not args.quartile_only:
                summary_rows.extend(bench_rows)
                pd.DataFrame(bench_rows).to_csv(out_dir / f"{benchmark.lower()}_{protocol}_summary.csv", index=False)

    if not args.quartile_only:
        pd.DataFrame(summary_rows).to_csv(out_dir / "benchmark_filter_ci_summary.csv", index=False)
        print(pd.DataFrame(summary_rows).to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
