#!/usr/bin/env python3
"""Anchor quartile analysis with overlap-safe baseline comparison.

Runs V2 and pairwise baselines on the exact same anchored rows, split by anchor
strength quartile. Training overlap filtering is enabled by default so
cross-dataset numbers are not inflated by seen proteins/drugs.

Usage:
  PYTHONPATH=src python scripts/eval_anchor_quartiles_vs_baselines.py --device cuda
"""
import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

def encode_smi(smi, ml=100):
    return [CHARISOSMISET.get(c, 0) for c in smi[:ml]] + [0] * max(0, ml - len(smi))
def encode_prot(seq, ml=1000):
    return [CHARPROTSET.get(c, 0) for c in seq[:ml]] + [0] * max(0, ml - len(seq))

def ci_fn(yt, yp):
    n = len(yt)
    if n < 2: return 0.5
    yt, yp = np.array(yt), np.array(yp)
    if n * (n - 1) // 2 > 100000:
        i = np.random.randint(0, n, 100000); j = np.random.randint(0, n, 100000)
        m = i != j; i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1); i, j = idx[0], idx[1]
    dt = yt[i] - yt[j]; dp = yp[i] - yp[j]; t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

def auroc_safe(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))


def load_esm2_embeddings():
    esm2 = {}
    for p in [
        "data/processed/esm2_35m_dtc_proteins.pt",
        "data/processed/esm2_35m_davis.pt",
        "data/processed/esm2_35m_benchmark.pt",
        "data/processed/esm2_35m_pdbbind.pt",
    ]:
        if Path(p).exists():
            e = torch.load(p, map_location="cpu", weights_only=False)
            for k, v in e.items():
                if k not in esm2:
                    esm2[k] = v
            logger.info("Loaded %d embeddings from %s (total=%d)", len(e), p, len(esm2))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    logger.info("Total valid ESM-2 embeddings: %d", len(esm2))
    return esm2


def load_base_sequences():
    seqs = {}
    seq_path = Path("data/processed/dtc_sequences.json")
    if seq_path.exists():
        seqs.update(json.load(open(seq_path)))
        logger.info("Loaded %d sequences from %s", len(seqs), seq_path)
    else:
        logger.warning("Missing %s, DeepDTA coverage may drop", seq_path)

    davis_csv = Path("data/raw/davis/davis_benchmark.csv")
    if davis_csv.exists():
        ddf = pd.read_csv(davis_csv)
        if "protein_name" in ddf.columns and "protein_sequence" in ddf.columns:
            for _, r in ddf.drop_duplicates("protein_name").iterrows():
                seqs[r["protein_name"]] = r["protein_sequence"]
    logger.info("Sequence map after Davis merge: %d", len(seqs))
    return seqs


def add_benchmark_sequences(df, seqs):
    seq_col = None
    for c in ["protein_sequence", "seq", "sequence"]:
        if c in df.columns:
            seq_col = c
            break
    if seq_col is None:
        return 0

    before = len(seqs)
    for uid, seq in df[["uniprot_id", seq_col]].dropna().drop_duplicates("uniprot_id").itertuples(index=False):
        if uid not in seqs:
            seqs[uid] = seq
    return len(seqs) - before


def build_dtc_train_filters(seqs, esm2, seed):
    dtc_path = Path("data/processed/dtc_training_interactions.csv")
    if not dtc_path.exists():
        logger.warning("Missing %s, overlap filtering disabled", dtc_path)
        return set(), set(), set()

    dtc = pd.read_csv(dtc_path)
    dtc = dtc[dtc.uniprot_id.isin(esm2)]
    all_prots = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    rng = random.Random(seed)
    rng.shuffle(all_prots)
    nt = max(1, int(len(all_prots) * 0.1))
    nv = max(1, int(len(all_prots) * 0.1))
    train_prots = set(all_prots[nt + nv:])
    train_seqs = {seqs[uid] for uid in train_prots if uid in seqs}
    train_drugs = set(dtc[dtc.uniprot_id.isin(train_prots)].ligand_smiles.unique())
    logger.info("DTC train filters: %d proteins, %d sequences, %d drugs",
                len(train_prots), len(train_seqs), len(train_drugs))
    return train_prots, train_seqs, train_drugs


def apply_overlap_filters(df, seqs, train_prots, train_seqs, train_drugs):
    before = len(df)
    df = df[~df.uniprot_id.isin(train_prots)].copy()

    if train_seqs:
        overlap_by_seq = set()
        for uid in df.uniprot_id.unique():
            if uid in seqs and seqs[uid] in train_seqs:
                overlap_by_seq.add(uid)
        if overlap_by_seq:
            df = df[~df.uniprot_id.isin(overlap_by_seq)].copy()
            logger.info("  Removed %d proteins by sequence overlap", len(overlap_by_seq))

    if train_drugs:
        before_drug = len(df)
        df = df[~df.ligand_smiles.isin(train_drugs)].copy()
        removed_drugs = before_drug - len(df)
        if removed_drugs > 0:
            logger.info("  Removed %d interactions by drug overlap", removed_drugs)

    logger.info("  Overlap filtering removed %d interactions", before - len(df))
    return df


def build_anchor_maps(df):
    strongest = {}
    second = {}
    for smi, grp in df.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        top = s.iloc[0]
        strongest[smi] = (top["uniprot_id"], float(top["pki"]))
        if len(s) > 1:
            snd = s.iloc[1]
            second[smi] = (snd["uniprot_id"], float(snd["pki"]))
    return strongest, second


def find_anchored_subset(df, esm2, strongest, second):
    rows = []
    anchor_uids = []
    anchor_pkis = []

    for i, row in df.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if smi not in strongest:
            continue

        anc_uid, anc_pki = strongest[smi]
        if anc_uid == uid:
            if smi not in second:
                continue
            anc_uid, anc_pki = second[smi]
        if anc_uid not in esm2:
            continue

        rows.append(i)
        anchor_uids.append(anc_uid)
        anchor_pkis.append(anc_pki)

    subset = df.loc[rows].copy()
    subset["anchor_uid"] = anchor_uids
    subset["anchor_pki"] = anchor_pkis
    return subset


def batch_predict_v2(model, esm2, rows, anchor_uids, device):
    """Batched V2 predictions. Returns list of pred_pki aligned with rows."""
    preds = [None] * len(rows)
    batch_a, batch_q, batch_d, batch_idx = [], [], [], []

    for i, (_, row) in enumerate(rows.iterrows()):
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        anc_uid = anchor_uids[i]
        if uid not in esm2 or anc_uid not in esm2:
            continue
        if anc_uid == uid:
            continue
        batch_a.append(esm2[anc_uid])
        batch_q.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        batch_idx.append(i)

        if len(batch_a) >= 512:
            at = torch.stack(batch_a).to(device)
            qt = torch.stack(batch_q).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)
                for k, p in enumerate(out["pki_pred"].cpu().tolist()):
                    preds[batch_idx[k]] = p
            batch_a, batch_q, batch_d, batch_idx = [], [], [], []

    if batch_a:
        at = torch.stack(batch_a).to(device)
        qt = torch.stack(batch_q).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)
            for k, p in enumerate(out["pki_pred"].cpu().tolist()):
                preds[batch_idx[k]] = p

    return preds


def batch_predict_deepdta(model, seqs, rows, device):
    preds = [None] * len(rows)
    batch_s, batch_p, batch_idx = [], [], []

    for i, (_, row) in enumerate(rows.iterrows()):
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if uid not in seqs: continue
        batch_s.append(encode_smi(smi))
        batch_p.append(encode_prot(seqs[uid]))
        batch_idx.append(i)

        if len(batch_s) >= 512:
            st = torch.tensor(batch_s, dtype=torch.long, device=device)
            pt = torch.tensor(batch_p, dtype=torch.long, device=device)
            with torch.no_grad():
                for k, p in enumerate(model(st, pt).cpu().tolist()):
                    preds[batch_idx[k]] = p
            batch_s, batch_p, batch_idx = [], [], []

    if batch_s:
        st = torch.tensor(batch_s, dtype=torch.long, device=device)
        pt = torch.tensor(batch_p, dtype=torch.long, device=device)
        with torch.no_grad():
            for k, p in enumerate(model(st, pt).cpu().tolist()):
                preds[batch_idx[k]] = p

    return preds


def batch_predict_esm_model(model, model_name, esm2, rows, device):
    preds = [None] * len(rows)
    batch_p, batch_d, batch_idx = [], [], []

    for i, (_, row) in enumerate(rows.iterrows()):
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if uid not in esm2: continue
        batch_p.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        batch_idx.append(i)

        if len(batch_p) >= 512:
            pt = torch.stack(batch_p).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                if model_name == "conplex":
                    out = model(pt, dt)["score"].cpu().tolist()
                else:  # esm_dta
                    out = model(dt, pt).cpu().tolist()
                for k, p in enumerate(out):
                    preds[batch_idx[k]] = p
            batch_p, batch_d, batch_idx = [], [], []

    if batch_p:
        pt = torch.stack(batch_p).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            if model_name == "conplex":
                out = model(pt, dt)["score"].cpu().tolist()
            else:
                out = model(dt, pt).cpu().tolist()
            for k, p in enumerate(out):
                preds[batch_idx[k]] = p

    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-overlap-filter", action="store_true",
                        help="Disable training overlap filtering (not recommended).")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Load ESM-2 and sequence maps
    esm2 = load_esm2_embeddings()
    base_seqs = load_base_sequences()
    train_prots, train_seqs, train_drugs = build_dtc_train_filters(base_seqs, esm2, args.seed)

    # Load V2
    from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
    v2 = AnchorTransferDTAv2(esm2_dim=480).to(device)
    ck = torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)
    v2.load_state_dict(ck["model_state_dict"]); v2.eval()

    # Load DeepDTA
    class DeepDTAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
            self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
            self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
            self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
            self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1)
            self.do = nn.Dropout(0.1)
        def forward(self, s, p):
            s = self.smiles_embed(s).permute(0, 2, 1)
            s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s)); s = s.max(2)[0]
            p = self.protein_embed(p).permute(0, 2, 1)
            p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p)); p = p.max(2)[0]
            x = torch.cat([s, p], 1)
            x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
            x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)
    deepdta = DeepDTAModel().to(device)
    ck = torch.load("models/deepdta_dtc/best_model.pt", map_location=device, weights_only=False)
    deepdta.load_state_dict(ck["model_state_dict"]); deepdta.eval()

    # Load ESM-DTA
    from idr_gat.model.esm_dta import EsmDTAModel
    esm_dta = EsmDTAModel(esm2_dim=480).to(device)
    ck = torch.load("models/esm_dta_dtc/best_model.pt", map_location=device, weights_only=False)
    esm_dta.load_state_dict(ck["model_state_dict"]); esm_dta.eval()

    # Load ConPlex
    from idr_gat.model.conplex import ConPlex
    conplex = ConPlex(esm2_dim=480).to(device)
    ck = torch.load("models/conplex_dtc/best_model.pt", map_location=device, weights_only=False)
    conplex.load_state_dict(ck["model_state_dict"]); conplex.eval()

    all_results = {}

    for bench_name, bench_path, rename_cols in [
        ("Davis", "data/raw/davis/davis_benchmark.csv",
         {"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"}),
        ("PDBbind", "data/raw/pdbbind_benchmark.csv", {}),
    ]:
        if not Path(bench_path).exists(): continue
        df = pd.read_csv(bench_path)
        if rename_cols:
            df = df.rename(columns=rename_cols)
        if "pKd" in df.columns and "pki" not in df.columns:
            df = df.rename(columns={"pKd": "pki"})

        # Build benchmark-local sequence map to avoid requiring manual edits to
        # dtc_sequences.json for synthetic IDs (e.g., PDB_00001).
        seqs = dict(base_seqs)
        added = add_benchmark_sequences(df, seqs)
        if added > 0:
            logger.info("%s: added %d benchmark sequences", bench_name, added)

        if not args.disable_overlap_filter:
            logger.info("%s: applying DTC overlap filters", bench_name)
            df = apply_overlap_filters(df, seqs, train_prots, train_seqs, train_drugs)
        else:
            logger.warning("%s: overlap filtering disabled by flag", bench_name)

        # Filter to proteins with ESM-2 + sequence (same subset for all models)
        valid = set(esm2.keys()) & set(seqs.keys())
        df = df[df.uniprot_id.isin(valid)].copy()
        if len(df) == 0:
            logger.warning("%s: no interactions remain after basic filtering", bench_name)
            continue

        # Build dataset-internal anchors
        strongest, second = build_anchor_maps(df)
        subset = find_anchored_subset(df, esm2, strongest, second)
        if len(subset) < 10:
            logger.warning("%s: anchored subset too small (%d)", bench_name, len(subset))
            continue

        print(f"\n{'='*90}")
        print(f"{bench_name}: {len(subset)} anchored interactions, {subset.uniprot_id.nunique()} proteins")

        # Predict all models on full subset
        v2_preds = batch_predict_v2(v2, esm2, subset, subset["anchor_uid"].tolist(), device)
        dd_preds = batch_predict_deepdta(deepdta, seqs, subset, device)
        esm_preds = batch_predict_esm_model(esm_dta, "esm_dta", esm2, subset, device)
        cpx_preds = batch_predict_esm_model(conplex, "conplex", esm2, subset, device)

        subset["v2_pred"] = v2_preds
        subset["deepdta_pred"] = dd_preds
        subset["esm_dta_pred"] = esm_preds
        subset["conplex_pred"] = cpx_preds

        # Enforce identical rows for all models.
        pred_cols = ["v2_pred", "deepdta_pred", "esm_dta_pred", "conplex_pred"]
        before_common = len(subset)
        subset = subset[subset[pred_cols].notna().all(axis=1)].copy()
        dropped = before_common - len(subset)
        if dropped > 0:
            logger.warning("%s: dropped %d rows to enforce same subset across models", bench_name, dropped)
        if len(subset) < 10:
            logger.warning("%s: too few rows after common-subset filtering (%d)", bench_name, len(subset))
            continue

        # Assign quartiles
        quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
        subset["anchor_q"] = pd.qcut(
            subset.anchor_pki.rank(method="first"),
            4,
            labels=quartile_labels,
        )

        print(f"\n{'Quartile':<16} {'Anchor pKi':<16} {'n':<7} | {'V2':<8} {'DeepDTA':<8} {'ESM-DTA':<8} {'ConPlex':<8} | {'V2 CI':<7} {'DD CI':<7}")
        print("-" * 105)

        bench_results = []
        for q in quartile_labels:
            sub = subset[subset.anchor_q == q]
            if len(sub) == 0:
                continue
            lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
            trues = sub.pki.values
            n = len(sub)

            row_result = {"quartile": q, "anchor_range": f"[{lo:.1f}-{hi:.1f}]", "n": n}

            for model_name, pred_col in [("V2", "v2_pred"), ("DeepDTA", "deepdta_pred"),
                                          ("ESM-DTA", "esm_dta_pred"), ("ConPlex", "conplex_pred")]:
                if len(sub) < 10:
                    row_result[f"{model_name}_auroc"] = None
                    row_result[f"{model_name}_ci"] = None
                    continue
                p = sub[pred_col].values
                t = sub["pki"].values
                row_result[f"{model_name}_auroc"] = auroc_safe(t, p)
                row_result[f"{model_name}_ci"] = ci_fn(t, p)

            v2a = row_result.get("V2_auroc", None)
            dda = row_result.get("DeepDTA_auroc", None)
            esa = row_result.get("ESM-DTA_auroc", None)
            cpa = row_result.get("ConPlex_auroc", None)
            v2c = row_result.get("V2_ci", None)
            ddc = row_result.get("DeepDTA_ci", None)

            fmt = lambda x: f"{x:.3f}" if x is not None and not np.isnan(x) else "  N/A"
            print(f"{q:<16} [{lo:.1f}-{hi:.1f}]{'':<6} {n:<7} | {fmt(v2a):<8} {fmt(dda):<8} {fmt(esa):<8} {fmt(cpa):<8} | {fmt(v2c):<7} {fmt(ddc):<7}")
            bench_results.append(row_result)

        # Also print overall
        for model_name, pred_col in [("V2", "v2_pred"), ("DeepDTA", "deepdta_pred"),
                                      ("ESM-DTA", "esm_dta_pred"), ("ConPlex", "conplex_pred")]:
            p = subset[pred_col].values
            t = subset["pki"].values
            a = auroc_safe(t, p)
            c = ci_fn(t, p)
            print(f"  {model_name} overall: AUROC={a:.3f}, CI={c:.3f}, n={len(subset)}")

        all_results[bench_name] = bench_results

    Path("results").mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open("results/anchor_quartile_vs_baselines.json", "w"), indent=2, default=str)
    print("\nSaved to results/anchor_quartile_vs_baselines.json")

if __name__ == "__main__":
    main()
