#!/usr/bin/env python3
"""Comprehensive anchor quality + cross-dataset evaluation.

For each benchmark:
  1. Remove proteins that overlap with the training set
  2. Build dataset-internal anchors (strongest binder from benchmark itself)
  3. Evaluate V2 (oracle, weakest, random_protein) + pairwise baselines
  4. All on the same anchored subset

Runs for both DTC-trained and BDB-trained models.

Usage:
  PYTHONPATH=src python scripts/eval_comprehensive.py --device cuda
"""
import argparse, json, logging, math, random
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

def auroc_fn(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))

def bootstrap_ci(trues, preds, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(trues)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        s = metric_fn(trues[idx], preds[idx])
        if not np.isnan(s): scores.append(s)
    scores = np.array(scores)
    if len(scores) == 0: return float("nan"), float("nan"), float("nan")
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def build_anchors(df, esm2):
    """Build drug→strongest, drug→weakest, drug→second from a dataframe."""
    df_valid = df[df.uniprot_id.isin(esm2)]
    strongest, weakest, second = {}, {}, {}
    for smi, grp in df_valid.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        prots = s.uniprot_id.values
        strongest[smi] = prots[0]
        weakest[smi] = prots[-1]
        if len(prots) > 1: second[smi] = prots[1]
    return strongest, weakest, second


def find_anchored_subset(eval_df, esm2, strongest, second):
    """Rows where a valid oracle anchor exists."""
    rows = []
    for i, row in eval_df.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if uid not in esm2: continue
        if smi not in strongest: continue
        a = strongest[smi]
        if a == uid: a = second.get(smi)
        if a is None or a not in esm2: continue
        rows.append(i)
    return eval_df.loc[rows].copy()


BATCH_SIZE = 512

def eval_v2(model, esm2, subset, anchor_fn, device):
    preds, trues = [], []
    batch_a, batch_q, batch_d, batch_pki = [], [], [], []

    for _, row in subset.iterrows():
        uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
        a = anchor_fn(uid, smi)
        if a is None or a not in esm2: continue
        batch_a.append(esm2[a])
        batch_q.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        batch_pki.append(pki)

        if len(batch_a) >= BATCH_SIZE:
            at = torch.stack(batch_a).to(device)
            qt = torch.stack(batch_q).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)
                preds.extend(out["pki_pred"].cpu().tolist())
                trues.extend(batch_pki)
            batch_a, batch_q, batch_d, batch_pki = [], [], [], []

    if batch_a:
        at = torch.stack(batch_a).to(device)
        qt = torch.stack(batch_q).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)
            preds.extend(out["pki_pred"].cpu().tolist())
            trues.extend(batch_pki)

    return np.array(preds), np.array(trues)


def eval_pairwise(name, model, esm2, seqs, subset, device):
    preds, trues = [], []

    if name == "deepdta":
        batch_s, batch_p, batch_pki = [], [], []
        for _, row in subset.iterrows():
            uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
            if uid not in seqs: continue
            batch_s.append(encode_smi(smi))
            batch_p.append(encode_prot(seqs[uid]))
            batch_pki.append(pki)
            if len(batch_s) >= BATCH_SIZE:
                st = torch.tensor(batch_s, dtype=torch.long, device=device)
                pt = torch.tensor(batch_p, dtype=torch.long, device=device)
                with torch.no_grad():
                    preds.extend(model(st, pt).cpu().tolist())
                    trues.extend(batch_pki)
                batch_s, batch_p, batch_pki = [], [], []
        if batch_s:
            st = torch.tensor(batch_s, dtype=torch.long, device=device)
            pt = torch.tensor(batch_p, dtype=torch.long, device=device)
            with torch.no_grad():
                preds.extend(model(st, pt).cpu().tolist())
                trues.extend(batch_pki)

    elif name == "conplex":
        batch_p, batch_d, batch_pki = [], [], []
        for _, row in subset.iterrows():
            uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
            if uid not in esm2: continue
            batch_p.append(esm2[uid])
            batch_d.append(encode_smi(smi))
            batch_pki.append(pki)
            if len(batch_p) >= BATCH_SIZE:
                pt = torch.stack(batch_p).to(device)
                dt = torch.tensor(batch_d, dtype=torch.long, device=device)
                with torch.no_grad():
                    preds.extend(model(pt, dt)["score"].cpu().tolist())
                    trues.extend(batch_pki)
                batch_p, batch_d, batch_pki = [], [], []
        if batch_p:
            pt = torch.stack(batch_p).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                preds.extend(model(pt, dt)["score"].cpu().tolist())
                trues.extend(batch_pki)

    elif name == "esm_dta":
        batch_p, batch_d, batch_pki = [], [], []
        for _, row in subset.iterrows():
            uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
            if uid not in esm2: continue
            batch_p.append(esm2[uid])
            batch_d.append(encode_smi(smi))
            batch_pki.append(pki)
            if len(batch_p) >= BATCH_SIZE:
                pt = torch.stack(batch_p).to(device)
                dt = torch.tensor(batch_d, dtype=torch.long, device=device)
                with torch.no_grad():
                    preds.extend(model(dt, pt).cpu().tolist())
                    trues.extend(batch_pki)
                batch_p, batch_d, batch_pki = [], [], []
        if batch_p:
            pt = torch.stack(batch_p).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                preds.extend(model(dt, pt).cpu().tolist())
                trues.extend(batch_pki)

    return np.array(preds), np.array(trues)


def report(strategy, preds, trues, n_boot=1000):
    if len(preds) < 10:
        logger.warning("  %-20s n=%d — too few", strategy, len(preds))
        return {}
    auroc_m, auroc_lo, auroc_hi = bootstrap_ci(trues, preds, auroc_fn, n_boot)
    ci_m, ci_lo, ci_hi = bootstrap_ci(trues, preds, ci_fn, n_boot)
    rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
    logger.info("  %-20s n=%-6d AUROC=%.3f [%.3f,%.3f] CI=%.3f [%.3f,%.3f] RMSE=%.3f",
                strategy, len(preds), auroc_m, auroc_lo, auroc_hi, ci_m, ci_lo, ci_hi, rmse)
    return {"strategy": strategy, "n": int(len(preds)),
            "auroc": auroc_m, "auroc_ci": [auroc_lo, auroc_hi],
            "ci": ci_m, "ci_ci": [ci_lo, ci_hi], "rmse": rmse}


def load_deepdta(path, device):
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
    m = DeepDTAModel().to(device)
    ck = torch.load(path, map_location=device, weights_only=False)
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    return m


def run_on_benchmark(bench_name, eval_df, train_prots, v2_model, baselines,
                     esm2, seqs, train_prots_list, device, rng, n_boot,
                     train_seqs=None, train_drugs=None):
    """Filter overlapping proteins AND drugs, build anchors, evaluate all strategies."""
    before = len(eval_df)

    # Remove proteins seen during training — by ID
    eval_df = eval_df[~eval_df.uniprot_id.isin(train_prots)].copy()

    # Also remove by sequence match (handles Davis gene names vs BDB UniProt IDs)
    if train_seqs:
        overlap_by_seq = set()
        for uid in eval_df.uniprot_id.unique():
            if uid in seqs and seqs[uid] in train_seqs:
                overlap_by_seq.add(uid)
        if overlap_by_seq:
            eval_df = eval_df[~eval_df.uniprot_id.isin(overlap_by_seq)].copy()
            logger.info("  Removed %d additional proteins by sequence overlap", len(overlap_by_seq))

    # Remove drugs seen during training
    if train_drugs:
        before_drug = len(eval_df)
        eval_df = eval_df[~eval_df.ligand_smiles.isin(train_drugs)].copy()
        drug_removed = before_drug - len(eval_df)
        if drug_removed > 0:
            logger.info("  Removed %d interactions with training-set drugs", drug_removed)

    n_removed = before - len(eval_df)
    if n_removed > 0:
        logger.info("  Removed %d interactions total (protein+drug overlap)", n_removed)

    # Filter to proteins that ALL models can handle (ESM-2 + sequence)
    valid_prots = set(esm2.keys()) & set(seqs.keys())
    eval_df = eval_df[eval_df.uniprot_id.isin(valid_prots)].copy()
    logger.info("  After requiring ESM-2 + sequence: %d interactions, %d proteins",
                len(eval_df), eval_df.uniprot_id.nunique())

    # Build dataset-internal anchors (only from valid proteins)
    strongest, weakest, second = build_anchors(eval_df, esm2)
    subset = find_anchored_subset(eval_df, esm2, strongest, second)
    if len(subset) < 10:
        logger.warning("  %s: anchored subset too small (%d)", bench_name, len(subset))
        return []

    n_prots = subset.uniprot_id.nunique()
    logger.info("  Anchored subset: %d interactions, %d proteins (filtered from %d)",
                len(subset), n_prots, before)

    results = []

    # V2 strategies
    def oracle_fn(uid, smi):
        a = strongest.get(smi)
        if a == uid: a = second.get(smi)
        return a
    def weakest_fn(uid, smi):
        a = weakest.get(smi)
        if a == uid: a = strongest.get(smi)
        if a == uid: a = second.get(smi)
        return a
    def random_fn(uid, smi):
        for _ in range(20):
            p = rng.choice(train_prots_list)
            if p != uid: return p
        return rng.choice(train_prots_list)

    for strat_name, fn in [("V2_oracle", oracle_fn), ("V2_weakest", weakest_fn), ("V2_random_prot", random_fn)]:
        p, t = eval_v2(v2_model, esm2, subset, fn, device)
        r = report(strat_name, p, t, n_boot)
        if r: results.append(r)

    # Pairwise baselines
    for bname, bmodel in baselines.items():
        p, t = eval_pairwise(bname, bmodel, esm2, seqs, subset, device)
        r = report(bname, p, t, n_boot)
        if r: results.append(r)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-set", choices=["DTC", "BDB"], default="DTC",
                        help="Which training-set family to evaluate.")
    parser.add_argument("--benchmarks", nargs="+", default=["Davis", "Metz"],
                        help="Benchmarks to run (default: Davis Metz).")
    parser.add_argument("--v2-kind", choices=["v2", "v2_latent_attn"], default="v2",
                        help="Anchor model variant to load.")
    parser.add_argument("--v2-path", default=None,
                        help="Override checkpoint path for the selected anchor model.")
    parser.add_argument("--output", default="results/comprehensive_eval.json",
                        help="Output JSON path.")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load all ESM-2 embeddings
    esm2 = {}
    for p in ["data/processed/esm2_35m_dtc_proteins.pt",
              "data/processed/esm2_35m_davis.pt",
              "data/processed/esm2_35m_benchmark.pt",
              "data/processed/esm2_35m_foldseek_anchors_all.pt",
              "data/processed/esm2_35m_glass.pt",
              "data/processed/esm2_35m_metz.pt",
              "data/processed/esm2_35m_pdbbind.pt"]:
        if Path(p).exists():
            e = torch.load(p, map_location="cpu", weights_only=False)
            for k, v in e.items():
                if k not in esm2: esm2[k] = v
            logger.info("Loaded %d from %s (total %d)", len(e), p, len(esm2))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    logger.info("Total ESM-2: %d", len(esm2))

    # Sequences
    seqs = {}
    if Path("data/processed/dtc_sequences.json").exists():
        seqs.update(json.load(open("data/processed/dtc_sequences.json")))
    for dp in ["data/raw/davis/davis_benchmark.csv"]:
        if Path(dp).exists():
            ddf = pd.read_csv(dp)
            if "protein_name" in ddf.columns and "protein_sequence" in ddf.columns:
                for _, r in ddf.drop_duplicates("protein_name").iterrows():
                    seqs[r["protein_name"]] = r["protein_sequence"]
    metz_proteins = Path("data/raw/metz_proteins.csv")
    if metz_proteins.exists():
        mdf = pd.read_csv(metz_proteins)
        if {"uniprot_id", "sequence"} <= set(mdf.columns):
            for _, r in mdf.drop_duplicates("uniprot_id").iterrows():
                seqs[r["uniprot_id"]] = r["sequence"]

    # ──────────────────────────────────────────────────────────────────
    # Load benchmarks
    # ──────────────────────────────────────────────────────────────────
    benchmarks = {}

    # Davis
    for dp in ["data/raw/davis/davis_benchmark.csv"]:
        if Path(dp).exists():
            davis = pd.read_csv(dp)
            davis = davis.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles"})
            if "pKd" in davis.columns and "pki" not in davis.columns:
                davis = davis.rename(columns={"pKd": "pki"})
            benchmarks["Davis"] = davis

    # PDBbind (replaces GLASS — balanced classes, diverse families)
    pdbbind_path = Path("data/raw/pdbbind_benchmark.csv")
    if pdbbind_path.exists():
        pdb = pd.read_csv(pdbbind_path)
        # Already has uniprot_id, ligand_smiles, pki, seq columns
        benchmarks["PDBbind"] = pdb
        logger.info("PDBbind: %d interactions, %d proteins", len(pdb), pdb.uniprot_id.nunique())

    # Metz benchmark
    metz_path = Path("data/raw/metz_benchmark.csv")
    if metz_path.exists():
        metz = pd.read_csv(metz_path)
        benchmarks["Metz"] = metz
        logger.info("Metz: %d interactions, %d proteins", len(metz), metz.uniprot_id.nunique())

    # IDP benchmark
    if Path("data/raw/benchmark_affinity.csv").exists():
        bench = pd.read_csv("data/raw/benchmark_affinity.csv")
        benchmarks["IDP_ALL"] = bench
        if "protein_type" in bench.columns:
            benchmarks["IDP_only"] = bench[bench.protein_type.str.lower() == "idp"].copy()
            benchmarks["Ordered_only"] = bench[bench.protein_type.str.lower() == "ordered"].copy()

    # ──────────────────────────────────────────────────────────────────
    # Training sets: DTC and BDB
    # ──────────────────────────────────────────────────────────────────
    training_sets = {}

    # DTC
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    dtc = dtc[dtc.uniprot_id.isin(esm2)]
    all_prots_dtc = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    random.seed(args.seed); random.shuffle(all_prots_dtc)
    nt = max(1, int(len(all_prots_dtc) * 0.1))
    nv = max(1, int(len(all_prots_dtc) * 0.1))
    dtc_train_prots = set(all_prots_dtc[nt + nv:])
    dtc_train_seqs = set()
    for uid in dtc_train_prots:
        if uid in seqs: dtc_train_seqs.add(seqs[uid])
    dtc_train_drugs = set(dtc[dtc.uniprot_id.isin(dtc_train_prots)].ligand_smiles.unique())
    training_sets["DTC"] = {
        "train_prots": dtc_train_prots,
        "train_prots_list": [p for p in dtc_train_prots if p in esm2],
        "train_seqs": dtc_train_seqs,
        "train_drugs": dtc_train_drugs,
    }
    logger.info("DTC: %d train proteins, %d train drugs", len(dtc_train_prots), len(dtc_train_drugs))

    # BDB
    bdb_path = Path("data/processed/bindingdb_interactions.csv")
    if bdb_path.exists():
        bdb = pd.read_csv(bdb_path)
        bdb = bdb[bdb.uniprot_id.isin(esm2)]
        all_prots_bdb = sorted(set(bdb.uniprot_id) & set(esm2.keys()))
        random.seed(args.seed); random.shuffle(all_prots_bdb)
        nt_b = max(1, int(len(all_prots_bdb) * 0.1))
        nv_b = max(1, int(len(all_prots_bdb) * 0.1))
        bdb_train_prots = set(all_prots_bdb[nt_b + nv_b:])
        bdb_train_seqs = set()
        for uid in bdb_train_prots:
            if uid in seqs: bdb_train_seqs.add(seqs[uid])
        bdb_train_drugs = set(bdb[bdb.uniprot_id.isin(bdb_train_prots)].ligand_smiles.unique())
        training_sets["BDB"] = {
            "train_prots": bdb_train_prots,
            "train_prots_list": [p for p in bdb_train_prots if p in esm2],
            "train_seqs": bdb_train_seqs,
            "train_drugs": bdb_train_drugs,
        }
        logger.info("BDB: %d train proteins, %d train drugs", len(bdb_train_prots), len(bdb_train_drugs))

    # ──────────────────────────────────────────────────────────────────
    # Models per training set
    # ──────────────────────────────────────────────────────────────────
    model_configs = {
        "DTC": {
            "v2": "models/v2_dtc/best_model.pt",
            "v2_latent_attn": "models/v2_latent_attn_dtc/best_model.pt",
            "deepdta": "models/deepdta_dtc/best_model.pt",
            "conplex": "models/conplex_dtc/best_model.pt",
            "esm_dta": "models/esm_dta_dtc/best_model.pt",
        },
        "BDB": {
            "v2": "models/v2_bdb/best_model.pt",
            "deepdta": "models/deepdta_bdb/best_model.pt",
            "conplex": "models/conplex_bdb/best_model.pt",
        },
    }

    all_results = {}

    selected_training_sets = {args.train_set: training_sets[args.train_set]} if args.train_set in training_sets else {}

    for train_name, train_info in selected_training_sets.items():
        if train_name not in model_configs:
            continue
        configs = model_configs[train_name]

        # Load V2
        v2_path = args.v2_path or configs.get(args.v2_kind)
        if not v2_path or not Path(v2_path).exists():
            logger.warning("%s model not found for %s at %s", args.v2_kind, train_name, v2_path)
            continue
        if args.v2_kind == "v2_latent_attn":
            from idr_gat.model.anchor_transfer_latent_attn import AnchorTransferLatentAttn
            v2 = AnchorTransferLatentAttn(esm2_dim=480).to(device)
        else:
            from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
            v2 = AnchorTransferDTAv2(esm2_dim=480).to(device)
        ck = torch.load(v2_path, map_location=device, weights_only=False)
        v2.load_state_dict(ck["model_state_dict"]); v2.eval()
        logger.info("Loaded %s-%s from %s", args.v2_kind, train_name, v2_path)

        # Load baselines
        baselines = {}
        if "deepdta" in configs and Path(configs["deepdta"]).exists():
            baselines["deepdta"] = load_deepdta(configs["deepdta"], device)
            logger.info("Loaded DeepDTA-%s", train_name)
        if "conplex" in configs and Path(configs["conplex"]).exists():
            from idr_gat.model.conplex import ConPlex
            m = ConPlex(esm2_dim=480).to(device)
            ck = torch.load(configs["conplex"], map_location=device, weights_only=False)
            m.load_state_dict(ck["model_state_dict"]); m.eval()
            baselines["conplex"] = m
            logger.info("Loaded ConPlex-%s", train_name)
        if "esm_dta" in configs and Path(configs["esm_dta"]).exists():
            from idr_gat.model.esm_dta import EsmDTAModel
            m = EsmDTAModel(esm2_dim=480).to(device)
            ck = torch.load(configs["esm_dta"], map_location=device, weights_only=False)
            m.load_state_dict(ck["model_state_dict"]); m.eval()
            baselines["esm_dta"] = m
            logger.info("Loaded ESM-DTA-%s", train_name)

        # Run on each benchmark
        selected_benchmarks = {k: v for k, v in benchmarks.items() if k in set(args.benchmarks)}
        for bench_name, eval_df in selected_benchmarks.items():
            key = f"{bench_name}__{train_name}"
            logger.info("\n{'='*80}")
            logger.info("=== %s (%s trained on %s) ===", bench_name, args.v2_kind, train_name)

            results = run_on_benchmark(
                bench_name, eval_df.copy(), train_info["train_prots"],
                v2, baselines, esm2, seqs, train_info["train_prots_list"],
                device, rng, args.n_boot,
                train_seqs=train_info.get("train_seqs"),
                train_drugs=train_info.get("train_drugs"))
            all_results[key] = results

    # Summary
    logger.info("\n" + "=" * 110)
    logger.info("COMPREHENSIVE EVALUATION — OVERLAP-FILTERED, DATASET-INTERNAL ANCHORS")
    logger.info("=" * 110)
    for key, ress in all_results.items():
        if not ress: continue
        logger.info("\n--- %s (n=%d) ---", key, ress[0]["n"] if ress else 0)
        logger.info("%-20s  %-25s  %-25s  %-8s", "Strategy", "AUROC [95% CI]", "CI [95% CI]", "RMSE")
        for r in ress:
            logger.info("%-20s  %.3f [%.3f, %.3f]  %.3f [%.3f, %.3f]  %.3f",
                        r["strategy"],
                        r["auroc"], r["auroc_ci"][0], r["auroc_ci"][1],
                        r["ci"], r["ci_ci"][0], r["ci_ci"][1],
                        r["rmse"])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out, "w"), indent=2)
    logger.info("\nSaved to %s", out)


if __name__ == "__main__":
    main()
