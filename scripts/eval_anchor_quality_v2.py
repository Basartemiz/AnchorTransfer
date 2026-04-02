#!/usr/bin/env python3
"""Anchor quality sensitivity on the SAME anchored subset.

For each benchmark, first finds the anchored subset (interactions where
an oracle anchor exists in DTC training), then evaluates ALL strategies
on that SAME subset. The only variable is which protein is used as anchor.

Strategies:
  - oracle: strongest known binder of the drug
  - weakest: weakest known binder of the drug
  - random_protein: a random training protein (not a binder of the drug)

Usage:
  PYTHONPATH=src python scripts/eval_anchor_quality_v2.py --device cuda
"""
import argparse, json, logging, math, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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

def encode_smi(smi, ml=100):
    return [CHARISOSMISET.get(c, 0) for c in smi[:ml]] + [0] * max(0, ml - len(smi))

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


def find_anchored_subset(eval_df, esm2, drug_strongest):
    """Find interactions where: (1) query has ESM-2, (2) drug has an oracle anchor,
    (3) oracle anchor != query, (4) oracle anchor has ESM-2."""
    rows = []
    for i, row in eval_df.iterrows():
        uid = row["uniprot_id"]
        smi = row["ligand_smiles"]
        if uid not in esm2: continue
        if smi not in drug_strongest: continue
        anchor = drug_strongest[smi]
        if anchor == uid: continue
        if anchor not in esm2: continue
        rows.append(i)
    return eval_df.loc[rows].copy()


def eval_on_subset(model, esm2, subset_df, anchor_fn, device):
    """Evaluate model on a fixed subset with a given anchor function.

    anchor_fn(uid, smi) → anchor_protein_id
    """
    all_preds, all_trues = [], []
    for _, row in subset_df.iterrows():
        uid = row["uniprot_id"]
        smi = row["ligand_smiles"]
        pki = row["pki"]

        anchor = anchor_fn(uid, smi)
        if anchor is None or anchor not in esm2:
            continue

        q = esm2[uid].unsqueeze(0).to(device)
        at = esm2[anchor].unsqueeze(0).to(device)
        dt = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, q, dt)
            all_preds.append(out["pki_pred"].item())
            all_trues.append(pki)

    return np.array(all_preds), np.array(all_trues)


def report(name, strategy, preds, trues, n_boot=1000):
    if len(preds) < 10:
        logger.warning("  %-18s n=%d — too few predictions", strategy, len(preds))
        return {}
    auroc_m, auroc_lo, auroc_hi = bootstrap_ci(trues, preds, auroc_fn, n_boot)
    ci_m, ci_lo, ci_hi = bootstrap_ci(trues, preds, ci_fn, n_boot)
    rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))

    logger.info("  %-18s n=%-6d AUROC=%.3f [%.3f,%.3f] CI=%.3f [%.3f,%.3f] RMSE=%.3f",
                strategy, len(preds), auroc_m, auroc_lo, auroc_hi, ci_m, ci_lo, ci_hi, rmse)
    return {
        "strategy": strategy, "n": int(len(preds)),
        "auroc": auroc_m, "auroc_ci": [auroc_lo, auroc_hi],
        "ci": ci_m, "ci_ci": [ci_lo, ci_hi],
        "rmse": rmse,
    }


def run_benchmark(name, eval_df, model, esm2, drug_strongest, drug_weakest,
                   train_prots_list, device, rng, n_boot):
    """Run all anchor strategies on the same anchored subset of a benchmark."""
    # Step 1: find the anchored subset
    subset = find_anchored_subset(eval_df, esm2, drug_strongest)
    if len(subset) < 10:
        logger.warning("%s: anchored subset too small (%d), skipping", name, len(subset))
        return []

    logger.info("\n=== %s (anchored subset: %d of %d total) ===", name, len(subset), len(eval_df))

    results = []

    # Oracle: strongest binder
    def oracle_fn(uid, smi):
        return drug_strongest.get(smi)

    preds, trues = eval_on_subset(model, esm2, subset, oracle_fn, device)
    r = report(name, "oracle_strongest", preds, trues, n_boot)
    if r: results.append(r)

    # Weakest binder
    def weakest_fn(uid, smi):
        a = drug_weakest.get(smi)
        if a == uid: return drug_strongest.get(smi)  # fallback if weakest=query
        return a

    preds, trues = eval_on_subset(model, esm2, subset, weakest_fn, device)
    r = report(name, "weakest_binder", preds, trues, n_boot)
    if r: results.append(r)

    # Random protein (not a binder, just any training protein)
    def random_fn(uid, smi):
        for _ in range(20):
            p = rng.choice(train_prots_list)
            if p != uid: return p
        return rng.choice(train_prots_list)

    preds, trues = eval_on_subset(model, esm2, subset, random_fn, device)
    r = report(name, "random_protein", preds, trues, n_boot)
    if r: results.append(r)

    return results


CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}
def encode_prot(seq, ml=1000):
    return [CHARPROTSET.get(c, 0) for c in seq[:ml]] + [0] * max(0, ml - len(seq))


def eval_pairwise_on_subset(model_name, model, esm2, seqs, subset_df, device):
    """Evaluate a pairwise model (DeepDTA/ConPlex/ESM-DTA) on the same anchored subset."""
    import torch.nn.functional as F
    all_preds, all_trues = [], []
    for _, row in subset_df.iterrows():
        uid = row["uniprot_id"]
        smi = row["ligand_smiles"]
        pki = row["pki"]
        if uid not in esm2: continue

        if model_name == "deepdta":
            if uid not in seqs: continue
            seq = seqs[uid]
            pe = torch.tensor([encode_prot(seq)], dtype=torch.long, device=device)
            se = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
            with torch.no_grad():
                all_preds.append(model(se, pe).item())
                all_trues.append(pki)

        elif model_name == "conplex":
            p = esm2[uid].unsqueeze(0).to(device)
            d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(p, d)
                all_preds.append(out["score"].item())
                all_trues.append(pki)

        elif model_name == "esm_dta":
            p = esm2[uid].unsqueeze(0).to(device)
            d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
            with torch.no_grad():
                all_preds.append(model(d, p).item())
                all_trues.append(pki)

    return np.array(all_preds), np.array(all_trues)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", default="models/v2_dtc/best_model.pt")
    parser.add_argument("--esm-path", default="data/processed/esm2_35m_dtc_proteins.pt")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
    model = AnchorTransferDTAv2(esm2_dim=480).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded V2 from %s", args.model_path)

    # Load all ESM-2 embeddings
    esm2 = torch.load(args.esm_path, map_location="cpu", weights_only=False)
    for extra in ["data/processed/esm2_35m_davis.pt",
                   "data/processed/esm2_35m_benchmark.pt",
                   "data/processed/esm2_35m_foldseek_anchors_all.pt",
                   "data/processed/esm2_35m_glass.pt"]:
        if Path(extra).exists():
            e = torch.load(extra, map_location="cpu", weights_only=False)
            for k, v in e.items():
                if k not in esm2: esm2[k] = v
            logger.info("Merged %d from %s (total %d)", len(e), extra, len(esm2))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    logger.info("Total ESM-2 embeddings: %d", len(esm2))

    # DTC split
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    dtc = dtc[dtc.uniprot_id.isin(esm2)]
    all_prots = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    random.seed(args.seed); random.shuffle(all_prots)
    n_test = max(1, int(len(all_prots) * 0.1))
    n_val = max(1, int(len(all_prots) * 0.1))
    test_prots = set(all_prots[:n_test])
    train_prots = set(all_prots[n_test + n_val:])
    train_prots_list = [p for p in train_prots if p in esm2]
    train_dtc = dtc[dtc.uniprot_id.isin(train_prots)]

    # Build anchor maps
    idx = train_dtc.groupby("ligand_smiles")["pki"].idxmax()
    drug_strongest = dict(zip(train_dtc.loc[idx].ligand_smiles, train_dtc.loc[idx].uniprot_id))
    idx_w = train_dtc.groupby("ligand_smiles")["pki"].idxmin()
    drug_weakest = dict(zip(train_dtc.loc[idx_w].ligand_smiles, train_dtc.loc[idx_w].uniprot_id))
    logger.info("Training proteins: %d, Drugs with anchors: %d", len(train_prots_list), len(drug_strongest))

    all_results = {}

    # 1. DTC test
    test_df = dtc[dtc.uniprot_id.isin(test_prots)]
    all_results["DTC_test"] = run_benchmark(
        "DTC_test", test_df, model, esm2, drug_strongest, drug_weakest,
        train_prots_list, device, rng, args.n_boot)

    # 2. Davis
    for dp in ["data/raw/davis/davis_benchmark.csv", "data/raw/davis_ki.csv"]:
        if Path(dp).exists():
            davis = pd.read_csv(dp)
            if "protein_name" in davis.columns:
                davis = davis.rename(columns={"protein_name": "uniprot_id"})
            if "drug_smiles" in davis.columns:
                davis = davis.rename(columns={"drug_smiles": "ligand_smiles"})
            if "pKd" in davis.columns and "pki" not in davis.columns:
                davis = davis.rename(columns={"pKd": "pki"})
            all_results["Davis"] = run_benchmark(
                "Davis", davis, model, esm2, drug_strongest, drug_weakest,
                train_prots_list, device, rng, args.n_boot)
            break

    # 3. GLASS Ki
    glass_data = Path("data/raw/glass/glass2_reg_major.csv")
    glass_lig = Path("data/raw/glass/ligands.tsv")
    if glass_data.exists() and glass_lig.exists():
        glass = pd.read_csv(glass_data)
        ligands = pd.read_csv(glass_lig, sep="\t")
        ik_to_smi = dict(zip(ligands["InChIKey"], ligands["SMILES"]))
        glass = glass.rename(columns={"target_uniprot_id": "uniprot_id"})
        glass["ligand_smiles"] = glass["compound_inchikey"].map(ik_to_smi)
        glass = glass.dropna(subset=["ligand_smiles"])
        if "standard_type" in glass.columns:
            glass = glass[glass.standard_type == "Ki"]
        glass["pki"] = glass["standard_value"].apply(
            lambda x: -math.log10(float(x) * 1e-9) if float(x) > 0 else 0)
        glass = glass[glass.pki > 0]
        all_results["GLASS_Ki"] = run_benchmark(
            "GLASS_Ki", glass, model, esm2, drug_strongest, drug_weakest,
            train_prots_list, device, rng, args.n_boot)

    # 4. IDP benchmark
    bench_path = Path("data/raw/benchmark_affinity.csv")
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        all_results["IDP_bench_ALL"] = run_benchmark(
            "IDP_bench_ALL", bench, model, esm2, drug_strongest, drug_weakest,
            train_prots_list, device, rng, args.n_boot)

        if "protein_type" in bench.columns:
            idps = bench[bench.protein_type.str.lower() == "idp"]
            ordered = bench[bench.protein_type.str.lower() == "ordered"]
            if len(idps) > 50:
                all_results["IDP_only"] = run_benchmark(
                    "IDP_only", idps, model, esm2, drug_strongest, drug_weakest,
                    train_prots_list, device, rng, args.n_boot)
            if len(ordered) > 50:
                all_results["Ordered_only"] = run_benchmark(
                    "Ordered_only", ordered, model, esm2, drug_strongest, drug_weakest,
                    train_prots_list, device, rng, args.n_boot)

    # --- Pairwise baselines on same anchored subsets ---
    import torch.nn as nn
    import torch.nn.functional as F

    seqs = {}
    seq_path = Path("data/processed/dtc_sequences.json")
    if seq_path.exists():
        seqs = json.load(open(seq_path))

    baseline_models = {}

    # DeepDTA
    deepdta_path = Path("models/deepdta_dtc/best_model.pt")
    if deepdta_path.exists():
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
        ck = torch.load(deepdta_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        baseline_models["deepdta"] = m
        logger.info("Loaded DeepDTA")

    # ConPlex
    conplex_path = Path("models/conplex_dtc/best_model.pt")
    if conplex_path.exists():
        from idr_gat.model.conplex import ConPlex
        m = ConPlex(esm2_dim=480).to(device)
        ck = torch.load(conplex_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        baseline_models["conplex"] = m
        logger.info("Loaded ConPlex")

    # ESM-DTA
    esm_dta_path = Path("models/esm_dta_dtc/best_model.pt")
    if esm_dta_path.exists():
        from idr_gat.model.esm_dta import EsmDTAModel
        m = EsmDTAModel(esm2_dim=480).to(device)
        ck = torch.load(esm_dta_path, map_location=device, weights_only=False)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        baseline_models["esm_dta"] = m
        logger.info("Loaded ESM-DTA")

    # Evaluate baselines on same anchored subsets
    if baseline_models:
        logger.info("\n" + "=" * 100)
        logger.info("PAIRWISE BASELINES ON SAME ANCHORED SUBSETS")
        logger.info("=" * 100)

        for bench_name in all_results:
            # Re-find the anchored subset for this benchmark
            if bench_name == "DTC_test":
                eval_df = dtc[dtc.uniprot_id.isin(test_prots)]
            elif bench_name == "IDP_bench_ALL":
                eval_df = pd.read_csv("data/raw/benchmark_affinity.csv")
            elif bench_name == "IDP_only":
                b = pd.read_csv("data/raw/benchmark_affinity.csv")
                eval_df = b[b.protein_type.str.lower() == "idp"]
            elif bench_name == "Ordered_only":
                b = pd.read_csv("data/raw/benchmark_affinity.csv")
                eval_df = b[b.protein_type.str.lower() == "ordered"]
            elif bench_name == "Davis":
                for dp in ["data/raw/davis/davis_benchmark.csv"]:
                    if Path(dp).exists():
                        eval_df = pd.read_csv(dp)
                        if "protein_name" in eval_df.columns:
                            eval_df = eval_df.rename(columns={"protein_name": "uniprot_id"})
                        if "drug_smiles" in eval_df.columns:
                            eval_df = eval_df.rename(columns={"drug_smiles": "ligand_smiles"})
                        if "pKd" in eval_df.columns and "pki" not in eval_df.columns:
                            eval_df = eval_df.rename(columns={"pKd": "pki"})
                        break
                else:
                    continue
            elif bench_name == "GLASS_Ki":
                glass = pd.read_csv("data/raw/glass/glass2_reg_major.csv")
                ligands = pd.read_csv("data/raw/glass/ligands.tsv", sep="\t")
                ik_to_smi = dict(zip(ligands["InChIKey"], ligands["SMILES"]))
                glass = glass.rename(columns={"target_uniprot_id": "uniprot_id"})
                glass["ligand_smiles"] = glass["compound_inchikey"].map(ik_to_smi)
                glass = glass.dropna(subset=["ligand_smiles"])
                glass = glass[glass.standard_type == "Ki"]
                glass["pki"] = glass["standard_value"].apply(
                    lambda x: -math.log10(float(x) * 1e-9) if float(x) > 0 else 0)
                eval_df = glass[glass.pki > 0]
            else:
                continue

            subset = find_anchored_subset(eval_df, esm2, drug_strongest)
            if len(subset) < 10: continue

            logger.info("\n--- %s (n=%d) ---", bench_name, len(subset))
            baseline_results = []
            for bname, bmodel in baseline_models.items():
                preds, trues = eval_pairwise_on_subset(bname, bmodel, esm2, seqs, subset, device)
                r = report(bench_name, bname, preds, trues, args.n_boot)
                if r:
                    baseline_results.append(r)
                    all_results[bench_name].append(r)

    # Final summary
    logger.info("\n" + "=" * 100)
    logger.info("ANCHOR QUALITY SENSITIVITY — SAME ANCHORED SUBSET PER BENCHMARK")
    logger.info("=" * 100)
    for bench_name, ress in all_results.items():
        if not ress: continue
        n_subset = ress[0]["n"] if ress else 0
        logger.info("\n--- %s (n=%d) ---", bench_name, n_subset)
        logger.info("%-18s  %-25s  %-25s  %-8s", "Strategy", "AUROC [95% CI]", "CI [95% CI]", "RMSE")
        for r in ress:
            logger.info("%-18s  %.3f [%.3f, %.3f]  %.3f [%.3f, %.3f]  %.3f",
                        r["strategy"],
                        r["auroc"], r["auroc_ci"][0], r["auroc_ci"][1],
                        r["ci"], r["ci_ci"][0], r["ci_ci"][1],
                        r["rmse"])

    out = Path("results/anchor_quality_same_subset.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(all_results, open(out, "w"), indent=2)
    logger.info("\nSaved to %s", out)


if __name__ == "__main__":
    main()
