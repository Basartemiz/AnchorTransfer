#!/usr/bin/env python3
"""Evaluate V2-Attention model on all external benchmarks (anchored only).

Only reports results for interactions where a valid anchor is found
(drug exists in DTC training data with a known strong binder ≠ query protein).

Usage:
  PYTHONPATH=src python scripts/eval_v2_attn_benchmarks.py --device cuda
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


def eval_benchmark(model, esm2, benchmark_df, drug_strongest, device, name="benchmark"):
    """Evaluate on anchored interactions only (skip when no valid anchor found)."""
    all_preds, all_trues = [], []
    n_total, n_anchored, n_no_anchor, n_no_esm = 0, 0, 0, 0

    for _, row in benchmark_df.iterrows():
        uid = row["uniprot_id"]
        smi = row["ligand_smiles"]
        pki = row["pki"]
        n_total += 1

        if uid not in esm2:
            n_no_esm += 1
            continue

        # Find anchor: strongest binder of this drug from DTC training
        anchor = None
        if smi in drug_strongest:
            a = drug_strongest[smi]
            if a != uid and a in esm2:
                anchor = a
        if not anchor:
            n_no_anchor += 1
            continue

        n_anchored += 1
        q = esm2[uid].unsqueeze(0).to(device)
        at = esm2[anchor].unsqueeze(0).to(device)
        dt = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, q, dt)
            all_preds.append(out["pki_pred"].item())
            all_trues.append(pki)

    if not all_preds:
        logger.warning("%s: no anchored predictions (total=%d, no_esm=%d, no_anchor=%d)",
                        name, n_total, n_no_esm, n_no_anchor)
        return {}

    preds = np.array(all_preds)
    trues = np.array(all_trues)
    ci = ci_fn(trues, preds)
    rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))

    # AUROC with 7/5 thresholds
    binder = trues >= 7.0
    non_binder = trues <= 5.0
    mask = binder | non_binder
    auroc = None
    if mask.sum() > 0 and binder[mask].sum() > 0 and non_binder[mask].sum() > 0:
        auroc = float(roc_auc_score(binder[mask].astype(int), preds[mask]))

    logger.info("%s: n=%d (of %d total, %d no_anchor, %d no_esm) AUROC=%s CI=%.3f RMSE=%.3f",
                name, n_anchored, n_total, n_no_anchor, n_no_esm,
                f"{auroc:.3f}" if auroc else "N/A", ci, rmse)
    return {"name": name, "n": n_anchored, "n_total": n_total, "auroc": auroc,
            "ci": ci, "rmse": rmse, "n_no_anchor": n_no_anchor, "n_no_esm": n_no_esm}


def load_glass_with_smiles(glass_path, ligands_path):
    """Load GLASS data and join with SMILES from ligands.tsv."""
    glass = pd.read_csv(glass_path)
    ligands = pd.read_csv(ligands_path, sep="\t")

    # Build InChIKey → SMILES mapping
    ik_to_smi = dict(zip(ligands["InChIKey"], ligands["SMILES"]))

    glass = glass.rename(columns={"target_uniprot_id": "uniprot_id"})
    glass["ligand_smiles"] = glass["compound_inchikey"].map(ik_to_smi)
    glass = glass.dropna(subset=["ligand_smiles"])

    # Filter to Ki only and convert nM → pKi
    if "standard_type" in glass.columns:
        glass = glass[glass.standard_type == "Ki"]
    glass["pki"] = glass["standard_value"].apply(
        lambda x: -math.log10(float(x) * 1e-9) if float(x) > 0 else 0)
    glass = glass[glass.pki > 0]

    logger.info("GLASS: %d interactions with SMILES (Ki only)", len(glass))
    return glass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", default="models/v2_attn_dtc/best_model.pt")
    parser.add_argument("--esm-path", default="data/processed/esm2_35m_dtc_proteins.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    from idr_gat.model.anchor_transfer_attn import AnchorTransferAttn
    model = AnchorTransferAttn(esm2_dim=480).to(device)
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded v2_attn from %s (epoch %d)", args.model_path, ckpt.get("epoch", -1))

    # Load ESM-2 embeddings (merge all available)
    esm2 = torch.load(args.esm_path, map_location="cpu", weights_only=False)
    for extra in ["data/processed/esm2_35m_davis.pt",
                   "data/processed/esm2_35m_benchmark.pt",
                   "data/processed/esm2_35m_foldseek_anchors_all.pt"]:
        p = Path(extra)
        if p.exists():
            e = torch.load(p, map_location="cpu", weights_only=False)
            for k, v in e.items():
                if k not in esm2:
                    esm2[k] = v
            logger.info("Merged %d from %s (total %d)", len(e), extra, len(esm2))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    logger.info("Total ESM-2 embeddings: %d", len(esm2))

    # Build drug→strongest anchor from DTC training proteins only
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    dtc = dtc[dtc.uniprot_id.isin(esm2)]
    all_prots = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    random.seed(args.seed); random.shuffle(all_prots)
    n_test = max(1, int(len(all_prots) * 0.1))
    n_val = max(1, int(len(all_prots) * 0.1))
    train_prots = set(all_prots[n_test + n_val:])
    train_dtc = dtc[dtc.uniprot_id.isin(train_prots)]

    idx = train_dtc.groupby("ligand_smiles")["pki"].idxmax()
    drug_strongest = dict(zip(train_dtc.loc[idx].ligand_smiles, train_dtc.loc[idx].uniprot_id))
    logger.info("Drug→strongest anchor mapping: %d drugs", len(drug_strongest))

    results = []

    # 1. DTC test set (anchored only)
    test_prots = set(all_prots[:n_test])
    test_dtc = dtc[dtc.uniprot_id.isin(test_prots)]
    r = eval_benchmark(model, esm2, test_dtc, drug_strongest, device, "DTC_test")
    results.append(r)

    # 2. Davis
    for dp in ["data/raw/davis/davis_benchmark.csv", "data/raw/davis_ki.csv",
               "data/raw/davis_benchmark.csv"]:
        if Path(dp).exists():
            davis = pd.read_csv(dp)
            if "protein_name" in davis.columns:
                davis = davis.rename(columns={"protein_name": "uniprot_id"})
            if "drug_smiles" in davis.columns:
                davis = davis.rename(columns={"drug_smiles": "ligand_smiles"})
            if "pKd" in davis.columns and "pki" not in davis.columns:
                davis = davis.rename(columns={"pKd": "pki"})
            r = eval_benchmark(model, esm2, davis, drug_strongest, device, "Davis")
            results.append(r)
            break
    else:
        logger.warning("Davis not found")

    # 3. GLASS GPCRs (needs InChIKey→SMILES join)
    glass_data = Path("data/raw/glass/glass2_reg_major.csv")
    glass_lig = Path("data/raw/glass/ligands.tsv")
    if glass_data.exists() and glass_lig.exists():
        glass = load_glass_with_smiles(glass_data, glass_lig)
        r = eval_benchmark(model, esm2, glass, drug_strongest, device, "GLASS_GPCR")
        results.append(r)
    else:
        logger.warning("GLASS data not found")

    # 4. BDB-Ki clean
    bdb_path = Path("data/raw/bindingdb_ki_no_overlap.csv")
    if bdb_path.exists():
        bdb = pd.read_csv(bdb_path)
        if "Target_ID" in bdb.columns:
            bdb = bdb.rename(columns={"Target_ID": "uniprot_id"})
        if "Drug" in bdb.columns:
            bdb = bdb.rename(columns={"Drug": "ligand_smiles"})
        r = eval_benchmark(model, esm2, bdb, drug_strongest, device, "BDB_Ki_clean")
        results.append(r)
    else:
        logger.warning("BDB-Ki clean not found")

    # 5. IDP benchmark
    bench_path = Path("data/raw/benchmark_affinity.csv")
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        if "protein_type" in bench.columns:
            idps = bench[bench.protein_type.str.lower() == "idp"]
            ordered = bench[bench.protein_type.str.lower() == "ordered"]
        else:
            idps = pd.DataFrame()
            ordered = bench

        if len(idps) > 0:
            r = eval_benchmark(model, esm2, idps, drug_strongest, device, "IDP_only")
            results.append(r)
        if len(ordered) > 0:
            r = eval_benchmark(model, esm2, ordered, drug_strongest, device, "Ordered_only")
            results.append(r)
        r = eval_benchmark(model, esm2, bench, drug_strongest, device, "IDP_bench_ALL")
        results.append(r)
    else:
        logger.warning("IDP benchmark not found")

    # Summary
    logger.info("\n=== V2-ATTENTION BENCHMARK SUMMARY (anchored only) ===")
    for r in results:
        if r:
            logger.info("%-15s  n=%-6d  AUROC=%-6s  CI=%.3f  RMSE=%.3f",
                        r["name"], r["n"],
                        f"{r['auroc']:.3f}" if r["auroc"] else "N/A",
                        r["ci"], r["rmse"])

    out_path = Path("results/v2_attn_benchmarks.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
