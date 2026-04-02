#!/usr/bin/env python3
"""Analyze V2 performance by anchor pKi quartiles on Davis and PDBbind."""
import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

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

def auroc_safe(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))

def predict_v2(model, esm2, eval_df, device):
    """Predict with oracle anchors from the dataset itself, return results with anchor_pki."""
    strongest, second = {}, {}
    valid = eval_df[eval_df.uniprot_id.isin(esm2)]
    for smi, grp in valid.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        prots, pkis = s.uniprot_id.values, s.pki.values
        strongest[smi] = (prots[0], pkis[0])
        if len(prots) > 1:
            second[smi] = (prots[1], pkis[1])

    results = []
    batch_a, batch_q, batch_d, batch_meta = [], [], [], []

    for _, row in eval_df.iterrows():
        uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
        if uid not in esm2 or smi not in strongest: continue
        anc_uid, anc_pki = strongest[smi]
        if anc_uid == uid:
            if smi not in second: continue
            anc_uid, anc_pki = second[smi]
        if anc_uid not in esm2: continue

        batch_a.append(esm2[anc_uid])
        batch_q.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        batch_meta.append({"uid": uid, "smi": smi, "true_pki": pki, "anchor_pki": anc_pki})

        if len(batch_a) >= 512:
            at = torch.stack(batch_a).to(device)
            qt = torch.stack(batch_q).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)
                for k, pred in enumerate(out["pki_pred"].cpu().tolist()):
                    batch_meta[k]["pred_pki"] = pred
                    results.append(batch_meta[k])
            batch_a, batch_q, batch_d, batch_meta = [], [], [], []

    if batch_a:
        at = torch.stack(batch_a).to(device)
        qt = torch.stack(batch_q).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)
            for k, pred in enumerate(out["pki_pred"].cpu().tolist()):
                batch_meta[k]["pred_pki"] = pred
                results.append(batch_meta[k])

    return pd.DataFrame(results)


def report_quartiles(name, rdf):
    print(f"\n{'='*85}")
    print(f"{name} -- V2 performance by ANCHOR pKi QUARTILE")
    print(f"{'='*85}")
    print(f"{'Quartile':<20} {'Anchor pKi':<18} {'n':<8} {'AUROC':<8} {'CI':<8} {'RMSE':<8}")
    print("-" * 85)

    rdf["anchor_q"] = pd.qcut(rdf.anchor_pki, 4, labels=["Q1 weakest", "Q2", "Q3", "Q4 strongest"])
    for q in ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]:
        sub = rdf[rdf.anchor_q == q]
        lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
        trues = sub.true_pki.values
        preds = sub.pred_pki.values
        auroc = auroc_safe(trues, preds)
        ci = ci_fn(trues, preds)
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
        print(f"{q:<20} [{lo:.1f} - {hi:.1f}]{'':<6} {len(sub):<8} {auroc:<8.3f} {ci:<8.3f} {rmse:<8.3f}")

    # Also report gap quartiles
    rdf["gap"] = rdf.anchor_pki - rdf.true_pki
    rdf["gap_q"] = pd.qcut(rdf.gap, 4, labels=["Q1 small gap", "Q2", "Q3", "Q4 large gap"], duplicates="drop")
    print(f"\n{'Quartile':<20} {'Gap range':<18} {'n':<8} {'AUROC':<8} {'CI':<8} {'RMSE':<8}")
    print("-" * 85)
    for q in rdf.gap_q.cat.categories:
        sub = rdf[rdf.gap_q == q]
        lo, hi = sub.gap.min(), sub.gap.max()
        trues = sub.true_pki.values
        preds = sub.pred_pki.values
        auroc = auroc_safe(trues, preds)
        ci = ci_fn(trues, preds)
        rmse = float(np.sqrt(np.mean((trues - preds) ** 2)))
        print(f"{q:<20} [{lo:.1f} - {hi:.1f}]{'':<6} {len(sub):<8} {auroc:<8.3f} {ci:<8.3f} {rmse:<8.3f}")


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
    model = AnchorTransferDTAv2(esm2_dim=480).to(device)
    ck = torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    esm2 = {}
    for p in ["data/processed/esm2_35m_dtc_proteins.pt", "data/processed/esm2_35m_davis.pt",
              "data/processed/esm2_35m_benchmark.pt", "data/processed/esm2_35m_pdbbind.pt"]:
        if Path(p).exists():
            esm2.update(torch.load(p, map_location="cpu", weights_only=False))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    print(f"ESM-2 embeddings: {len(esm2)}")

    # Davis
    davis = pd.read_csv("data/raw/davis/davis_benchmark.csv")
    davis = davis.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"})
    davis = davis[davis.uniprot_id.isin(esm2)]
    rdf_davis = predict_v2(model, esm2, davis, device)
    print(f"Davis: {len(rdf_davis)} predictions")
    report_quartiles("DAVIS", rdf_davis)

    # PDBbind
    pdb = pd.read_csv("data/raw/pdbbind_benchmark.csv")
    pdb = pdb[pdb.uniprot_id.isin(esm2)]
    rdf_pdb = predict_v2(model, esm2, pdb, device)
    print(f"\nPDBbind: {len(rdf_pdb)} predictions")
    report_quartiles("PDBbind", rdf_pdb)

    # Save
    json.dump({
        "davis": rdf_davis.to_dict(orient="records"),
        "pdbbind": rdf_pdb.to_dict(orient="records"),
    }, open("results/anchor_quartile_analysis.json", "w"), indent=2, default=str)
    print("\nSaved to results/anchor_quartile_analysis.json")

if __name__ == "__main__":
    main()
