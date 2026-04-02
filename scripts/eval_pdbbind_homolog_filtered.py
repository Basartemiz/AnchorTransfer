#!/usr/bin/env python3
"""PDBbind evaluation with homolog filtering at 30%, 40%, 50% identity thresholds.

Removes PDBbind proteins that have ANY homolog >= threshold in DTC training set.
Runs V2 (oracle, random) + baselines on same anchored subset per threshold.

Usage:
  PYTHONPATH=src python scripts/eval_pdbbind_homolog_filtered.py --device cuda
"""
import json, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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


def batch_v2(model, esm2, subset, device):
    preds = []
    batch_a, batch_q, batch_d = [], [], []
    for _, row in subset.iterrows():
        uid = row["uniprot_id"]
        anc = row["anchor_uid"]
        smi = row["ligand_smiles"]
        if uid not in esm2 or anc not in esm2 or anc == uid:
            preds.append(None); continue
        batch_a.append(esm2[anc]); batch_q.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        if len(batch_a) >= 512:
            at = torch.stack(batch_a).to(device)
            qt = torch.stack(batch_q).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)
                preds.extend(out["pki_pred"].cpu().tolist())
            batch_a, batch_q, batch_d = [], [], []
    if batch_a:
        at = torch.stack(batch_a).to(device)
        qt = torch.stack(batch_q).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)
            preds.extend(out["pki_pred"].cpu().tolist())
    return preds


def batch_v2_random(model, esm2, subset, train_prots_list, device, rng):
    preds = []
    batch_a, batch_q, batch_d = [], [], []
    for _, row in subset.iterrows():
        uid = row["uniprot_id"]
        smi = row["ligand_smiles"]
        if uid not in esm2:
            preds.append(None); continue
        anc = rng.choice(train_prots_list)
        attempts = 0
        while anc == uid and attempts < 20:
            anc = rng.choice(train_prots_list); attempts += 1
        batch_a.append(esm2[anc]); batch_q.append(esm2[uid])
        batch_d.append(encode_smi(smi))
        if len(batch_a) >= 512:
            at = torch.stack(batch_a).to(device)
            qt = torch.stack(batch_q).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                out = model(at, qt, dt)
                preds.extend(out["pki_pred"].cpu().tolist())
            batch_a, batch_q, batch_d = [], [], []
    if batch_a:
        at = torch.stack(batch_a).to(device)
        qt = torch.stack(batch_q).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(at, qt, dt)
            preds.extend(out["pki_pred"].cpu().tolist())
    return preds


def batch_deepdta(model, seqs, subset, device):
    preds = []
    batch_s, batch_p = [], []
    for _, row in subset.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if uid not in seqs:
            preds.append(None); continue
        batch_s.append(encode_smi(smi)); batch_p.append(encode_prot(seqs[uid]))
        if len(batch_s) >= 512:
            st = torch.tensor(batch_s, dtype=torch.long, device=device)
            pt = torch.tensor(batch_p, dtype=torch.long, device=device)
            with torch.no_grad(): preds.extend(model(st, pt).cpu().tolist())
            batch_s, batch_p = [], []
    if batch_s:
        st = torch.tensor(batch_s, dtype=torch.long, device=device)
        pt = torch.tensor(batch_p, dtype=torch.long, device=device)
        with torch.no_grad(): preds.extend(model(st, pt).cpu().tolist())
    return preds


def batch_esm_model(model, name, esm2, subset, device):
    preds = []
    batch_p, batch_d = [], []
    for _, row in subset.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if uid not in esm2:
            preds.append(None); continue
        batch_p.append(esm2[uid]); batch_d.append(encode_smi(smi))
        if len(batch_p) >= 512:
            pt = torch.stack(batch_p).to(device)
            dt = torch.tensor(batch_d, dtype=torch.long, device=device)
            with torch.no_grad():
                if name == "conplex":
                    preds.extend(model(pt, dt)["score"].cpu().tolist())
                else:
                    preds.extend(model(dt, pt).cpu().tolist())
            batch_p, batch_d = [], []
    if batch_p:
        pt = torch.stack(batch_p).to(device)
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            if name == "conplex":
                preds.extend(model(pt, dt)["score"].cpu().tolist())
            else:
                preds.extend(model(dt, pt).cpu().tolist())
    return preds


def main():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    rng = random.Random(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ESM-2
    esm2 = {}
    for p in ["data/processed/esm2_35m_dtc_proteins.pt", "data/processed/esm2_35m_pdbbind.pt"]:
        if Path(p).exists():
            esm2.update(torch.load(p, map_location="cpu", weights_only=False))
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}

    seqs = json.load(open("data/processed/dtc_sequences.json"))

    # DTC training proteins for random anchor pool
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    all_prots = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    random.seed(42); random.shuffle(all_prots)
    nt = max(1, int(len(all_prots) * 0.1))
    nv = max(1, int(len(all_prots) * 0.1))
    train_prots_list = [p for p in all_prots[nt + nv:] if p in esm2]
    train_drugs = set(dtc[dtc.uniprot_id.isin(set(train_prots_list))].ligand_smiles.unique())

    # Load models
    from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
    v2 = AnchorTransferDTAv2(esm2_dim=480).to(device)
    ck = torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)
    v2.load_state_dict(ck["model_state_dict"]); v2.eval()

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

    from idr_gat.model.esm_dta import EsmDTAModel
    esm_dta = EsmDTAModel(esm2_dim=480).to(device)
    ck = torch.load("models/esm_dta_dtc/best_model.pt", map_location=device, weights_only=False)
    esm_dta.load_state_dict(ck["model_state_dict"]); esm_dta.eval()

    from idr_gat.model.conplex import ConPlex
    conplex = ConPlex(esm2_dim=480).to(device)
    ck = torch.load("models/conplex_dtc/best_model.pt", map_location=device, weights_only=False)
    conplex.load_state_dict(ck["model_state_dict"]); conplex.eval()

    # Load PDBbind
    pdb = pd.read_csv("data/raw/pdbbind_benchmark.csv")

    # Filter training drugs
    pdb = pdb[~pdb.ligand_smiles.isin(train_drugs)]

    # Load homolog lists
    homolog_files = {
        30: "/tmp/pdbbind_homologs_30.txt",
        40: "/tmp/pdbbind_homologs_40.txt",
        50: "/tmp/pdbbind_homologs_50.txt",
    }

    results = {}

    for thresh, hfile in sorted(homolog_files.items()):
        if not Path(hfile).exists():
            print(f"Missing {hfile}, skipping {thresh}%")
            continue

        homologs = set(open(hfile).read().strip().split("\n"))
        df = pdb[~pdb.uniprot_id.isin(homologs)].copy()

        # Filter to proteins with ESM-2 + sequence
        valid = set(esm2.keys()) & set(seqs.keys())
        df = df[df.uniprot_id.isin(valid)].copy()

        # Build dataset-internal anchors
        strongest, second = {}, {}
        for smi, grp in df.groupby("ligand_smiles"):
            s = grp.sort_values("pki", ascending=False)
            prots, pkis = s.uniprot_id.values, s.pki.values
            strongest[smi] = (prots[0], pkis[0])
            if len(prots) > 1: second[smi] = (prots[1], pkis[1])

        # Build anchored subset
        rows, anchor_uids, anchor_pkis = [], [], []
        for i, row in df.iterrows():
            uid, smi = row["uniprot_id"], row["ligand_smiles"]
            if smi not in strongest: continue
            anc_uid, anc_pki = strongest[smi]
            if anc_uid == uid:
                if smi not in second: continue
                anc_uid, anc_pki = second[smi]
            if anc_uid not in esm2: continue
            rows.append(i); anchor_uids.append(anc_uid); anchor_pkis.append(anc_pki)

        subset = df.loc[rows].copy()
        subset["anchor_uid"] = anchor_uids
        subset["anchor_pki"] = anchor_pkis

        # Predict all models
        subset["v2_oracle"] = batch_v2(v2, esm2, subset, device)
        subset["v2_random"] = batch_v2_random(v2, esm2, subset, train_prots_list, device, rng)
        subset["deepdta_pred"] = batch_deepdta(deepdta, seqs, subset, device)
        subset["esm_dta_pred"] = batch_esm_model(esm_dta, "esm_dta", esm2, subset, device)
        subset["conplex_pred"] = batch_esm_model(conplex, "conplex", esm2, subset, device)

        # Drop rows where any model has None
        pred_cols = ["v2_oracle", "v2_random", "deepdta_pred", "esm_dta_pred", "conplex_pred"]
        for c in pred_cols:
            subset[c] = pd.to_numeric(subset[c], errors="coerce")
        subset = subset.dropna(subset=pred_cols)

        n_prots = subset.uniprot_id.nunique()
        print(f"\n{'='*90}")
        print(f"PDBbind @ {thresh}% homolog filter: {len(subset)} interactions, {n_prots} proteins")
        print(f"{'='*90}")

        # Quartile analysis
        subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4,
                                      labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")

        print(f"{'Quartile':<10} {'Anchor pKi':<16} {'n':<7} | {'V2 orac':<8} {'V2 rand':<8} {'DeepDTA':<8} {'ESM-DTA':<8} {'ConPlex':<8}")
        print("-" * 95)

        thresh_results = []
        for q in subset.anchor_q.cat.categories:
            sub = subset[subset.anchor_q == q]
            lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
            trues = sub.pki.values
            row_r = {"quartile": str(q), "anchor_range": f"[{lo:.1f}-{hi:.1f}]", "n": len(sub)}
            vals = []
            for col, name in [("v2_oracle", "V2_oracle"), ("v2_random", "V2_random"),
                              ("deepdta_pred", "DeepDTA"), ("esm_dta_pred", "ESM-DTA"),
                              ("conplex_pred", "ConPlex")]:
                p = sub[col].values
                a = auroc_safe(trues, p)
                row_r[name] = a
                vals.append(f"{a:.3f}" if not np.isnan(a) else "  N/A")
            print(f"{q:<10} [{lo:.1f}-{hi:.1f}]{'':<6} {len(sub):<7} | {' '.join(f'{v:<8}' for v in vals)}")
            thresh_results.append(row_r)

        # Overall
        print(f"\n  Overall (n={len(subset)}):")
        for col, name in [("v2_oracle", "V2_oracle"), ("v2_random", "V2_random"),
                          ("deepdta_pred", "DeepDTA"), ("esm_dta_pred", "ESM-DTA"),
                          ("conplex_pred", "ConPlex")]:
            p = subset[col].values; t = subset.pki.values
            a = auroc_safe(t, p); c = ci_fn(t, p)
            print(f"    {name:<12} AUROC={a:.3f} CI={c:.3f}")

        results[f"thresh_{thresh}"] = thresh_results

    json.dump(results, open("results/pdbbind_homolog_filtered.json", "w"), indent=2, default=str)
    print("\nSaved to results/pdbbind_homolog_filtered.json")

if __name__ == "__main__":
    main()
