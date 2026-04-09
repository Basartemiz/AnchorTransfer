#!/usr/bin/env python3
"""V2-35M vs V2-650M quartile comparison on Davis and PDBbind."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch
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
    if n*(n-1)//2 > 100000:
        i = np.random.randint(0,n,100000); j = np.random.randint(0,n,100000)
        m = i!=j; i,j = i[m],j[m]
    else:
        idx = np.triu_indices(n,k=1); i,j = idx
    dt=yt[i]-yt[j]; dp=yp[i]-yp[j]; t=dt==0
    return float(((dt*dp)>0).sum()/(~t).sum()) if (~t).sum()>0 else 0.5
def auroc_safe(t, p):
    b=t>=7; nb=t<=5; m=b|nb
    if m.sum()==0 or b[m].sum()==0 or nb[m].sum()==0: return float("nan")
    return float(roc_auc_score(b[m].astype(int), p[m]))

random.seed(42); np.random.seed(42); torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
v2_650 = AnchorTransferDTAv2(esm2_dim=1280).to(device)
ck = torch.load("models/v2_650m_dtc/best_model.pt", map_location=device, weights_only=False)
v2_650.load_state_dict(ck["model_state_dict"]); v2_650.eval()

v2_35 = AnchorTransferDTAv2(esm2_dim=480).to(device)
ck = torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)
v2_35.load_state_dict(ck["model_state_dict"]); v2_35.eval()

esm35 = {}
for p in ["data/processed/esm2_35m_dtc_proteins.pt", "data/processed/esm2_35m_davis.pt",
          "data/processed/esm2_35m_pdbbind.pt", "data/processed/esm2_35m_benchmark.pt"]:
    if Path(p).exists(): esm35.update(torch.load(p, map_location="cpu", weights_only=False))
esm35 = {k:v for k,v in esm35.items() if not torch.isnan(v).any()}

esm650 = torch.load("data/processed/esm2_650m_all.pt", map_location="cpu", weights_only=False)
esm650 = {k:v for k,v in esm650.items() if not torch.isnan(v).any()}
print(f"ESM-2 35M: {len(esm35)}, 650M: {len(esm650)}")


def run_benchmark(name, df):
    valid = set(esm35.keys()) & set(esm650.keys())
    df = df[df.uniprot_id.isin(valid)].copy()

    strongest, second = {}, {}
    for smi, grp in df.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        p, pk = s.uniprot_id.values, s.pki.values
        strongest[smi] = (p[0], pk[0])
        if len(p) > 1: second[smi] = (p[1], pk[1])

    batch_a35, batch_q35, batch_a650, batch_q650, batch_d, batch_meta = [], [], [], [], [], []
    results = []

    def flush():
        nonlocal batch_a35, batch_q35, batch_a650, batch_q650, batch_d, batch_meta
        if not batch_a35: return
        dt = torch.tensor(batch_d, dtype=torch.long, device=device)
        with torch.no_grad():
            o35 = v2_35(torch.stack(batch_a35).to(device), torch.stack(batch_q35).to(device), dt)
            o650 = v2_650(torch.stack(batch_a650).to(device), torch.stack(batch_q650).to(device), dt)
        for k in range(len(batch_meta)):
            batch_meta[k]["pred_35m"] = o35["pki_pred"][k].item()
            batch_meta[k]["pred_650m"] = o650["pki_pred"][k].item()
            results.append(batch_meta[k])
        batch_a35, batch_q35, batch_a650, batch_q650, batch_d, batch_meta = [], [], [], [], [], []

    for _, row in df.iterrows():
        uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
        if smi not in strongest: continue
        au, ap = strongest[smi]
        if au == uid:
            if smi not in second: continue
            au, ap = second[smi]
        if au not in esm35 or au not in esm650: continue
        batch_a35.append(esm35[au]); batch_q35.append(esm35[uid])
        batch_a650.append(esm650[au]); batch_q650.append(esm650[uid])
        batch_d.append(encode_smi(smi))
        batch_meta.append({"uid": uid, "smi": smi, "pki": pki, "anchor_pki": ap})
        if len(batch_a35) >= 512: flush()
    flush()

    rdf = pd.DataFrame(results)
    if len(rdf) == 0:
        print(f"{name}: no predictions"); return

    rdf["anchor_q"] = pd.qcut(rdf.anchor_pki, 4, labels=["Q1", "Q2", "Q3", "Q4"])

    print(f"\n{'='*85}")
    print(f"{name}: {len(rdf)} predictions")
    print(f"{'='*85}")
    header = f"{'Quartile':<10} {'Anchor pKi':<16} {'n':<7} {'V2-35M':<10} {'V2-650M':<10} {'35M CI':<10} {'650M CI'}"
    print(header)
    print("-" * 85)

    for q in ["Q1", "Q2", "Q3", "Q4"]:
        sub = rdf[rdf.anchor_q == q]
        lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
        t = sub.pki.values
        a35 = auroc_safe(t, sub.pred_35m.values)
        a650 = auroc_safe(t, sub.pred_650m.values)
        c35 = ci_fn(t, sub.pred_35m.values)
        c650 = ci_fn(t, sub.pred_650m.values)
        print(f"{q:<10} [{lo:.1f}-{hi:.1f}]{'':>6} {len(sub):<7} {a35:<10.3f} {a650:<10.3f} {c35:<10.3f} {c650:.3f}")

    t = rdf.pki.values
    a35_all = auroc_safe(t, rdf.pred_35m.values)
    a650_all = auroc_safe(t, rdf.pred_650m.values)
    c35_all = ci_fn(t, rdf.pred_35m.values)
    c650_all = ci_fn(t, rdf.pred_650m.values)
    print(f"{'Overall':<10} {'':>16} {len(rdf):<7} {a35_all:<10.3f} {a650_all:<10.3f} {c35_all:<10.3f} {c650_all:.3f}")


# Davis
davis = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"})
run_benchmark("Davis", davis)

# PDBbind
pdb = pd.read_csv("data/raw/pdbbind_benchmark.csv")
run_benchmark("PDBbind", pdb)

# IDP benchmark
if Path("data/raw/benchmark_affinity.csv").exists():
    bench = pd.read_csv("data/raw/benchmark_affinity.csv")
    run_benchmark("IDP_ALL", bench)
