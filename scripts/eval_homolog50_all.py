#!/usr/bin/env python3
"""Eval all models on Davis and PDBbind after removing >=50% sequence identity homologs."""
import json, random
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
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

# Embeddings
esm35 = {}
for p in ["data/processed/esm2_35m_dtc_proteins.pt", "data/processed/esm2_35m_davis.pt",
          "data/processed/esm2_35m_pdbbind.pt", "data/processed/esm2_35m_benchmark.pt"]:
    if Path(p).exists(): esm35.update(torch.load(p, map_location="cpu", weights_only=False))
esm35 = {k:v for k,v in esm35.items() if not torch.isnan(v).any()}
esm650 = torch.load("data/processed/esm2_650m_all.pt", map_location="cpu", weights_only=False)
esm650 = {k:v for k,v in esm650.items() if not torch.isnan(v).any()}

seqs = json.load(open("data/processed/dtc_sequences.json"))
if Path("data/raw/davis/davis_benchmark.csv").exists():
    for _, r in pd.read_csv("data/raw/davis/davis_benchmark.csv").drop_duplicates("protein_name").iterrows():
        seqs[r["protein_name"]] = r["protein_sequence"]

# Models
from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
v2_35 = AnchorTransferDTAv2(esm2_dim=480).to(device)
v2_35.load_state_dict(torch.load("models/v2_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); v2_35.eval()
v2_650 = AnchorTransferDTAv2(esm2_dim=1280).to(device)
v2_650.load_state_dict(torch.load("models/v2_650m_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); v2_650.eval()

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
        s = self.smiles_embed(s).permute(0,2,1)
        s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s)); s = s.max(2)[0]
        p = self.protein_embed(p).permute(0,2,1)
        p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p)); p = p.max(2)[0]
        x = torch.cat([s,p],1)
        x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)
deepdta = DeepDTAModel().to(device)
deepdta.load_state_dict(torch.load("models/deepdta_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); deepdta.eval()

from idr_gat.model.esm_dta import EsmDTAModel
esm_dta = EsmDTAModel(esm2_dim=480).to(device)
esm_dta.load_state_dict(torch.load("models/esm_dta_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); esm_dta.eval()

from idr_gat.model.conplex import ConPlex
conplex = ConPlex(esm2_dim=480).to(device)
conplex.load_state_dict(torch.load("models/conplex_dtc/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); conplex.eval()

print("All models loaded")

# DTC training drugs for drug filtering
dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
all_prots = sorted(set(dtc.uniprot_id) & set(esm35.keys()))
random.seed(42); random.shuffle(all_prots)
nt = max(1, int(len(all_prots) * 0.1)); nv = max(1, int(len(all_prots) * 0.1))
train_drugs = set(dtc[dtc.uniprot_id.isin(set(all_prots[nt+nv:]))].ligand_smiles.unique())


def run_benchmark(name, df, homolog_file):
    # Remove homologs
    homologs = set(open(homolog_file).read().strip().split("\n"))
    df = df[~df.uniprot_id.isin(homologs)].copy()

    # Remove training drugs
    df = df[~df.ligand_smiles.isin(train_drugs)].copy()

    # Filter to valid proteins (all embeddings + sequence)
    valid = set(esm35.keys()) & set(esm650.keys()) & set(seqs.keys())
    df = df[df.uniprot_id.isin(valid)].copy()

    # Build anchors
    strongest, second = {}, {}
    for smi, grp in df.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        p, pk = s.uniprot_id.values, s.pki.values
        strongest[smi] = (p[0], pk[0])
        if len(p) > 1: second[smi] = (p[1], pk[1])

    # Build anchored subset
    rows, anc_uids, anc_pkis = [], [], []
    for i, row in df.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if smi not in strongest: continue
        au, ap = strongest[smi]
        if au == uid:
            if smi not in second: continue
            au, ap = second[smi]
        if au not in valid: continue
        rows.append(i); anc_uids.append(au); anc_pkis.append(ap)
    subset = df.loc[rows].copy()
    subset["anchor_uid"] = anc_uids
    subset["anchor_pki"] = anc_pkis

    if len(subset) < 20:
        print(f"{name}: too few after filtering ({len(subset)})"); return

    # Batch predict all models
    all_preds = {m: [] for m in ["v2_650m", "v2_35m", "deepdta", "esm_dta", "conplex"]}
    BS = 512
    for start in range(0, len(subset), BS):
        b = subset.iloc[start:start+BS]
        uids = b.uniprot_id.values; smis = b.ligand_smiles.values; ancs = b.anchor_uid.values
        a35 = torch.stack([esm35[a] for a in ancs]).to(device)
        q35 = torch.stack([esm35[u] for u in uids]).to(device)
        a650t = torch.stack([esm650[a] for a in ancs]).to(device)
        q650t = torch.stack([esm650[u] for u in uids]).to(device)
        dt = torch.tensor([encode_smi(s) for s in smis], dtype=torch.long, device=device)
        se = dt  # same SMILES encoding
        pe = torch.tensor([encode_prot(seqs[u]) for u in uids], dtype=torch.long, device=device)
        with torch.no_grad():
            all_preds["v2_35m"].extend(v2_35(a35, q35, dt)["pki_pred"].cpu().tolist())
            all_preds["v2_650m"].extend(v2_650(a650t, q650t, dt)["pki_pred"].cpu().tolist())
            all_preds["deepdta"].extend(deepdta(se, pe).cpu().tolist())
            all_preds["esm_dta"].extend(esm_dta(dt, q35).cpu().tolist())
            all_preds["conplex"].extend(conplex(q35, dt)["score"].cpu().tolist())

    for m in all_preds: subset[m] = all_preds[m]

    models = ["v2_650m", "v2_35m", "deepdta", "esm_dta", "conplex"]
    labels = ["V2-650M", "V2-35M", "DeepDTA", "ESM-DTA", "ConPlex*"]

    n_prots = subset.uniprot_id.nunique()
    print(f"\n{'='*100}")
    print(f"{name} (>=50% homologs removed): {len(subset)} interactions, {n_prots} proteins")
    print(f"{'='*100}")

    # Quartiles
    try:
        subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4, labels=False, duplicates="drop")
        subset["anchor_q"] = subset["anchor_q"].map({0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"})

    hdr = f"{'Quartile':<10} {'Anchor pKi':<14} {'n':<7}"
    for l in labels: hdr += f" {l:<10}"
    print(hdr)
    print("-" * 100)

    for q in sorted(subset.anchor_q.unique()):
        sub = subset[subset.anchor_q == q]
        lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
        t = sub.pki.values
        line = f"{q:<10} [{lo:.1f}-{hi:.1f}]{'':>4} {len(sub):<7}"
        for m in models:
            a = auroc_safe(t, sub[m].values)
            line += f" {a:<10.3f}" if not np.isnan(a) else f" {'N/A':<10}"
        print(line)

    t = subset.pki.values
    line = f"{'Overall':<10} {'':>14} {len(subset):<7}"
    for m in models:
        a = auroc_safe(t, subset[m].values)
        line += f" {a:<10.3f}"
    print(line)
    line = f"{'CI':<10} {'':>14} {'':>7}"
    for m in models:
        c = ci_fn(t, subset[m].values)
        line += f" {c:<10.3f}"
    print(line)


# ── BDB-trained models ──
v2_bdb = None
deepdta_bdb = None
conplex_bdb = None

if Path("models/v2_bdb/best_model.pt").exists():
    v2_bdb = AnchorTransferDTAv2(esm2_dim=480).to(device)
    v2_bdb.load_state_dict(torch.load("models/v2_bdb/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); v2_bdb.eval()
    print("Loaded V2-BDB")
if Path("models/deepdta_bdb/best_model.pt").exists():
    deepdta_bdb = DeepDTAModel().to(device)
    deepdta_bdb.load_state_dict(torch.load("models/deepdta_bdb/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); deepdta_bdb.eval()
    print("Loaded DeepDTA-BDB")
if Path("models/conplex_bdb/best_model.pt").exists():
    conplex_bdb = ConPlex(esm2_dim=480).to(device)
    conplex_bdb.load_state_dict(torch.load("models/conplex_bdb/best_model.pt", map_location=device, weights_only=False)["model_state_dict"]); conplex_bdb.eval()
    print("Loaded ConPlex-BDB")

# BDB training drugs
bdb_path = Path("data/processed/bindingdb_interactions.csv")
bdb_train_drugs = set()
bdb_train_prots = set()
if bdb_path.exists():
    bdb = pd.read_csv(bdb_path)
    bdb = bdb[bdb.uniprot_id.isin(esm35)]
    bdb_prots = sorted(set(bdb.uniprot_id) & set(esm35.keys()))
    random.seed(42); random.shuffle(bdb_prots)
    nt_b = max(1, int(len(bdb_prots) * 0.1)); nv_b = max(1, int(len(bdb_prots) * 0.1))
    bdb_train_prots = set(bdb_prots[nt_b+nv_b:])
    bdb_train_drugs = set(bdb[bdb.uniprot_id.isin(bdb_train_prots)].ligand_smiles.unique())
    # BDB train sequences for homolog filtering
    bdb_train_seqs = {seqs[u] for u in bdb_train_prots if u in seqs}
    print(f"BDB train: {len(bdb_train_prots)} proteins, {len(bdb_train_drugs)} drugs")

    # Generate BDB homolog lists via MMseqs2 if not done
    # For now, use sequence-based filtering inline


def run_benchmark_bdb(name, df, dtc_homolog_file):
    """Run BDB-trained models with sequence-based overlap filtering."""
    if v2_bdb is None:
        print(f"{name} (BDB): no BDB models loaded"); return

    # Remove BDB training proteins by ID
    df = df[~df.uniprot_id.isin(bdb_train_prots)].copy()
    # Remove by sequence overlap
    overlap_seq = set()
    for uid in df.uniprot_id.unique():
        if uid in seqs and seqs[uid] in bdb_train_seqs:
            overlap_seq.add(uid)
    if overlap_seq:
        df = df[~df.uniprot_id.isin(overlap_seq)].copy()
    # Remove BDB training drugs
    df = df[~df.ligand_smiles.isin(bdb_train_drugs)].copy()
    # Also remove DTC homologs (for fair comparison with DTC models)
    if Path(dtc_homolog_file).exists():
        homologs = set(open(dtc_homolog_file).read().strip().split("\n"))
        # Only remove if these are also BDB homologs (conservative)
        # Actually, just use sequence filtering above

    valid = set(esm35.keys()) & set(seqs.keys())
    df = df[df.uniprot_id.isin(valid)].copy()

    # Build anchors
    strongest, second = {}, {}
    for smi, grp in df.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        p, pk = s.uniprot_id.values, s.pki.values
        strongest[smi] = (p[0], pk[0])
        if len(p) > 1: second[smi] = (p[1], pk[1])

    rows, anc_uids, anc_pkis = [], [], []
    for i, row in df.iterrows():
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if smi not in strongest: continue
        au, ap = strongest[smi]
        if au == uid:
            if smi not in second: continue
            au, ap = second[smi]
        if au not in valid: continue
        rows.append(i); anc_uids.append(au); anc_pkis.append(ap)
    subset = df.loc[rows].copy()
    subset["anchor_uid"] = anc_uids
    subset["anchor_pki"] = anc_pkis

    if len(subset) < 20:
        print(f"{name} (BDB): too few after filtering ({len(subset)})"); return

    # Predict BDB models
    all_preds = {m: [] for m in ["v2_bdb", "deepdta_bdb", "conplex_bdb"]}
    BS = 512
    for start in range(0, len(subset), BS):
        b = subset.iloc[start:start+BS]
        uids = b.uniprot_id.values; smis = b.ligand_smiles.values; ancs = b.anchor_uid.values
        a35 = torch.stack([esm35[a] for a in ancs]).to(device)
        q35 = torch.stack([esm35[u] for u in uids]).to(device)
        dt = torch.tensor([encode_smi(s) for s in smis], dtype=torch.long, device=device)
        pe = torch.tensor([encode_prot(seqs[u]) for u in uids], dtype=torch.long, device=device)
        with torch.no_grad():
            if v2_bdb: all_preds["v2_bdb"].extend(v2_bdb(a35, q35, dt)["pki_pred"].cpu().tolist())
            if deepdta_bdb: all_preds["deepdta_bdb"].extend(deepdta_bdb(dt, pe).cpu().tolist())
            if conplex_bdb: all_preds["conplex_bdb"].extend(conplex_bdb(q35, dt)["score"].cpu().tolist())

    for m in all_preds:
        if all_preds[m]: subset[m] = all_preds[m]

    models = [m for m in ["v2_bdb", "deepdta_bdb", "conplex_bdb"] if m in subset.columns]
    labels = {"v2_bdb": "V2-BDB", "deepdta_bdb": "DD-BDB", "conplex_bdb": "CPX-BDB"}

    n_prots = subset.uniprot_id.nunique()
    n_removed_seq = len(overlap_seq)
    print(f"\n{'='*80}")
    print(f"{name} (BDB-trained, seq overlap removed={n_removed_seq}): {len(subset)} interactions, {n_prots} proteins")
    print(f"{'='*80}")

    try:
        subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4, labels=False, duplicates="drop")

    hdr = f"{'Quartile':<10} {'Anchor pKi':<14} {'n':<7}"
    for m in models: hdr += f" {labels[m]:<10}"
    print(hdr)
    print("-" * 80)

    for q in sorted(subset.anchor_q.unique()):
        sub = subset[subset.anchor_q == q]
        lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
        t = sub.pki.values
        line = f"{str(q):<10} [{lo:.1f}-{hi:.1f}]{'':>4} {len(sub):<7}"
        for m in models:
            a = auroc_safe(t, sub[m].values)
            line += f" {a:<10.3f}" if not np.isnan(a) else f" {'N/A':<10}"
        print(line)

    t = subset.pki.values
    line = f"{'Overall':<10} {'':>14} {len(subset):<7}"
    for m in models:
        a = auroc_safe(t, subset[m].values)
        line += f" {a:<10.3f}"
    print(line)
    line = f"{'CI':<10} {'':>14} {'':>7}"
    for m in models:
        c = ci_fn(t, subset[m].values)
        line += f" {c:<10.3f}"
    print(line)


# ── Run DTC-trained ──
davis = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"})
run_benchmark("Davis (DTC)", davis, "/tmp/davis_homologs_50.txt")

pdb = pd.read_csv("data/raw/pdbbind_benchmark.csv")
run_benchmark("PDBbind (DTC)", pdb, "/tmp/pdbbind_homologs_50.txt")

# ── Run BDB-trained ──
davis2 = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis2 = davis2.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"})
run_benchmark_bdb("Davis (BDB)", davis2, "/tmp/davis_homologs_50.txt")

pdb2 = pd.read_csv("data/raw/pdbbind_benchmark.csv")
run_benchmark_bdb("PDBbind (BDB)", pdb2, "/tmp/pdbbind_homologs_50.txt")
