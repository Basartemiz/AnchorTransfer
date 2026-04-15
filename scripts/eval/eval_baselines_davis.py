#!/usr/bin/env python3
"""V2-650M vs V2-35M vs baselines on Davis — paper Table 2 protocol.

All models evaluated on the SAME Tanimoto-anchored subset.
Anchors retrieved from DTC training set by chirality-aware Morgan Tanimoto
after canonical SMILES duplicate exclusion (paper protocol).
AUROC: >=7 positive, <=5 negative, ambiguous excluded.
"""
import json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

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
        i = np.random.randint(0, n, 100000); j = np.random.randint(0, n, 100000)
        m = i != j; i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1); i, j = idx
    dt = yt[i]-yt[j]; dp = yp[i]-yp[j]; t = dt == 0
    return float(((dt*dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

def auroc_safe(t, p):
    b = t >= 7; nb = t <= 5; m = b | nb
    if m.sum() == 0 or b[m].sum() == 0 or nb[m].sum() == 0: return float("nan")
    return float(roc_auc_score(b[m].astype(int), p[m]))

def auprc_safe(t, p):
    b = t >= 7; nb = t <= 5; m = b | nb
    if m.sum() == 0 or b[m].sum() == 0 or nb[m].sum() == 0: return float("nan")
    return float(average_precision_score(b[m].astype(int), p[m]))


SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT = Path(__file__).resolve().parents[2]

# ── Load ESM-2 embeddings ──
esm35 = torch.load(PROJECT / "data/processed/esm2_35m_dtc.pt", map_location="cpu", weights_only=False)
bench35 = torch.load(PROJECT / "data/processed/esm2_35m_benchmark.pt", map_location="cpu", weights_only=False)
esm35.update(bench35)
esm35 = {k: v for k, v in esm35.items() if torch.isfinite(v).all()}

esm650 = torch.load(PROJECT / "data/processed/esm2_650m_dtc.pt", map_location="cpu", weights_only=False)
bench650 = torch.load(PROJECT / "data/processed/esm2_650m_benchmark.pt", map_location="cpu", weights_only=False)
esm650.update(bench650)
esm650 = {k: v for k, v in esm650.items() if torch.isfinite(v).all()}

# DTC + Sequences
dtc = pd.read_csv(PROJECT / "data/processed/dtc_training_interactions.csv")
seqs = json.load(open(PROJECT / "data/processed/merged_sequences.json"))
davis_csv = PROJECT / "data/raw/davis_benchmark.csv"
for _, r in pd.read_csv(davis_csv).drop_duplicates("protein_name").iterrows():
    seqs[r["protein_name"]] = r["protein_sequence"]

log.info(f"ESM-2 35M: {len(esm35)}, 650M: {len(esm650)}, sequences: {len(seqs)}")

# ── Load models ──
from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
from anchor_transfer.model.esm_dta import EsmDTAModel

models_loaded = {}

# V2-650M
v2_650_path = PROJECT / "models/v2_650m/best_model.pt"
if v2_650_path.exists():
    v2_650 = AnchorTransferDTAv2(esm2_dim=1280).to(device)
    v2_650.load_state_dict(torch.load(v2_650_path, map_location=device, weights_only=False)["model_state_dict"])
    v2_650.eval(); models_loaded["v2_650m"] = v2_650; log.info("Loaded V2-650M")

# V2-35M
v2_35_path = PROJECT / "models/v2_35m/best_model.pt"
if v2_35_path.exists():
    v2_35 = AnchorTransferDTAv2(esm2_dim=480).to(device)
    v2_35.load_state_dict(torch.load(v2_35_path, map_location=device, weights_only=False)["model_state_dict"])
    v2_35.eval(); models_loaded["v2_35m"] = v2_35; log.info("Loaded V2-35M")

# DeepDTA
class DeepDTAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
        self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
        self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
        self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1); self.do = nn.Dropout(0.1)
    def forward(self, s, p):
        s = self.smiles_embed(s).permute(0, 2, 1)
        s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s)); s = s.max(2)[0]
        p = self.protein_embed(p).permute(0, 2, 1)
        p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p)); p = p.max(2)[0]
        x = torch.cat([s, p], 1)
        x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)

ddta_path = PROJECT / "models/deepdta_dtc/best_model.pt"
if ddta_path.exists():
    deepdta = DeepDTAModel().to(device)
    deepdta.load_state_dict(torch.load(ddta_path, map_location=device, weights_only=False)["model_state_dict"])
    deepdta.eval(); models_loaded["deepdta"] = deepdta; log.info("Loaded DeepDTA")

# ESM-DTA
esmdta_path = PROJECT / "models/esm_dta_dtc/best_model.pt"
if esmdta_path.exists():
    esm_dta = EsmDTAModel(esm2_dim=480).to(device)
    esm_dta.load_state_dict(torch.load(esmdta_path, map_location=device, weights_only=False)["model_state_dict"])
    esm_dta.eval(); models_loaded["esm_dta"] = esm_dta; log.info("Loaded ESM-DTA")

# ConPlex (optional)
try:
    from anchor_transfer.model.conplex import ConPlex
    cpx_path = PROJECT / "models/conplex_dtc/best_model.pt"
    if cpx_path.exists():
        conplex = ConPlex(esm2_dim=480).to(device)
        conplex.load_state_dict(torch.load(cpx_path, map_location=device, weights_only=False)["model_state_dict"])
        conplex.eval(); models_loaded["conplex"] = conplex; log.info("Loaded ConPlex")
except Exception as e:
    log.warning(f"ConPlex not loaded: {e}")

log.info(f"Models loaded: {list(models_loaded.keys())}")

# ── Load Davis ──
davis = pd.read_csv(davis_csv)
davis = davis.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles"})

valid = set(esm35.keys()) & set(esm650.keys()) & set(seqs.keys())
davis = davis[davis.uniprot_id.isin(valid)].copy()
log.info(f"Davis (valid proteins): {len(davis)} interactions, {davis.uniprot_id.nunique()} proteins")

# ── Build DTC Tanimoto anchor pool (paper protocol) ──
try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("RDKit required for Tanimoto anchor retrieval")

def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol, canonical=True) if mol else None

def smiles_to_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)

# Recreate DTC 80/10/10 protein split
dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
all_prots = sorted(set(dtc_valid.uniprot_id) & set(esm35.keys()))
rng = random.Random(SEED)
rng.shuffle(all_prots)
nt = max(1, int(len(all_prots) * 0.1))
nv = max(1, int(len(all_prots) * 0.1))
train_prots = set(all_prots[nt + nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(train_prots)]
log.info(f"DTC train split: {len(dtc_train)} interactions, {len(train_prots)} proteins")

# Canonical SMILES for Davis drugs (for exclusion)
davis_canonical = set()
for smi in davis.ligand_smiles.unique():
    c = canonicalize(smi)
    if c: davis_canonical.add(c)

# Build anchor pool: strongest binder per drug, pKi >= 7, exclude Davis canonical duplicates
anchor_pool = {}
for smi, group in dtc_train.groupby("ligand_smiles"):
    best = group.sort_values("pki", ascending=False).iloc[0]
    uid, pki = best["uniprot_id"], float(best["pki"])
    if pki < 7.0 or uid not in esm35: continue
    c = canonicalize(smi)
    if c and c in davis_canonical: continue
    fp = smiles_to_fp(smi)
    if fp is None: continue
    anchor_pool[smi] = {"uid": uid, "pki": pki, "canonical": c, "fp": fp}
log.info(f"DTC anchor pool (after canonical exclusion): {len(anchor_pool)} drugs")

# Tanimoto nearest-neighbor for each Davis drug
davis_drug_fps = {}
for smi in davis.ligand_smiles.unique():
    fp = smiles_to_fp(smi)
    if fp: davis_drug_fps[smi] = fp

anchor_fps = [(smi, meta["fp"], meta) for smi, meta in anchor_pool.items()]
nearest = {}
for davis_smi, davis_fp in davis_drug_fps.items():
    best_sim, best_meta, best_smi = -1, None, None
    for a_smi, a_fp, a_meta in anchor_fps:
        sim = DataStructs.TanimotoSimilarity(davis_fp, a_fp)
        if sim > best_sim:
            best_sim, best_meta, best_smi = sim, a_meta, a_smi
    if best_meta:
        nearest[davis_smi] = {"anchor_uid": best_meta["uid"], "anchor_pki": best_meta["pki"],
                               "tanimoto": best_sim, "anchor_drug": best_smi}
log.info(f"Tanimoto anchors: {len(nearest)}/{len(davis_drug_fps)} Davis drugs matched")

# Build self-anchor exclusion map (sequence-based)
seq_to_dtc = {}
dtc_prots_csv = PROJECT / "data/raw/dtc_proteins.csv"
if dtc_prots_csv.exists():
    for _, r in pd.read_csv(dtc_prots_csv).iterrows():
        seq_to_dtc.setdefault(r.get("sequence", ""), set()).add(str(r["uniprot_id"]))

# Build anchored subset
rows, anc_uids, anc_pkis, tanimotos = [], [], [], []
for i, row in davis.iterrows():
    smi = row["ligand_smiles"]
    if smi not in nearest: continue
    match = nearest[smi]
    au = match["anchor_uid"]
    # Self-anchor exclusion: skip if anchor protein sequence matches Davis protein
    davis_seq = davis.loc[davis.uniprot_id == row["uniprot_id"], "protein_sequence"].iloc[0] if "protein_sequence" in davis.columns else None
    if davis_seq and davis_seq in seq_to_dtc and au in seq_to_dtc[davis_seq]:
        continue
    if au not in valid: continue
    rows.append(i)
    anc_uids.append(au)
    anc_pkis.append(match["anchor_pki"])
    tanimotos.append(match["tanimoto"])

subset = davis.loc[rows].copy()
subset["anchor_uid"] = anc_uids
subset["anchor_pki"] = anc_pkis
subset["tanimoto"] = tanimotos
log.info(f"Anchored subset (Tanimoto from DTC): {len(subset)} interactions, "
         f"{subset.uniprot_id.nunique()} proteins, tanimoto mean={np.mean(tanimotos):.3f}")

# ── Batch predict all models on the SAME subset ──
BS = 512
all_preds = {m: [] for m in models_loaded}

for start in range(0, len(subset), BS):
    batch = subset.iloc[start:start+BS]
    uids = batch.uniprot_id.values
    smis = batch.ligand_smiles.values
    ancs = batch.anchor_uid.values

    dt = torch.tensor([encode_smi(s) for s in smis], dtype=torch.long, device=device)

    with torch.no_grad():
        if "v2_650m" in models_loaded:
            a650 = torch.stack([esm650[a] for a in ancs]).to(device)
            q650 = torch.stack([esm650[u] for u in uids]).to(device)
            all_preds["v2_650m"].extend(models_loaded["v2_650m"](a650, q650, dt)["pki_pred"].cpu().tolist())

        if "v2_35m" in models_loaded:
            a35 = torch.stack([esm35[a] for a in ancs]).to(device)
            q35 = torch.stack([esm35[u] for u in uids]).to(device)
            all_preds["v2_35m"].extend(models_loaded["v2_35m"](a35, q35, dt)["pki_pred"].cpu().tolist())

        if "deepdta" in models_loaded:
            se = torch.tensor([encode_smi(s) for s in smis], dtype=torch.long, device=device)
            pe = torch.tensor([encode_prot(seqs[u]) for u in uids], dtype=torch.long, device=device)
            all_preds["deepdta"].extend(models_loaded["deepdta"](se, pe).cpu().tolist())

        if "esm_dta" in models_loaded:
            pt = torch.stack([esm35[u] for u in uids]).to(device)
            all_preds["esm_dta"].extend(models_loaded["esm_dta"](dt, pt).cpu().tolist())

        if "conplex" in models_loaded:
            pt = torch.stack([esm35[u] for u in uids]).to(device)
            all_preds["conplex"].extend(models_loaded["conplex"](pt, dt)["score"].cpu().tolist())

    if (start + BS) % 5000 < BS:
        log.info(f"  Predicted {min(start+BS, len(subset))}/{len(subset)}")

for m in all_preds:
    subset[m] = all_preds[m]

# ── Results ──
subset["anchor_q"] = pd.qcut(subset.anchor_pki, 4, labels=["Q1", "Q2", "Q3", "Q4"])

model_keys = [m for m in ["v2_650m", "v2_35m", "deepdta", "esm_dta", "conplex"] if m in models_loaded]
model_labels = {"v2_650m": "V2-650M", "v2_35m": "V2-35M", "deepdta": "DeepDTA",
                "esm_dta": "ESM-DTA", "conplex": "ConPlex"}

log.info(f"\n{'='*110}")
log.info(f"Davis: {len(subset)} interactions, {subset.uniprot_id.nunique()} proteins (SAME subset for all models)")
log.info(f"{'='*110}")

# Header
hdr = f"{'Metric':<12} {'':>14} {'n':<7}"
for m in model_keys:
    hdr += f" {model_labels[m]:<12}"
log.info(hdr)
log.info("-" * 110)

# Quartile AUROC
for q in ["Q1", "Q2", "Q3", "Q4"]:
    sub = subset[subset.anchor_q == q]
    lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
    t = sub.pki.values
    line = f"{'AUROC ' + q:<12} [{lo:.1f}-{hi:.1f}]{'':>4} {len(sub):<7}"
    for m in model_keys:
        a = auroc_safe(t, sub[m].values)
        line += f" {a:<12.4f}" if not np.isnan(a) else f" {'N/A':<12}"
    log.info(line)

# Overall
t = subset.pki.values
line = f"{'AUROC':<12} {'Overall':>14} {len(subset):<7}"
for m in model_keys:
    a = auroc_safe(t, subset[m].values)
    line += f" {a:<12.4f}"
log.info(line)

# AUPRC
line = f"{'AUPRC':<12} {'Overall':>14} {len(subset):<7}"
for m in model_keys:
    a = auprc_safe(t, subset[m].values)
    line += f" {a:<12.4f}" if not np.isnan(a) else f" {'N/A':<12}"
log.info(line)

# CI
line = f"{'CI':<12} {'Overall':>14} {len(subset):<7}"
for m in model_keys:
    c = ci_fn(t, subset[m].values)
    line += f" {c:<12.4f}"
log.info(line)

# RMSE
line = f"{'RMSE':<12} {'Overall':>14} {len(subset):<7}"
for m in model_keys:
    p = subset[m].values
    r = np.sqrt(np.mean((t - p) ** 2))
    line += f" {r:<12.4f}"
log.info(line)

# Pearson r
line = f"{'Pearson r':<12} {'Overall':>14} {len(subset):<7}"
for m in model_keys:
    r = np.corrcoef(t, subset[m].values)[0, 1]
    line += f" {r:<12.4f}"
log.info(line)

# Per-protein macro CI
line = f"{'CI (macro)':<12} {'Per-protein':>14} {subset.uniprot_id.nunique():<7}"
for m in model_keys:
    cis = []
    for uid, grp in subset.groupby("uniprot_id"):
        if len(grp) >= 2:
            cis.append(ci_fn(grp.pki.values, grp[m].values))
    line += f" {np.mean(cis):<12.4f}" if cis else f" {'N/A':<12}"
log.info(line)

log.info(f"{'='*110}")
log.info("=== Evaluation complete ===")
