#!/usr/bin/env python3
"""Evaluate DeepDTA and ESM-DTA baselines on Davis for comparison."""
import sys, logging, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT = Path(__file__).resolve().parents[2]
davis = pd.read_csv(PROJECT / "data" / "raw" / "davis_benchmark.csv")
log.info(f"Davis: {len(davis)} interactions, {davis.protein_name.nunique()} proteins")

CHARSMILES = {c: i+1 for i, c in enumerate("CNOSFClBrIPHcs()[]=@+\\/#-1234567890")}
def enc_smi(s, ml=200):
    return [CHARSMILES.get(c, 0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

CHARPROT = {c: i+1 for i, c in enumerate("ACBEDGFIHKMLONQPSRUTWVYXZ")}
def enc_prot(s, ml=1000):
    return [CHARPROT.get(c, 0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

def ci_fn(y, f):
    y, f = np.array(y), np.array(f)
    n = len(y)
    if n < 2: return 0.5
    idx = np.triu_indices(n, k=1); i, j = idx[0], idx[1]
    dt = y[i] - y[j]; dp = f[i] - f[j]; t = dt == 0
    return float(((dt*dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5


class DeepDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
        self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
        self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
        self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1); self.do = nn.Dropout(0.1)

    def forward(self, smi_tok, prot_tok):
        s = self.smiles_embed(smi_tok).transpose(1, 2)
        s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s))
        s = F.adaptive_max_pool1d(s, 1).squeeze(-1)
        p = self.protein_embed(prot_tok).transpose(1, 2)
        p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p))
        p = F.adaptive_max_pool1d(p, 1).squeeze(-1)
        x = torch.cat([s, p], dim=1)
        x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)


from anchor_transfer.model.esm_dta import EsmDTAModel

esm2_35m = None
for mname, mpath in [("DeepDTA", "models/deepdta_dtc/best_model.pt"),
                      ("ESM-DTA", "models/esm_dta_dtc/best_model.pt")]:
    full_path = PROJECT / mpath
    if not full_path.exists():
        log.info(f"Skipping {mname} (checkpoint not found at {full_path})")
        continue
    ckpt = torch.load(full_path, map_location=device, weights_only=False)

    if mname == "DeepDTA":
        model = DeepDTA().to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        log.info(f"Loaded {mname} (epoch {ckpt.get('epoch', '?')})")
        preds = []
        with torch.no_grad():
            for _, row in davis.iterrows():
                smi_t = torch.tensor([enc_smi(row.drug_smiles)]).to(device)
                prot_t = torch.tensor([enc_prot(row.protein_sequence)]).to(device)
                preds.append(model(smi_t, prot_t).item())
    else:
        if esm2_35m is None:
            esm2_35m = torch.load(PROJECT / "data/processed/esm2_35m_dtc.pt",
                                   map_location="cpu", weights_only=False)
            bench = torch.load(PROJECT / "data/processed/esm2_35m_benchmark.pt",
                                map_location="cpu", weights_only=False)
            esm2_35m.update(bench)
        model = EsmDTAModel(esm2_dim=480).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        log.info(f"Loaded {mname} (epoch {ckpt.get('epoch', '?')})")
        preds = []
        with torch.no_grad():
            for _, row in davis.iterrows():
                uid = row.protein_name
                if uid not in esm2_35m:
                    preds.append(float("nan")); continue
                smi_t = torch.tensor([enc_smi(row.drug_smiles)]).to(device)
                prot_emb = esm2_35m[uid].unsqueeze(0).to(device)
                preds.append(model(smi_t, prot_emb).item())

    t = davis.pki.values
    p = np.array(preds)
    valid = ~np.isnan(p)
    tv, pv = t[valid], p[valid]
    ci = ci_fn(tv, pv)
    rmse = np.sqrt(np.mean((tv - pv) ** 2))
    binary = (tv >= 7.0).astype(int)
    auroc = roc_auc_score(binary, pv) if 0 < binary.sum() < len(binary) else 0
    r = np.corrcoef(tv, pv)[0, 1] if len(tv) > 1 else 0
    log.info(f"{mname:20s} CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(tv)}")

log.info("=== Baseline evaluation complete ===")
