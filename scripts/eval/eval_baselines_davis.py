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

# Load anchor-filtered subset if available (for apples-to-apples comparison)
# This is the same subset that V2-650M was evaluated on
anchor_subset = None
v2_pred_path = PROJECT / "results" / "v2_650m" / "davis" / "predictions.csv"
if v2_pred_path.exists():
    anchor_preds = pd.read_csv(v2_pred_path)
    anchor_pairs = set(zip(anchor_preds["uniprot_id"], anchor_preds["ligand_smiles"]))
    log.info(f"Loaded anchor-filtered subset: {len(anchor_pairs)} pairs from V2-650M predictions")
else:
    anchor_pairs = None
    log.info("No anchor predictions found — evaluating on full Davis")

esm2_35m = None
for mname, mpath in [("DeepDTA", "models/deepdta_dtc/best_model.pt"),
                      ("ESM-DTA", "models/esm_dta_dtc/best_model.pt")]:
    full_path = PROJECT / mpath
    if not full_path.exists():
        log.info(f"Skipping {mname} (checkpoint not found at {full_path})")
        continue
    ckpt = torch.load(full_path, map_location=device, weights_only=False)

    BATCH_SIZE = 512
    if mname == "DeepDTA":
        model = DeepDTA().to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        log.info(f"Loaded {mname} (epoch {ckpt.get('epoch', '?')})")
        all_smi = [enc_smi(s) for s in davis.drug_smiles]
        all_prot = [enc_prot(s) for s in davis.protein_sequence]
        preds = []
        with torch.no_grad():
            for i in range(0, len(all_smi), BATCH_SIZE):
                smi_t = torch.tensor(all_smi[i:i+BATCH_SIZE]).to(device)
                prot_t = torch.tensor(all_prot[i:i+BATCH_SIZE]).to(device)
                preds.extend(model(smi_t, prot_t).cpu().tolist())
                if (i + BATCH_SIZE) % 5000 < BATCH_SIZE:
                    log.info(f"  {mname}: {min(i+BATCH_SIZE, len(all_smi))}/{len(all_smi)}")
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
        # Batch: group by protein for efficient ESM-2 lookup
        all_smi = [enc_smi(s) for s in davis.drug_smiles]
        preds = [float("nan")] * len(davis)
        valid_indices = [i for i, uid in enumerate(davis.protein_name) if uid in esm2_35m]
        with torch.no_grad():
            for start in range(0, len(valid_indices), BATCH_SIZE):
                batch_idx = valid_indices[start:start+BATCH_SIZE]
                smi_t = torch.tensor([all_smi[i] for i in batch_idx]).to(device)
                prot_t = torch.stack([esm2_35m[davis.protein_name.iloc[i]] for i in batch_idx]).to(device)
                batch_preds = model(smi_t, prot_t).cpu().tolist()
                for j, idx in enumerate(batch_idx):
                    preds[idx] = batch_preds[j]
                if (start + BATCH_SIZE) % 5000 < BATCH_SIZE:
                    log.info(f"  {mname}: {min(start+BATCH_SIZE, len(valid_indices))}/{len(valid_indices)}")

    t = davis.pki.values
    p = np.array(preds)
    valid = ~np.isnan(p)
    tv, pv = t[valid], p[valid]
    ci = ci_fn(tv, pv)
    rmse = np.sqrt(np.mean((tv - pv) ** 2))
    r = np.corrcoef(tv, pv)[0, 1] if len(tv) > 1 else 0
    # Paper protocol: >=7 positive, <=5 negative, exclude ambiguous 5-7 range
    pos_mask = tv >= 7.0
    neg_mask = tv <= 5.0
    cls_mask = pos_mask | neg_mask
    cls_labels = pos_mask[cls_mask].astype(int)
    auroc = roc_auc_score(cls_labels, pv[cls_mask]) if 0 < cls_labels.sum() < len(cls_labels) else 0
    auprc = 0
    try:
        from sklearn.metrics import average_precision_score
        auprc = average_precision_score(cls_labels, pv[cls_mask]) if 0 < cls_labels.sum() < len(cls_labels) else 0
    except Exception:
        pass
    log.info(f"{mname:20s} CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f} r={r:.4f} n={len(tv)} (full Davis, cls_n={cls_mask.sum()})")

    # Also evaluate on the anchor-filtered subset for fair comparison
    if anchor_pairs is not None:
        mask = [
            (not np.isnan(p[i])) and (davis.iloc[i].protein_name, davis.iloc[i].drug_smiles) in anchor_pairs
            for i in range(len(davis))
        ]
        if sum(mask) > 0:
            tf = t[mask]
            pf = p[mask]
            ci_f = ci_fn(tf, pf)
            rmse_f = np.sqrt(np.mean((tf - pf) ** 2))
            r_f = np.corrcoef(tf, pf)[0, 1] if len(tf) > 1 else 0
            pos_f = tf >= 7.0; neg_f = tf <= 5.0; cls_f = pos_f | neg_f
            cls_lbl_f = pos_f[cls_f].astype(int)
            auroc_f = roc_auc_score(cls_lbl_f, pf[cls_f]) if 0 < cls_lbl_f.sum() < len(cls_lbl_f) else 0
            log.info(f"{mname:20s} CI={ci_f:.4f} RMSE={rmse_f:.4f} AUROC={auroc_f:.4f} r={r_f:.4f} n={sum(mask)} (anchor-filtered, cls_n={cls_f.sum()})")

log.info("=== Baseline evaluation complete ===")
