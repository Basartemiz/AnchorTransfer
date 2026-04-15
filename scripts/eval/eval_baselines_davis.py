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
try:
    from rdkit import Chem
    from rdkit.RDLogger import DisableLog
    DisableLog("rdApp.*")
except ImportError:
    raise SystemExit("RDKit required for canonical SMILES deduplication")

# Exclude Davis drugs that overlap with DTC training set (paper protocol)
dtc = pd.read_csv(PROJECT / "data" / "processed" / "dtc_training_interactions.csv")
def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, canonical=True) if m else None
dtc_canon = set(c for c in (canon(s) for s in dtc.ligand_smiles.unique()) if c)
davis["canonical"] = davis.drug_smiles.map(canon)
davis_filtered = davis[~davis.canonical.isin(dtc_canon)].copy()
log.info(f"Drug overlap exclusion: {len(davis)} -> {len(davis_filtered)} interactions "
         f"({davis.drug_smiles.nunique()} -> {davis_filtered.drug_smiles.nunique()} drugs, "
         f"{davis_filtered.protein_name.nunique()} proteins)")

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

    # Per-protein macro-averaged metrics on drug-filtered subset (paper protocol)
    davis[f"{mname}_pred"] = preds
    davis_eval = davis_filtered.copy()
    davis_eval[f"{mname}_pred"] = davis.loc[davis_eval.index, f"{mname}_pred"]
    davis_eval = davis_eval[~davis_eval[f"{mname}_pred"].isna()]

    from sklearn.metrics import average_precision_score

    def macro_eval(df, label):
        ci_vals, rmse_vals, auroc_vals, auprc_vals, pearson_vals = [], [], [], [], []
        for uid, group in df.groupby("protein_name"):
            true = group.pki.values
            pred = group[f"{mname}_pred"].values
            if len(true) < 2:
                continue
            ci_vals.append(ci_fn(true, pred))
            rmse_vals.append(np.sqrt(np.mean((true - pred) ** 2)))
            pearson_vals.append(np.corrcoef(true, pred)[0, 1] if len(true) > 1 else float("nan"))
            # Paper protocol: >=7 positive, <=5 negative, exclude ambiguous 5-7 range
            pos_mask = true >= 7.0
            neg_mask = true <= 5.0
            cls_mask = pos_mask | neg_mask
            if cls_mask.sum() >= 2:
                cls_labels = pos_mask[cls_mask].astype(int)
                cls_pred = pred[cls_mask]
                if len(np.unique(cls_labels)) == 2:
                    auroc_vals.append(roc_auc_score(cls_labels, cls_pred))
                    auprc_vals.append(average_precision_score(cls_labels, cls_pred))
        ci = np.mean(ci_vals) if ci_vals else 0
        rmse = np.mean(rmse_vals) if rmse_vals else 0
        auroc = np.mean(auroc_vals) if auroc_vals else 0
        auprc = np.mean(auprc_vals) if auprc_vals else 0
        r = np.nanmean(pearson_vals) if pearson_vals else 0
        log.info(f"{mname:20s} CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} AUPRC={auprc:.4f} r={r:.4f} n_proteins={len(ci_vals)} ({label})")

    macro_eval(davis_eval, "drug-filtered, per-protein macro-avg")

log.info("=== Baseline evaluation complete ===")
