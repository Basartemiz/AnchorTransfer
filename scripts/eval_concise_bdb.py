"""Evaluate vanilla CoNCISE on BDB using same anchored subset.

Uses cached Raygun BDB embeddings + Morgan FPs.
CoNCISE is pairwise (no anchors) but evaluated on same rows for fair comparison.
"""
import os, sys, json, logging, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ci_fn(y, f):
    n = len(y)
    if n < 2: return 0.5
    y, f = np.array(y), np.array(f)
    if n * (n - 1) // 2 > 100000:
        i = np.random.randint(0, n, 100000); j = np.random.randint(0, n, 100000)
        m = i != j; i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1); i, j = idx[0], idx[1]
    dt = y[i] - y[j]; dp = f[i] - f[j]; t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

# Load anchored subset
sub = pd.read_csv('results/bdb_new_models_anchor_metadata.csv')
log.info(f'Loaded: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins')

# Load Raygun embeddings (cached from ConciseAnchor BDB eval)
raygun_embs = torch.load('results/raygun_bdb.pt', map_location='cpu', weights_only=False)
log.info(f'Raygun: {len(raygun_embs)} proteins')

# Load Morgan FPs
with open('results/concise_bdb_fp.pkl', 'rb') as f:
    fp_dict = pickle.load(f)
log.info(f'Morgan FPs: {len(fp_dict)}')

# Load CoNCISE model
from concise.model.concise import Concise

class ConciseRegression(nn.Module):
    def __init__(self, nheads=32):
        super().__init__()
        drug_layers = [[32], [32], [32]]
        proj_dim = 256
        self.backbone = Concise(
            drug_layers=drug_layers, ligand_dim=2048, residue_dim=1280,
            drug_dim=proj_dim, proj_dim=proj_dim, nheads=nheads,
            activation="gelu", cosine_prediction=False,
        )
        fused_dim = len(drug_layers) * proj_dim + proj_dim
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.backbone.final[-1].bias, 6.5)
    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]

model = ConciseRegression(nheads=32).to(device)
ckpt = torch.load('models/concise_dtc/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
log.info(f'Loaded CoNCISE (epoch {ckpt.get("epoch", "?")})')

# Predict
log.info('Predicting CoNCISE on BDB...')
valid_idx = []
for i, (_, row) in enumerate(sub.iterrows()):
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if uid in raygun_embs and smi in fp_dict:
        valid_idx.append(i)
log.info(f'Valid: {len(valid_idx)}/{len(sub)}')

pred_list = [None] * len(sub)
batch_size = 512
for start in range(0, len(valid_idx), batch_size):
    batch_idx = valid_idx[start:start + batch_size]
    fps = torch.tensor(np.array([fp_dict[sub.iloc[j].ligand_smiles] for j in batch_idx])).to(device)
    embs = torch.stack([raygun_embs[sub.iloc[j].uniprot_id] for j in batch_idx]).to(device)
    with torch.no_grad():
        preds = model(fps, embs).cpu().tolist()
    for k, idx in enumerate(batch_idx):
        pred_list[idx] = preds[k]
    if (start + batch_size) % 50000 < batch_size:
        log.info(f'  Predicted: {min(start+batch_size, len(valid_idx))}/{len(valid_idx)}')

sub['concise_pred'] = pred_list
sub_valid = sub.dropna(subset=['concise_pred']).copy()
log.info(f'Predictions: {len(sub_valid)}')

# Per-protein metrics by family
MIN_INTERACTIONS = 5
pp_rows = []
for uid, grp in sub_valid.groupby('uniprot_id'):
    t = grp.pki.values
    if len(t) < MIN_INTERACTIONS: continue
    fam = grp.family_group.values[0]
    p = grp.concise_pred.values
    pp_rows.append({
        'uniprot_id': uid, 'family': fam, 'model': 'CoNCISE',
        'ci': ci_fn(t, p), 'rmse': float(np.sqrt(mean_squared_error(t, p))), 'n': len(t),
    })
pp = pd.DataFrame(pp_rows)

ci_vals = pp.ci.dropna()
log.info(f'\n=== CoNCISE BDB OVERALL ===')
log.info(f'n={len(pp)} proteins  CI: mean={ci_vals.mean():.4f} med={ci_vals.median():.4f}  RMSE: mean={pp.rmse.mean():.4f}')

log.info(f'\n=== BY FAMILY ===')
for fam in sorted(pp.family.unique()):
    d = pp[pp.family == fam]
    if len(d) < 3: continue
    cv = d.ci.dropna()
    log.info(f'{fam:<40} n={len(d):<5} CI={cv.mean():.3f} (med {cv.median():.3f}) RMSE={d.rmse.mean():.3f}')

quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
log.info(f'\n=== ANCHOR QUARTILE ===')
for q in quartile_labels:
    qsub = sub_valid[sub_valid.anchor_q == q]
    if len(qsub) < 10: continue
    t, p = qsub.pki.values, qsub.concise_pred.values
    log.info(f'{q:<16} n={len(qsub):<7} CI={ci_fn(t,p):.4f} RMSE={np.sqrt(np.mean((t-p)**2)):.4f}')

pp.to_csv('results/bdb_concise_per_protein.csv', index=False)
log.info('\nSaved to results/bdb_concise_per_protein.csv')
log.info('=== DONE ===')
