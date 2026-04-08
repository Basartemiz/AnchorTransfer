"""Evaluate ConciseAnchor on Davis with per-protein CI/RMSE + anchor quartiles.

Uses cached Raygun embeddings + Morgan FPs from CoNCISE eval.
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

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

def auroc_safe(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))

# ============================================================
# 1. Load Davis + overlap filtering (same as other eval scripts)
# ============================================================
davis_raw = pd.read_csv(DATA_DIR / 'raw' / 'davis' / 'davis_benchmark.csv')
davis = davis_raw.rename(columns={'protein_name': 'uniprot_id', 'drug_smiles': 'ligand_smiles', 'pKd': 'pki'})
seqs = {}
if 'protein_sequence' in davis_raw.columns:
    for _, r in davis_raw.drop_duplicates('protein_name').iterrows():
        seqs[r['protein_name']] = r['protein_sequence']
merged_seq_path = DATA_DIR / 'processed' / 'merged_sequences.json'
if merged_seq_path.exists():
    seqs.update(json.load(open(merged_seq_path)))
log.info(f'Davis: {len(davis)}, Seqs: {len(seqs)}')

dtc = pd.read_csv(DATA_DIR / 'processed' / 'dtc_training_interactions.csv')
dtc_valid = dtc[dtc.uniprot_id.isin(seqs)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
dtc_train_prots = set(dtc_prots[nt + nv:])
train_seqs = {seqs[uid] for uid in dtc_train_prots if uid in seqs}
train_drugs = set(dtc[dtc.uniprot_id.isin(dtc_train_prots)].ligand_smiles.unique())

davis_filt = davis.copy()
overlap_prots = set()
for uid in davis_filt.uniprot_id.unique():
    if uid in dtc_train_prots or (uid in seqs and seqs[uid] in train_seqs):
        overlap_prots.add(uid)
davis_filt = davis_filt[~davis_filt.uniprot_id.isin(overlap_prots)]
davis_filt = davis_filt[~davis_filt.ligand_smiles.isin(train_drugs)]
davis_filt = davis_filt[davis_filt.uniprot_id.isin(seqs)]
log.info(f'Davis after overlap filtering: {len(davis_filt)}')

# Dataset-internal anchors
def build_anchor_maps(df):
    strongest, second = {}, {}
    for smi, grp in df.groupby('ligand_smiles'):
        s = grp.sort_values('pki', ascending=False)
        strongest[smi] = (s.iloc[0]['uniprot_id'], float(s.iloc[0]['pki']))
        if len(s) > 1:
            second[smi] = (s.iloc[1]['uniprot_id'], float(s.iloc[1]['pki']))
    return strongest, second

strongest, second = build_anchor_maps(davis_filt)
rows, anchor_uids, anchor_pkis = [], [], []
for i, row in davis_filt.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if smi not in strongest: continue
    au, ap = strongest[smi]
    if au == uid:
        if smi not in second: continue
        au, ap = second[smi]
    if au not in seqs: continue
    rows.append(i); anchor_uids.append(au); anchor_pkis.append(ap)

subset = davis_filt.loc[rows].copy()
subset['anchor_uid'] = anchor_uids
subset['anchor_pki'] = anchor_pkis
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
subset['anchor_q'] = pd.qcut(subset.anchor_pki.rank(method='first'), 4, labels=quartile_labels)
log.info(f'Anchored subset: {len(subset)} interactions')

# ============================================================
# 2. Load Raygun embeddings + Morgan FPs (cached from CoNCISE eval)
# ============================================================
raygun_embs = torch.load('results/raygun_davis.pt', map_location='cpu', weights_only=False)
log.info(f'Raygun embeddings: {len(raygun_embs)}')

FP_CACHE = Path('results/concise_davis_fp.pkl')
with open(FP_CACHE, 'rb') as f:
    fp_dict = pickle.load(f)
log.info(f'Morgan FPs: {len(fp_dict)}')

# ============================================================
# 3. Load ConciseAnchor model
# ============================================================
from anchor_transfer.model.concise_anchor import ConciseAnchor

model = ConciseAnchor(
    drug_layers=[[32], [32], [32]],
    residue_dim=1280, proj_dim=256, nheads=32, dropout=0.2,
).to(device)
ckpt = torch.load('models/concise_anchor_dtc/best_model.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model_state_dict']); model.eval()
log.info(f'Loaded ConciseAnchor (epoch {ckpt.get("epoch", "?")})')

# ============================================================
# 4. Predict
# ============================================================
log.info('Predicting ConciseAnchor...')
pred_list = [None] * len(subset)
valid_idx = []
for i, (_, row) in enumerate(subset.iterrows()):
    uid, smi, au = row['uniprot_id'], row['ligand_smiles'], row['anchor_uid']
    if uid in raygun_embs and au in raygun_embs and smi in fp_dict:
        valid_idx.append(i)

batch_size = 256
for start in range(0, len(valid_idx), batch_size):
    batch_idx = valid_idx[start:start + batch_size]
    fps = torch.tensor(np.array([fp_dict[subset.iloc[j].ligand_smiles] for j in batch_idx])).to(device)
    anc_embs = torch.stack([raygun_embs[subset.iloc[j].anchor_uid] for j in batch_idx]).to(device)
    qry_embs = torch.stack([raygun_embs[subset.iloc[j].uniprot_id] for j in batch_idx]).to(device)
    with torch.no_grad():
        preds = model(fps, anc_embs, qry_embs).cpu().tolist()
    for k, idx in enumerate(batch_idx):
        pred_list[idx] = preds[k]

subset['concise_anchor_pred'] = pred_list
subset_valid = subset.dropna(subset=['concise_anchor_pred']).copy()
log.info(f'Predictions: {len(subset_valid)}')

# ============================================================
# 5. Results
# ============================================================
t, p = subset_valid.pki.values, subset_valid.concise_anchor_pred.values
ci = ci_fn(t, p); rmse = np.sqrt(mean_squared_error(t, p))
auroc = auroc_safe(t, p); r = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0
log.info(f'\n=== ConciseAnchor DAVIS OVERALL ===')
log.info(f'CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(t)}')

# Per-protein metrics
pp_rows = []
for q in quartile_labels:
    sub = subset_valid[subset_valid.anchor_q == q]
    for uid, grp in sub.groupby('uniprot_id'):
        tv = grp.pki.values; pv = grp.concise_anchor_pred.values
        if len(tv) < 5: continue
        pp_rows.append({
            'quartile': q, 'uniprot_id': uid, 'model': 'ConciseAnchor',
            'n': len(tv), 'ci': ci_fn(tv, pv),
            'rmse': float(np.sqrt(np.mean((tv - pv) ** 2))),
        })
pp_df = pd.DataFrame(pp_rows)
ci_vals = pp_df.ci.dropna()
log.info(f'Per-protein: {len(pp_df)} rows, {pp_df.uniprot_id.nunique()} proteins')
log.info(f'ConciseAnchor per-protein: CI mean={ci_vals.mean():.4f} med={ci_vals.median():.4f} RMSE mean={pp_df.rmse.mean():.4f}')

# Quartile summary
log.info(f'\n=== QUARTILE ANALYSIS ===')
for q in quartile_labels:
    sub = subset_valid[subset_valid.anchor_q == q]
    if len(sub) < 5: continue
    t, p = sub.pki.values, sub.concise_anchor_pred.values
    log.info(f'{q:<16} n={len(sub):<6} CI={ci_fn(t,p):.4f} RMSE={np.sqrt(np.mean((t-p)**2)):.4f}')

# Save
os.makedirs('results', exist_ok=True)
pp_df.to_csv('results/davis_concise_anchor_per_protein.csv', index=False)
subset_valid.to_csv('results/davis_concise_anchor_predictions.csv', index=False)
log.info('Saved results to results/')
log.info('=== DONE ===')
