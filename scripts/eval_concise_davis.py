"""Evaluate CoNCISE on Davis with per-protein CI/RMSE.

CoNCISE needs Raygun embeddings (ESM-2 650M -> Raygun) + Morgan FP.
Computes Raygun embeddings for Davis proteins if not cached, then predicts.
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {device}')

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
# 1. Load Davis
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

# DTC overlap filtering
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
        top = s.iloc[0]
        strongest[smi] = (top['uniprot_id'], float(top['pki']))
        if len(s) > 1:
            snd = s.iloc[1]
            second[smi] = (snd['uniprot_id'], float(snd['pki']))
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
log.info(f'Anchored subset: {len(subset)} interactions')

quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
subset['anchor_q'] = pd.qcut(subset.anchor_pki.rank(method='first'), 4, labels=quartile_labels)

# ============================================================
# 2. Compute Raygun embeddings for Davis proteins
# ============================================================
RAYGUN_CACHE = Path('results/raygun_davis.pt')
davis_proteins = sorted(set(subset.uniprot_id.unique()) | set(subset.anchor_uid.unique()))
davis_proteins = [u for u in davis_proteins if u in seqs]

if RAYGUN_CACHE.exists():
    log.info(f'Loading cached Raygun-Davis embeddings')
    raygun_embs = torch.load(RAYGUN_CACHE, map_location='cpu', weights_only=False)
    missing = [u for u in davis_proteins if u not in raygun_embs]
    if missing:
        log.info(f'{len(missing)} proteins missing from cache, computing...')
    else:
        missing = []
else:
    raygun_embs = {}
    # Also try loading DTC raygun cache
    dtc_cache = Path('results/raygun_embeddings.pt')
    if dtc_cache.exists():
        dtc_raygun = torch.load(dtc_cache, map_location='cpu', weights_only=False)
        for u in davis_proteins:
            if u in dtc_raygun:
                raygun_embs[u] = dtc_raygun[u]
        del dtc_raygun
        log.info(f'Loaded {len(raygun_embs)} from DTC Raygun cache')
    missing = [u for u in davis_proteins if u not in raygun_embs]
    log.info(f'{len(missing)} Davis proteins need Raygun embeddings')

if missing:
    import esm
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device); esm_model.eval()

    esm_embeddings = {}
    with torch.no_grad():
        for i, uid in enumerate(missing):
            seq = seqs[uid][:1022]
            _, _, tokens = bc([(uid, seq)])
            emb = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
            esm_embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
            if (i + 1) % 50 == 0:
                log.info(f'  ESM-2: {i+1}/{len(missing)}')
    del esm_model; torch.cuda.empty_cache()

    log.info('Running Raygun encoder...')
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device); raymodel.eval()
    with torch.no_grad():
        for i, (uid, emb) in enumerate(esm_embeddings.items()):
            ray_enc = raymodel.encoder(emb.to(device)).squeeze().cpu()
            raygun_embs[uid] = ray_enc
            if (i + 1) % 50 == 0:
                log.info(f'  Raygun: {i+1}/{len(esm_embeddings)}')
    del raymodel, esm_embeddings; torch.cuda.empty_cache()

    os.makedirs('results', exist_ok=True)
    torch.save(raygun_embs, RAYGUN_CACHE)
    log.info(f'Saved {len(raygun_embs)} Raygun embeddings')

# ============================================================
# 3. Compute Morgan FPs for Davis drugs
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path('results/concise_davis_fp.pkl')
if FP_CACHE.exists():
    with open(FP_CACHE, 'rb') as f:
        fp_dict = pickle.load(f)
else:
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    all_smiles = sorted(set(subset.ligand_smiles.unique()))
    for i, smi in enumerate(all_smiles):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except: pass
    os.makedirs('results', exist_ok=True)
    with open(FP_CACHE, 'wb') as f:
        pickle.dump(fp_dict, f)
log.info(f'Morgan FPs: {len(fp_dict)} drugs')

# ============================================================
# 4. Load CoNCISE model
# ============================================================
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

# ============================================================
# 5. Predict CoNCISE
# ============================================================
log.info('Predicting CoNCISE...')
concise_preds = []
valid_idx = []
for i, (_, row) in enumerate(subset.iterrows()):
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if uid not in raygun_embs or smi not in fp_dict:
        continue
    valid_idx.append(i)

# Batch predict
batch_size = 256
pred_list = [None] * len(subset)
for start in range(0, len(valid_idx), batch_size):
    batch_idx = valid_idx[start:start + batch_size]
    fps = torch.tensor(np.array([fp_dict[subset.iloc[i].ligand_smiles] for i in batch_idx])).to(device)
    embs = torch.stack([raygun_embs[subset.iloc[i].uniprot_id] for i in batch_idx]).to(device)
    with torch.no_grad():
        preds = model(fps, embs).cpu().tolist()
    for j, idx in enumerate(batch_idx):
        pred_list[idx] = preds[j]

subset['concise_pred'] = pred_list
subset_valid = subset.dropna(subset=['concise_pred']).copy()
log.info(f'CoNCISE predictions: {len(subset_valid)}')

# ============================================================
# 6. Results
# ============================================================
t, p = subset_valid.pki.values, subset_valid.concise_pred.values
ci = ci_fn(t, p)
rmse = np.sqrt(mean_squared_error(t, p))
auroc = auroc_safe(t, p)
r = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0
log.info(f'\n=== CoNCISE DAVIS OVERALL ===')
log.info(f'CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(t)}')

# Per-protein metrics
pp_rows = []
for q in quartile_labels:
    sub = subset_valid[subset_valid.anchor_q == q]
    for uid, grp in sub.groupby('uniprot_id'):
        tv = grp.pki.values; pv = grp.concise_pred.values
        if len(tv) < 5: continue
        pp_rows.append({
            'quartile': q, 'uniprot_id': uid, 'model': 'CoNCISE',
            'n': len(tv), 'ci': ci_fn(tv, pv),
            'rmse': float(np.sqrt(np.mean((tv - pv) ** 2))),
        })
pp_df = pd.DataFrame(pp_rows)
log.info(f'Per-protein: {len(pp_df)} rows, {pp_df.uniprot_id.nunique()} proteins')
ci_vals = pp_df.ci.dropna()
log.info(f'CoNCISE per-protein: CI mean={ci_vals.mean():.4f} med={ci_vals.median():.4f} RMSE mean={pp_df.rmse.mean():.4f}')

# Quartile summary
log.info(f'\n=== QUARTILE ANALYSIS ===')
for q in quartile_labels:
    sub = subset_valid[subset_valid.anchor_q == q]
    if len(sub) < 5: continue
    t, p = sub.pki.values, sub.concise_pred.values
    log.info(f'{q:<16} n={len(sub):<6} CI={ci_fn(t,p):.4f} RMSE={np.sqrt(np.mean((t-p)**2)):.4f}')

# Save
os.makedirs('results', exist_ok=True)
pp_df.to_csv('results/davis_concise_per_protein.csv', index=False)
subset_valid.to_csv('results/davis_concise_predictions.csv', index=False)
log.info('Saved results to results/')
log.info('=== DONE ===')
