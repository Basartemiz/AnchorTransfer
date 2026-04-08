"""Evaluate ConciseAnchor on BDB using pre-computed anchor metadata.

Expects bdb_new_models_anchor_metadata.csv (from eval_new_models_bdb_family.py)
which contains the anchored subset with Tanimoto-matched DTC anchors.
Computes Raygun embeddings for BDB proteins, then predicts with ConciseAnchor.
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

# ============================================================
# 1. Load anchor metadata (from DrugBAN/AnchorDrugBAN BDB eval)
# ============================================================
META_PATH = Path('results/bdb_new_models_anchor_metadata.csv')
if not META_PATH.exists():
    log.error(f'{META_PATH} not found. Run eval_new_models_bdb_family.py first.')
    sys.exit(1)

sub = pd.read_csv(META_PATH)
log.info(f'Loaded anchor metadata: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins')

# Load sequences for Raygun computation
seqs = json.load(open(DATA_DIR / 'processed' / 'merged_sequences.json'))
log.info(f'Sequences: {len(seqs)}')

# Also load BDB to get pki values (metadata may not have them)
if 'pki' not in sub.columns:
    bdb = pd.read_csv(DATA_DIR / 'processed' / 'bindingdb_interactions.csv')
    sub = sub.merge(bdb[['uniprot_id', 'ligand_smiles', 'pki']].drop_duplicates(),
                    on=['uniprot_id', 'ligand_smiles'], how='left')
    log.info(f'Merged pki values: {sub.pki.notna().sum()}/{len(sub)}')

# ============================================================
# 2. Compute Raygun embeddings for all needed proteins
# ============================================================
all_proteins = sorted(set(sub.uniprot_id.unique()) | set(sub.anc_uid.unique()))
all_proteins = [u for u in all_proteins if u in seqs]
log.info(f'Need Raygun for {len(all_proteins)} proteins')

# Load existing caches
RAYGUN_BDB_CACHE = Path('results/raygun_bdb.pt')
raygun_embs = {}

# Try DTC cache first
dtc_cache = Path('results/raygun_embeddings.pt')
if dtc_cache.exists():
    dtc_raygun = torch.load(dtc_cache, map_location='cpu', weights_only=False)
    for u in all_proteins:
        if u in dtc_raygun:
            raygun_embs[u] = dtc_raygun[u]
    del dtc_raygun
    log.info(f'From DTC Raygun cache: {len(raygun_embs)} proteins')

# Try Davis cache
davis_cache = Path('results/raygun_davis.pt')
if davis_cache.exists():
    davis_raygun = torch.load(davis_cache, map_location='cpu', weights_only=False)
    for u in all_proteins:
        if u not in raygun_embs and u in davis_raygun:
            raygun_embs[u] = davis_raygun[u]
    del davis_raygun

# Try BDB cache
if RAYGUN_BDB_CACHE.exists():
    bdb_raygun = torch.load(RAYGUN_BDB_CACHE, map_location='cpu', weights_only=False)
    for u in all_proteins:
        if u not in raygun_embs and u in bdb_raygun:
            raygun_embs[u] = bdb_raygun[u]
    del bdb_raygun

missing = [u for u in all_proteins if u not in raygun_embs]
log.info(f'Have {len(raygun_embs)}, missing {len(missing)} proteins')

if missing:
    import esm
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device); esm_model.eval()

    esm_embeddings = {}
    with torch.no_grad():
        for i, uid in enumerate(missing):
            seq = seqs[uid][:1022]
            if len(seq) < 10:
                log.warning(f'  Skipping {uid}: sequence too short ({len(seq)})')
                continue
            _, _, tokens = bc([(uid, seq)])
            emb = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
            esm_embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
            if (i + 1) % 100 == 0:
                log.info(f'  ESM-2: {i+1}/{len(missing)}')
    del esm_model; torch.cuda.empty_cache()

    log.info('Running Raygun encoder...')
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device); raymodel.eval()
    with torch.no_grad():
        for i, (uid, emb) in enumerate(esm_embeddings.items()):
            try:
                ray_enc = raymodel.encoder(emb.to(device)).squeeze().cpu()
                raygun_embs[uid] = ray_enc
            except Exception as e:
                log.warning(f'  Raygun failed for {uid}: {e}')
            if (i + 1) % 100 == 0:
                log.info(f'  Raygun: {i+1}/{len(esm_embeddings)}')
    del raymodel, esm_embeddings; torch.cuda.empty_cache()

    os.makedirs('results', exist_ok=True)
    torch.save(raygun_embs, RAYGUN_BDB_CACHE)
    log.info(f'Saved {len(raygun_embs)} Raygun embeddings to {RAYGUN_BDB_CACHE}')

# ============================================================
# 3. Compute Morgan FPs for BDB drugs
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_BDB_CACHE = Path('results/concise_bdb_fp.pkl')
if FP_BDB_CACHE.exists():
    with open(FP_BDB_CACHE, 'rb') as f:
        fp_dict = pickle.load(f)
    log.info(f'Loaded FP cache: {len(fp_dict)}')
else:
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    all_smiles = sorted(set(sub.ligand_smiles.unique()))
    for i, smi in enumerate(all_smiles):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except: pass
        if (i + 1) % 10000 == 0:
            log.info(f'  FP: {i+1}/{len(all_smiles)}')
    os.makedirs('results', exist_ok=True)
    with open(FP_BDB_CACHE, 'wb') as f:
        pickle.dump(fp_dict, f)
    log.info(f'Computed {len(fp_dict)} FPs')

# ============================================================
# 4. Load ConciseAnchor
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
# 5. Predict
# ============================================================
log.info('Predicting ConciseAnchor on BDB...')
valid_idx = []
for i, (_, row) in enumerate(sub.iterrows()):
    uid, smi, au = row['uniprot_id'], row['ligand_smiles'], row['anc_uid']
    if uid in raygun_embs and au in raygun_embs and smi in fp_dict:
        valid_idx.append(i)
log.info(f'Valid interactions: {len(valid_idx)}/{len(sub)}')

pred_list = [None] * len(sub)
batch_size = 256
for start in range(0, len(valid_idx), batch_size):
    batch_idx = valid_idx[start:start + batch_size]
    fps = torch.tensor(np.array([fp_dict[sub.iloc[j].ligand_smiles] for j in batch_idx])).to(device)
    anc = torch.stack([raygun_embs[sub.iloc[j].anc_uid] for j in batch_idx]).to(device)
    qry = torch.stack([raygun_embs[sub.iloc[j].uniprot_id] for j in batch_idx]).to(device)
    with torch.no_grad():
        preds = model(fps, anc, qry).cpu().tolist()
    for k, idx in enumerate(batch_idx):
        pred_list[idx] = preds[k]
    if (start + batch_size) % 10000 < batch_size:
        log.info(f'  Predicted: {min(start + batch_size, len(valid_idx))}/{len(valid_idx)}')

sub['concise_anchor_pred'] = pred_list
sub_valid = sub.dropna(subset=['concise_anchor_pred']).copy()
log.info(f'Predictions: {len(sub_valid)}')

# ============================================================
# 6. Per-protein metrics by family
# ============================================================
MIN_INTERACTIONS = 5
pp_rows = []
for uid, grp in sub_valid.groupby('uniprot_id'):
    t = grp.pki.values
    if len(t) < MIN_INTERACTIONS: continue
    fam = grp.family_group.values[0]
    p = grp.concise_anchor_pred.values
    pp_rows.append({
        'uniprot_id': uid, 'family': fam, 'model': 'ConciseAnchor',
        'ci': ci_fn(t, p),
        'rmse': float(np.sqrt(mean_squared_error(t, p))),
        'n': len(t),
    })
pp = pd.DataFrame(pp_rows)
pp.to_csv('results/bdb_concise_anchor_per_protein.csv', index=False)

# Overall
ci_vals = pp.ci.dropna()
log.info(f'\n=== ConciseAnchor BDB OVERALL ===')
log.info(f'n={len(pp)} proteins  CI: mean={ci_vals.mean():.4f} med={ci_vals.median():.4f}  RMSE: mean={pp.rmse.mean():.4f}')

# By family
log.info(f'\n=== BY FAMILY ===')
for fam in sorted(pp.family.unique()):
    d = pp[pp.family == fam]
    ci_v = d.ci.dropna()
    if len(d) < 3: continue
    log.info(f'{fam:<40} n={len(d):<5} CI={ci_v.mean():.3f} (med {ci_v.median():.3f}) RMSE={d.rmse.mean():.3f}')

# By quartile
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
log.info(f'\n=== ANCHOR QUARTILE ===')
for q in quartile_labels:
    qsub = sub_valid[sub_valid.anchor_q == q]
    if len(qsub) < 10: continue
    t, p = qsub.pki.values, qsub.concise_anchor_pred.values
    log.info(f'{q:<16} n={len(qsub):<7} CI={ci_fn(t,p):.4f} RMSE={np.sqrt(np.mean((t-p)**2)):.4f}')

sub_valid.to_csv('results/bdb_concise_anchor_predictions.csv', index=False)
log.info('\nSaved results to results/')
log.info('=== DONE ===')
