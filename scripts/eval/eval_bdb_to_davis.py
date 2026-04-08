"""Evaluate BDB-trained CoNCISE and ConciseAnchor on Davis.

Cross-dataset: models trained on BindingDB, evaluated on Davis.
Uses Tanimoto anchor retrieval from BDB training set.
Reports per-protein CI/RMSE with quartile analysis.
"""
import os, sys, json, logging, random, pickle
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
# 1. Load Davis + sequences
# ============================================================
davis_raw = pd.read_csv(DATA_DIR / 'raw' / 'davis' / 'davis_benchmark.csv')
davis = davis_raw.rename(columns={'protein_name': 'uniprot_id', 'drug_smiles': 'ligand_smiles', 'pKd': 'pki'})
seqs = {}
if 'protein_sequence' in davis_raw.columns:
    for _, r in davis_raw.drop_duplicates('protein_name').iterrows():
        seqs[r['protein_name']] = r['protein_sequence']
merged_seq = DATA_DIR / 'processed' / 'merged_sequences.json'
if merged_seq.exists():
    seqs.update(json.load(open(merged_seq)))
log.info(f'Davis: {len(davis)} interactions, {davis.uniprot_id.nunique()} proteins, {davis.ligand_smiles.nunique()} drugs')

# ============================================================
# 2. Canonical drug overlap exclusion (BDB vs Davis)
# ============================================================
from rdkit import Chem

def canonical(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return smi
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except:
        return smi

bdb = pd.read_csv(DATA_DIR / 'processed' / 'bindingdb_interactions.csv')

# Build BDB train set (same split as training)
random.seed(42)
bdb_prots_all = sorted(set(bdb.uniprot_id) & set(seqs.keys()))
# Load Raygun cache to know which proteins were used
raygun_bdb = torch.load('results/raygun_bdb_embeddings.pt', map_location='cpu', weights_only=False)
bdb_prots = sorted(set(bdb.uniprot_id) & set(raygun_bdb.keys()) & set(seqs.keys()))
random.shuffle(bdb_prots)
nv = max(1, int(len(bdb_prots) * 0.1))
bdb_train_prots = set(bdb_prots[nv:])
bdb_train = bdb[bdb.uniprot_id.isin(bdb_train_prots)]
log.info(f'BDB train: {len(bdb_train)} interactions, {bdb_train.uniprot_id.nunique()} proteins')

# Canonical overlap
bdb_train_canon = set(bdb_train.ligand_smiles.apply(canonical).unique())
davis['canon_smiles'] = davis.ligand_smiles.apply(canonical)
overlap_drugs = set(davis.canon_smiles.unique()) & bdb_train_canon
davis_filt = davis[~davis.canon_smiles.isin(overlap_drugs)].copy()
log.info(f'Drug overlap: {len(overlap_drugs)} canonical matches removed')
log.info(f'Davis after overlap filtering: {len(davis_filt)} interactions, {davis_filt.uniprot_id.nunique()} proteins')

# Also exclude proteins in BDB train
bdb_train_seqs = {seqs.get(uid, '') for uid in bdb_train_prots if uid in seqs}
overlap_prots = set()
for uid in davis_filt.uniprot_id.unique():
    if uid in bdb_train_prots or (uid in seqs and seqs[uid] in bdb_train_seqs):
        overlap_prots.add(uid)
if overlap_prots:
    davis_filt = davis_filt[~davis_filt.uniprot_id.isin(overlap_prots)]
    log.info(f'Protein overlap: {len(overlap_prots)} proteins removed. Remaining: {len(davis_filt)}')

davis_filt = davis_filt[davis_filt.uniprot_id.isin(seqs)]
log.info(f'Final Davis eval: {len(davis_filt)} interactions, {davis_filt.uniprot_id.nunique()} proteins')

# ============================================================
# 3. Tanimoto anchor retrieval from BDB training set
# ============================================================
from rdkit.Chem import AllChem
from rdkit import DataStructs

log.info('Computing Tanimoto anchor retrieval from BDB train...')

# Build BDB anchor pool: strongest binder per drug, pKi >= 7
bdb_anchor_pool = {}
for smi, grp in bdb_train.groupby('ligand_smiles'):
    best = grp.sort_values('pki', ascending=False).iloc[0]
    if best['pki'] >= 7.0 and best['uniprot_id'] in raygun_bdb:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
                bdb_anchor_pool[smi] = (best['uniprot_id'], float(best['pki']), fp)
        except:
            pass
log.info(f'BDB anchor pool: {len(bdb_anchor_pool)} drugs with pKi >= 7')

# Retrieve best anchor for each Davis drug by Tanimoto
davis_drugs = sorted(set(davis_filt.ligand_smiles.unique()))
drug_to_anchor = {}
pool_keys = list(bdb_anchor_pool.keys())
pool_fps = [bdb_anchor_pool[k][2] for k in pool_keys]

for smi in davis_drugs:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        qfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
        sims = DataStructs.BulkTanimotoSimilarity(qfp, pool_fps)
        best_i = int(np.argmax(sims))
        best_smi = pool_keys[best_i]
        au, ap, _ = bdb_anchor_pool[best_smi]
        drug_to_anchor[smi] = (au, ap, float(sims[best_i]))
    except:
        pass

log.info(f'Anchored {len(drug_to_anchor)}/{len(davis_drugs)} Davis drugs')

# Build anchored subset
rows, anc_uids, anc_pkis, anc_tanimotos = [], [], [], []
for i, row in davis_filt.iterrows():
    smi = row['ligand_smiles']
    if smi not in drug_to_anchor:
        continue
    au, ap, tan = drug_to_anchor[smi]
    rows.append(i)
    anc_uids.append(au)
    anc_pkis.append(ap)
    anc_tanimotos.append(tan)

subset = davis_filt.loc[rows].copy()
subset['anchor_uid'] = anc_uids
subset['anchor_pki'] = anc_pkis
subset['tanimoto'] = anc_tanimotos
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
subset['anchor_q'] = pd.qcut(subset.anchor_pki.rank(method='first'), 4, labels=quartile_labels)
log.info(f'Anchored subset: {len(subset)} interactions, tanimoto mean={np.mean(anc_tanimotos):.3f}')

# ============================================================
# 4. Compute Raygun embeddings for Davis proteins
# ============================================================
all_proteins = sorted(set(subset.uniprot_id.unique()) | set(subset.anchor_uid.unique()))
all_proteins = [u for u in all_proteins if u in seqs]

# Load from BDB cache first
raygun_embs = {}
for u in all_proteins:
    if u in raygun_bdb:
        raygun_embs[u] = raygun_bdb[u]

missing = [u for u in all_proteins if u not in raygun_embs]
log.info(f'Raygun: {len(raygun_embs)} from BDB cache, {len(missing)} need computing')

if missing:
    ESM_DAVIS_CACHE = Path('results/esm2_davis_embeddings.pt')
    if ESM_DAVIS_CACHE.exists():
        esm_davis = torch.load(ESM_DAVIS_CACHE, map_location='cpu', weights_only=False)
        esm_embeddings = {u: esm_davis[u] for u in missing if u in esm_davis}
        still_missing = [u for u in missing if u not in esm_davis]
    else:
        esm_embeddings = {}
        still_missing = missing

    if still_missing:
        import esm
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = esm_alphabet.get_batch_converter()
        esm_model = esm_model.to(device); esm_model.eval()
        with torch.no_grad():
            for i, uid in enumerate(still_missing):
                seq = seqs[uid][:1022]
                _, _, tokens = bc([(uid, seq)])
                emb = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
                esm_embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
                if (i + 1) % 50 == 0:
                    log.info(f'  ESM-2: {i+1}/{len(still_missing)}')
        del esm_model; torch.cuda.empty_cache()

    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device); raymodel.eval()
    with torch.no_grad():
        for uid, emb in esm_embeddings.items():
            try:
                ray_enc = raymodel.encoder(emb.to(device)).squeeze().cpu()
                if ray_enc.dim() == 2 and ray_enc.size(0) == 50:
                    raygun_embs[uid] = ray_enc
            except:
                pass
    del raymodel; torch.cuda.empty_cache()
    log.info(f'Computed {len(raygun_embs)} total Raygun embeddings')

# ============================================================
# 5. Morgan FPs for Davis
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path('results/concise_davis_fp.pkl')
if FP_CACHE.exists():
    with open(FP_CACHE, 'rb') as f:
        fp_dict = pickle.load(f)
else:
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for smi in sorted(set(subset.ligand_smiles.unique())):
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
# 6. Load models and predict
# ============================================================
from concise.model.concise import Concise
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

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

# --- CoNCISE (BDB-trained) ---
concise = ConciseRegression(nheads=32).to(device)
ckpt = torch.load('models/concise_bdb/best_model.pt', map_location=device, weights_only=False)
concise.load_state_dict(ckpt['model_state_dict']); concise.eval()
log.info(f'Loaded CoNCISE-BDB (epoch {ckpt.get("epoch", "?")})')

# --- ConciseAnchor (BDB-trained) ---
anchor_model = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2).to(device)
ckpt2 = torch.load('models/concise_anchor_bdb/best_model.pt', map_location=device, weights_only=False)
anchor_model.load_state_dict(ckpt2['model_state_dict']); anchor_model.eval()
log.info(f'Loaded ConciseAnchor-BDB (epoch {ckpt2.get("epoch", "?")})')

# Predict
log.info('Predicting...')
concise_preds, anchor_preds = [], []
valid_mask = []
for _, row in subset.iterrows():
    uid, smi, au = row['uniprot_id'], row['ligand_smiles'], row['anchor_uid']
    if uid not in raygun_embs or smi not in fp_dict or au not in raygun_embs:
        concise_preds.append(np.nan); anchor_preds.append(np.nan)
        valid_mask.append(False)
        continue
    fp = torch.tensor(fp_dict[smi]).unsqueeze(0).to(device)
    qry = raygun_embs[uid].unsqueeze(0).to(device)
    anc = raygun_embs[au].unsqueeze(0).to(device)
    with torch.no_grad():
        cp = concise(fp, qry).item()
        ap = anchor_model(fp, anc, qry).item()
    concise_preds.append(cp); anchor_preds.append(ap)
    valid_mask.append(True)

subset['concise_pred'] = concise_preds
subset['anchor_pred'] = anchor_preds
subset_valid = subset[valid_mask].copy()
log.info(f'Valid predictions: {len(subset_valid)}')

# ============================================================
# 7. Results
# ============================================================
log.info(f'\n{"="*60}')
log.info(f'BDB→Davis Cross-Dataset Evaluation')
log.info(f'{"="*60}')

for name, col in [('CoNCISE-BDB', 'concise_pred'), ('ConciseAnchor-BDB', 'anchor_pred')]:
    t, p = subset_valid.pki.values, subset_valid[col].values
    ci = ci_fn(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    auroc = auroc_safe(t, p)
    r = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0
    log.info(f'\n{name}: CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(t)}')

    # Quartile analysis
    for q in quartile_labels:
        sub = subset_valid[subset_valid.anchor_q == q]
        if len(sub) < 5: continue
        tv, pv = sub.pki.values, sub[col].values
        log.info(f'  {q:<16} n={len(sub):<6} CI={ci_fn(tv,pv):.4f} AUROC={auroc_safe(tv,pv):.4f} RMSE={np.sqrt(np.mean((tv-pv)**2)):.4f}')

    # Per-protein CI
    pp_cis = []
    for uid, grp in subset_valid.groupby('uniprot_id'):
        tv, pv = grp.pki.values, grp[col].values
        if len(tv) >= 5:
            pp_cis.append(ci_fn(tv, pv))
    if pp_cis:
        log.info(f'  Per-protein CI: mean={np.mean(pp_cis):.4f} med={np.median(pp_cis):.4f} n={len(pp_cis)}')

# Retrieval-only baseline
t = subset_valid.pki.values
ret_p = subset_valid.anchor_pki.values
log.info(f'\nRetrieval-only: CI={ci_fn(t, ret_p):.4f} AUROC={auroc_safe(t, ret_p):.4f} RMSE={np.sqrt(np.mean((t-ret_p)**2)):.4f}')

# Save full results
os.makedirs('results', exist_ok=True)
subset_valid.to_csv('results/bdb_to_davis_predictions.csv', index=False)
log.info(f'\nSaved results to results/bdb_to_davis_predictions.csv')
log.info('=== DONE ===')
