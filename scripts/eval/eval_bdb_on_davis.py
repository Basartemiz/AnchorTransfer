"""Evaluate BDB-trained CoNCISE and ConciseAnchor on Davis.

Cross-dataset: Train on BDB, eval on Davis.
Anchors: Tanimoto retrieval from BDB training set (pKi >= 7, excl canonical overlap).
"""
import os, sys, json, logging, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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

# ============================================================
# 2. Load BDB training data for anchor retrieval
# ============================================================
bdb = pd.read_csv(DATA_DIR / 'processed' / 'bindingdb_interactions.csv')
random.seed(42)
bdb_prots_all = sorted(set(bdb.uniprot_id) & set(seqs.keys()))
random.shuffle(bdb_prots_all)
nv = max(1, int(len(bdb_prots_all) * 0.1))
bdb_train_prots = set(bdb_prots_all[nv:])
bdb_train = bdb[bdb.uniprot_id.isin(bdb_train_prots)]
log.info(f'BDB train: {len(bdb_train)} interactions, {len(bdb_train_prots)} proteins')

# Canonical drug overlap exclusion
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

def canonical(smi):
    mol = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(mol) if mol else None

log.info('Computing canonical SMILES...')
davis_canonical = {smi: canonical(smi) for smi in davis.ligand_smiles.unique()}
bdb_canonical = {}
for smi in bdb_train.ligand_smiles.unique():
    c = canonical(smi)
    if c: bdb_canonical[smi] = c

davis_can_set = set(c for c in davis_canonical.values() if c)
bdb_can_set = set(bdb_canonical.values())
overlap_can = davis_can_set & bdb_can_set
log.info(f'Canonical overlap: {len(overlap_can)} drugs')

# Filter BDB training drugs to exclude those with canonical overlap with Davis
bdb_train_no_overlap = bdb_train[~bdb_train.ligand_smiles.map(lambda s: bdb_canonical.get(s) in overlap_can)]
log.info(f'BDB train after removing overlap: {len(bdb_train_no_overlap)} interactions')

# ============================================================
# 3. Tanimoto-based anchor retrieval from BDB training set
# ============================================================
log.info('Computing Morgan fingerprints for anchor retrieval...')
fp_radius = 2
fp_bits = 2048

def get_morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)

# BDB binders (pKi >= 7)
bdb_binders = bdb_train_no_overlap[bdb_train_no_overlap.pki >= 7.0]
bdb_binder_drugs = bdb_binders.ligand_smiles.unique()
log.info(f'BDB binder drugs (pKi >= 7): {len(bdb_binder_drugs)}')

# Build anchor map: for each BDB binder drug, find strongest binder protein
drug_to_anchor = {}
for smi, grp in bdb_binders.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    uid = s.iloc[0].uniprot_id
    pki = s.iloc[0].pki
    drug_to_anchor[smi] = (uid, pki)

# Compute BDB binder FPs
log.info('Computing BDB binder fingerprints...')
bdb_fps = {}
for smi in bdb_binder_drugs:
    fp = get_morgan_fp(smi)
    if fp is not None:
        bdb_fps[smi] = fp
log.info(f'BDB binder FPs: {len(bdb_fps)}')

# For each Davis drug, find most similar BDB binder drug
log.info('Finding Tanimoto anchors for Davis drugs...')
davis_drugs = davis.ligand_smiles.unique()
davis_drug_fps = {}
for smi in davis_drugs:
    fp = get_morgan_fp(smi)
    if fp is not None:
        davis_drug_fps[smi] = fp

bdb_smi_list = list(bdb_fps.keys())
bdb_fp_list = list(bdb_fps.values())

davis_drug_to_anchor = {}
for d_smi, d_fp in davis_drug_fps.items():
    # Exclude exact canonical match
    d_can = davis_canonical.get(d_smi)
    best_sim, best_smi = -1, None
    for b_smi, b_fp in zip(bdb_smi_list, bdb_fp_list):
        if bdb_canonical.get(b_smi) == d_can:
            continue
        sim = DataStructs.TanimotoSimilarity(d_fp, b_fp)
        if sim > best_sim:
            best_sim = sim
            best_smi = b_smi
    if best_smi and best_smi in drug_to_anchor:
        au, ap = drug_to_anchor[best_smi]
        davis_drug_to_anchor[d_smi] = (au, ap, best_sim)

log.info(f'Davis drugs with anchors: {len(davis_drug_to_anchor)}/{len(davis_drugs)}')

# ============================================================
# 4. Load Raygun embeddings
# ============================================================
# Load BDB Raygun embeddings (computed during training)
raygun_embs = torch.load('results/raygun_bdb_embeddings.pt', map_location='cpu', weights_only=False)
log.info(f'BDB Raygun: {len(raygun_embs)} proteins')

# Compute Davis protein embeddings
davis_prots = sorted(set(davis.uniprot_id) & set(seqs.keys()))
missing = [u for u in davis_prots if u not in raygun_embs]
log.info(f'Davis proteins: {len(davis_prots)}, missing from BDB cache: {len(missing)}')

if missing:
    ESM_CACHE = Path('results/esm2_davis_embeddings.pt')
    if ESM_CACHE.exists():
        esm_embs = torch.load(ESM_CACHE, map_location='cpu', weights_only=False)
        log.info(f'Loaded {len(esm_embs)} cached ESM-2 Davis embeddings')
    else:
        import esm
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = esm_alphabet.get_batch_converter()
        esm_model = esm_model.to(device); esm_model.eval()
        esm_embs = {}
        with torch.no_grad():
            for i, uid in enumerate(missing):
                seq = seqs[uid][:1022]
                _, _, tokens = bc([(uid, seq)])
                emb = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
                esm_embs[uid] = emb["representations"][33][:, 1:-1, :].cpu()
                if (i + 1) % 50 == 0:
                    log.info(f'  ESM-2: {i+1}/{len(missing)}')
        del esm_model; torch.cuda.empty_cache()
        os.makedirs('results', exist_ok=True)
        torch.save(esm_embs, ESM_CACHE)

    log.info('Running Raygun for Davis proteins...')
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device); raymodel.eval()
    skipped = 0
    with torch.no_grad():
        for uid, emb in esm_embs.items():
            try:
                ray_enc = raymodel.encoder(emb.to(device)).squeeze().cpu()
                if ray_enc.dim() == 2 and ray_enc.size(0) == 50:
                    raygun_embs[uid] = ray_enc
                else:
                    skipped += 1
            except:
                skipped += 1
    del raymodel; torch.cuda.empty_cache()
    log.info(f'Added {len(missing) - skipped} Davis proteins (skipped {skipped})')

# ============================================================
# 5. Compute Morgan FPs
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path('results/concise_davis_fp_2048.pkl')
if FP_CACHE.exists():
    with open(FP_CACHE, 'rb') as f:
        fp_dict = pickle.load(f)
else:
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for smi in sorted(set(davis.ligand_smiles.unique())):
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
# 6. Load models
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
    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]

# Load CoNCISE BDB
concise = ConciseRegression(nheads=32).to(device)
ckpt = torch.load('models/concise_bdb/best_model.pt', map_location=device, weights_only=False)
concise.load_state_dict(ckpt['model_state_dict']); concise.eval()
log.info(f'Loaded CoNCISE-BDB (epoch {ckpt.get("epoch", "?")})')

# Load ConciseAnchor BDB
anchor_model = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2).to(device)
ckpt = torch.load('models/concise_anchor_bdb/best_model.pt', map_location=device, weights_only=False)
anchor_model.load_state_dict(ckpt['model_state_dict']); anchor_model.eval()
log.info(f'Loaded ConciseAnchor-BDB (epoch {ckpt.get("epoch", "?")})')

# ============================================================
# 7. Build eval dataset
# ============================================================
# Filter Davis to interactions where we have all data
eval_rows = []
for _, row in davis.iterrows():
    uid, smi, pki = row['uniprot_id'], row['ligand_smiles'], row['pki']
    if uid not in raygun_embs or smi not in fp_dict:
        continue
    anc_info = davis_drug_to_anchor.get(smi)
    if anc_info is None:
        continue
    au, ap, tanimoto = anc_info
    if au not in raygun_embs:
        continue
    eval_rows.append({
        'uniprot_id': uid, 'ligand_smiles': smi, 'pki': pki,
        'anchor_uid': au, 'anchor_pki': ap, 'tanimoto': tanimoto,
    })
eval_df = pd.DataFrame(eval_rows)
log.info(f'Eval dataset: {len(eval_df)} interactions, {eval_df.uniprot_id.nunique()} proteins')

# Add quartiles
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
eval_df['anchor_q'] = pd.qcut(eval_df.anchor_pki.rank(method='first'), 4, labels=quartile_labels)

# ============================================================
# 8. Predict
# ============================================================
log.info('Predicting...')
concise_preds = []
anchor_preds = []
batch_size = 512

for start in range(0, len(eval_df), batch_size):
    batch = eval_df.iloc[start:start + batch_size]
    fps = torch.tensor(np.array([fp_dict[s] for s in batch.ligand_smiles])).to(device)
    qry_embs = torch.stack([raygun_embs[u] for u in batch.uniprot_id]).to(device)
    anc_embs = torch.stack([raygun_embs[u] for u in batch.anchor_uid]).to(device)

    with torch.no_grad():
        cp = concise(fps, qry_embs).cpu().tolist()
        ap = anchor_model(fps, anc_embs, qry_embs).cpu().tolist()
    concise_preds.extend(cp)
    anchor_preds.extend(ap)

eval_df['concise_pred'] = concise_preds
eval_df['anchor_pred'] = anchor_preds

# ============================================================
# 9. Results
# ============================================================
log.info('\n' + '='*60)
log.info('CROSS-DATASET: BDB → Davis')
log.info('='*60)

for name, col in [('CoNCISE-BDB', 'concise_pred'), ('ConciseAnchor-BDB', 'anchor_pred')]:
    t, p = eval_df.pki.values, eval_df[col].values
    log.info(f'\n--- {name} OVERALL ---')
    log.info(f'CI={ci_fn(t,p):.4f} RMSE={np.sqrt(mean_squared_error(t,p)):.4f} AUROC={auroc_safe(t,p):.4f} r={np.corrcoef(t,p)[0,1]:.4f} n={len(t)}')

    log.info(f'\n--- {name} QUARTILE ---')
    for q in quartile_labels:
        sub = eval_df[eval_df.anchor_q == q]
        if len(sub) < 5: continue
        tv, pv = sub.pki.values, sub[col].values
        log.info(f'{q:<16} n={len(sub):<6} CI={ci_fn(tv,pv):.4f} RMSE={np.sqrt(np.mean((tv-pv)**2)):.4f} AUROC={auroc_safe(tv,pv):.4f}')

    # Per-protein CI
    pp_cis = []
    for uid, grp in eval_df.groupby('uniprot_id'):
        tv, pv = grp.pki.values, grp[col].values
        if len(tv) >= 5:
            pp_cis.append(ci_fn(tv, pv))
    if pp_cis:
        log.info(f'Per-protein CI: mean={np.mean(pp_cis):.4f} median={np.median(pp_cis):.4f} n={len(pp_cis)}')

# Retrieval-only baseline
log.info('\n--- Retrieval-only baseline ---')
t, p = eval_df.pki.values, eval_df.anchor_pki.values
log.info(f'CI={ci_fn(t,p):.4f} RMSE={np.sqrt(mean_squared_error(t,p)):.4f} AUROC={auroc_safe(t,p):.4f}')

# Save
os.makedirs('results', exist_ok=True)
eval_df.to_csv('results/bdb_on_davis_predictions.csv', index=False)
log.info(f'\nSaved predictions to results/bdb_on_davis_predictions.csv')
log.info('=== DONE ===')
