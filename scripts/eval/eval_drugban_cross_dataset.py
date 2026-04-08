"""Cross-dataset evaluation of DrugBAN (DTC-trained) on Davis + BDB by protein family.

DrugBAN is a pairwise model (no anchors): drug graph + protein sequence → pKi.
"""
import torch, torch.nn as nn, numpy as np, pandas as pd, sys, json, logging, random, os, pickle
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Batch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

from rdkit import Chem
from rdkit.Chem import AllChem

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# Load data
# ============================================================
dtc = pd.read_csv('data/processed/dtc_training_interactions.csv')
bdb = pd.read_csv('data/processed/bindingdb_interactions.csv')
seqs = json.load(open('data/processed/merged_sequences.json'))
davis_raw = pd.read_csv('data/raw/davis/davis_benchmark.csv')
davis = davis_raw.rename(columns={'protein_name': 'uniprot_id', 'drug_smiles': 'ligand_smiles', 'pKd': 'pki'})
# Davis uses gene names — add sequences from its own protein_sequence column
if 'protein_sequence' in davis_raw.columns:
    davis_seqs = dict(zip(davis_raw.protein_name, davis_raw.protein_sequence))
    seqs.update(davis_seqs)
log.info(f'DTC: {len(dtc)}, Davis: {len(davis)}, BDB: {len(bdb)}, Seqs: {len(seqs)}')

# Load graph cache
GRAPH_CACHE = Path('data/processed/drugban_graph_cache.pt')
if GRAPH_CACHE.exists():
    raw = torch.load(GRAPH_CACHE, map_location='cpu')
    from torch_geometric.data import Data
    graph_cache = {smi: Data(x=raw['x'][smi], edge_index=raw['edge_index'][smi]) for smi in raw['x']}
    log.info(f'Graph cache: {len(graph_cache)}')
else:
    log.info('No graph cache found, building on the fly...')
    from anchor_transfer.model.drug_encoder import smiles_to_graph
    graph_cache = {}

# ============================================================
# Model
# ============================================================
from anchor_transfer.model.drugban import DrugBANModel

model = DrugBANModel().to(device)
ckpt = torch.load('models/drugban_dtc/best_model.pt', map_location=device)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
log.info('Loaded DrugBAN model')

CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind = np.argsort(y); y = y[ind]; f = f[ind]
    n = np.sum(np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1)))
    if n == 0: return np.nan
    z = np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T < np.tile(f, (len(f), 1)))) + 0.5 * np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T == np.tile(f, (len(f), 1))))
    return z / n

def get_graph(smi):
    if smi in graph_cache:
        return graph_cache[smi].clone()
    from anchor_transfer.model.drug_encoder import smiles_to_graph
    try:
        g = smiles_to_graph(smi)
        graph_cache[smi] = g
        return g.clone()
    except:
        return None

def predict_drugban(df, seqs, batch_size=64):
    """Predict pKi for all (protein, drug) pairs in df."""
    preds = []
    valid_idx = []
    for start in range(0, len(df), batch_size):
        b = df.iloc[start:start + batch_size]
        graphs, prots, idxs = [], [], []
        for i, (_, row) in enumerate(b.iterrows()):
            uid, smi = row['uniprot_id'], row['ligand_smiles']
            if uid not in seqs:
                continue
            g = get_graph(smi)
            if g is None:
                continue
            graphs.append(g)
            prots.append(enc_prot(seqs[uid]))
            idxs.append(start + i)
        if not graphs:
            continue
        graph_batch = Batch.from_data_list(graphs).to(device)
        prot_tensor = torch.tensor(prots, dtype=torch.long, device=device)
        with torch.no_grad():
            pred = model(graph_batch, prot_tensor)
        preds.extend(pred.cpu().tolist())
        valid_idx.extend(idxs)
    return valid_idx, preds

def per_protein_metrics(df, min_interactions=5):
    """Compute per-protein CI and RMSE."""
    rows = []
    for uid, grp in df.groupby('uniprot_id'):
        t = grp.pki.values
        p = grp.pred.values
        if len(t) < min_interactions:
            continue
        ci = ci_fn(t, p)
        rmse = np.sqrt(mean_squared_error(t, p))
        rows.append({'uniprot_id': uid, 'ci': ci, 'rmse': rmse, 'n': len(t)})
    return pd.DataFrame(rows)

# ============================================================
# 1. Davis evaluation
# ============================================================
log.info('\n=== DAVIS ===')
davis_filt = davis[davis.uniprot_id.isin(seqs)].copy()
log.info(f'Davis filtered: {len(davis_filt)} interactions, {davis_filt.uniprot_id.nunique()} proteins')

idx, preds = predict_drugban(davis_filt, seqs)
davis_eval = davis_filt.iloc[[i - davis_filt.index[0] if davis_filt.index[0] > 0 else i for i in range(len(davis_filt))]].copy()
davis_eval = davis_filt.copy()
davis_eval['pred'] = np.nan
for i, p in zip(idx, preds):
    davis_eval.iloc[i - davis_eval.index[0], davis_eval.columns.get_loc('pred')] = p
davis_eval = davis_eval.dropna(subset=['pred'])
log.info(f'Davis predictions: {len(davis_eval)}')

pp_davis = per_protein_metrics(davis_eval)
y, p = davis_eval.pki.values, davis_eval.pred.values
overall_rmse = np.sqrt(mean_squared_error(y, p))
bin_labels = (y >= 7).astype(int)
overall_auroc = roc_auc_score(bin_labels, p) if bin_labels.sum() > 0 and (1 - bin_labels).sum() > 0 else np.nan
log.info(f'Davis per-protein (n={len(pp_davis)}): CI={pp_davis.ci.mean():.4f} (med {pp_davis.ci.median():.4f}), RMSE={pp_davis.rmse.mean():.4f}, AUROC={overall_auroc:.4f}')

# ============================================================
# 2. BDB evaluation with family stratification
# ============================================================
log.info('\n=== BINDINGDB ===')

# Canonical overlap exclusion
def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True)

# DTC train split (same as training)
dtc_valid = dtc[dtc.uniprot_id.isin(seqs)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
dtc_train_prots = set(dtc_prots[nt + nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]

dtc_smiles = set(dtc_train.ligand_smiles.unique())
dtc_canon = set()
for i, s in enumerate(dtc_smiles):
    c = canonicalize(s)
    if c: dtc_canon.add(c)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing DTC: {i+1}/{len(dtc_smiles)}')
log.info(f'DTC canonical drugs: {len(dtc_canon)}')

bdb_overlap = set()
for i, s in enumerate(bdb.ligand_smiles.unique()):
    c = canonicalize(s)
    if c and c in dtc_canon:
        bdb_overlap.add(s)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing BDB: {i+1}/{len(bdb.ligand_smiles.unique())}')
log.info(f'BDB overlap: {len(bdb_overlap)}/{bdb.ligand_smiles.nunique()} ({len(bdb_overlap)/bdb.ligand_smiles.nunique()*100:.1f}%)')

bdb_clean = bdb[~bdb.ligand_smiles.isin(bdb_overlap) & bdb.uniprot_id.isin(seqs)].copy()
log.info(f'BDB after exclusion: {len(bdb_clean)} interactions, {bdb_clean.uniprot_id.nunique()} proteins')

# Predict
log.info('Predicting on BDB...')
idx, preds = predict_drugban(bdb_clean, seqs, batch_size=64)
bdb_eval = bdb_clean.copy()
bdb_eval['pred'] = np.nan
for i, p in zip(idx, preds):
    bdb_eval.iloc[i - bdb_eval.index[0], bdb_eval.columns.get_loc('pred')] = p
bdb_eval = bdb_eval.dropna(subset=['pred'])
log.info(f'BDB predictions: {len(bdb_eval)}')

# Load families
families = {}
fam_path = Path('results/bdb_protein_families.json')
if fam_path.exists():
    families = json.load(open(fam_path))
bdb_eval['family'] = bdb_eval.uniprot_id.map(lambda u: families.get(u, 'Unknown'))

def collapse_family(f):
    if 'kinase' in f.lower(): return 'Protein kinase superfamily'
    elif 'G-protein coupled receptor 1' in f: return 'GPCR family 1'
    elif 'G-protein coupled receptor' in f: return 'GPCR (other)'
    return f
bdb_eval['family_group'] = bdb_eval.family.apply(collapse_family)
fam_counts = bdb_eval.groupby('family_group')['uniprot_id'].nunique()
top_families = set(fam_counts[fam_counts >= 10].index) - {'Unknown'}
bdb_eval['family_group'] = bdb_eval.family_group.apply(lambda f: f if f in top_families else 'Other')

# Per-protein per-family metrics
pp_bdb_rows = []
for uid, grp in bdb_eval.groupby('uniprot_id'):
    t = grp.pki.values; p = grp.pred.values
    if len(t) < 5: continue
    ci = ci_fn(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    fam = grp.family_group.values[0]
    pp_bdb_rows.append({'uniprot_id': uid, 'family': fam, 'ci': ci, 'rmse': rmse, 'n': len(t)})
pp_bdb = pd.DataFrame(pp_bdb_rows)

log.info(f'\nBDB per-protein (n={len(pp_bdb)}):')
log.info(f'Overall: CI={pp_bdb.ci.mean():.4f} (med {pp_bdb.ci.median():.4f}), RMSE={pp_bdb.rmse.mean():.4f}')
log.info(f'\nBy family:')
for fam in sorted(pp_bdb.family.unique()):
    d = pp_bdb[pp_bdb.family == fam]
    log.info(f'  {fam:<40} n={len(d):<5} CI={d.ci.mean():.3f} (med {d.ci.median():.3f}) RMSE={d.rmse.mean():.3f}')

# Save results
os.makedirs('results', exist_ok=True)
pp_davis.to_csv('results/drugban_davis_per_protein.csv', index=False)
pp_bdb.to_csv('results/drugban_bdb_per_protein.csv', index=False)
davis_eval.to_csv('results/drugban_davis_predictions.csv', index=False)
log.info('\nSaved results to results/')
log.info('DONE')
