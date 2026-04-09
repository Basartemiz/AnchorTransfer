"""Evaluate DrugBAN + AnchorDrugBAN on BindingDB with protein family stratification.

Follows eval_bdb_family.py structure:
  1. Canonical drug overlap exclusion (DTC train vs BDB)
  2. Tanimoto-based anchor retrieval from DTC training set
  3. Predict with DrugBAN (pairwise) and AnchorDrugBAN (anchor-based)
  4. Per-protein CI/RMSE grouped by family
  5. Violin plots
"""
import os, sys, json, logging, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'Device: {device}')

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

# ============================================================
# Helpers
# ============================================================
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,
               "M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,
               "T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_prot(s, ml=1000):
    return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

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

def canonicalize(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return Chem.MolToSmiles(mol, canonical=True)

def smi_to_fp(smi, chiral=True):
    mol = Chem.MolFromSmiles(smi)
    if mol is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=chiral)

# ============================================================
# 1. Load data
# ============================================================
log.info("Loading data...")
bdb = pd.read_csv(DATA_DIR / 'processed' / 'bindingdb_interactions.csv')
dtc = pd.read_csv(DATA_DIR / 'processed' / 'dtc_training_interactions.csv')
seqs = json.load(open(DATA_DIR / 'processed' / 'merged_sequences.json'))
log.info(f'BDB: {len(bdb)}, DTC: {len(dtc)}, Seqs: {len(seqs)}')

# Load graph cache
GRAPH_CACHE = DATA_DIR / 'processed' / 'drugban_graph_cache.pt'
if GRAPH_CACHE.exists():
    raw = torch.load(GRAPH_CACHE, map_location='cpu', weights_only=False)
    graph_cache = {smi: Data(x=raw['x'][smi], edge_index=raw['edge_index'][smi]) for smi in raw['x']}
    del raw
    log.info(f'Graph cache: {len(graph_cache)} drugs')
else:
    graph_cache = {}
    log.warning('No graph cache — building on the fly')

def get_graph(smi):
    if smi in graph_cache:
        return graph_cache[smi].clone()
    try:
        from anchor_transfer.model.drug_encoder import smiles_to_graph
        g = smiles_to_graph(smi)
        graph_cache[smi] = g
        return g.clone()
    except:
        return None

# ============================================================
# 2. DTC train split
# ============================================================
dtc_valid = dtc[dtc.uniprot_id.isin(seqs)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
dtc_train_prots = set(dtc_prots[nt + nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
log.info(f'DTC train: {len(dtc_train)} interactions, {dtc_train.uniprot_id.nunique()} proteins')

# ============================================================
# 3. Canonical drug overlap exclusion
# ============================================================
log.info("Canonical drug overlap exclusion...")
dtc_smiles = set(dtc_train.ligand_smiles.unique())
dtc_canon = set()
for i, s in enumerate(dtc_smiles):
    c = canonicalize(s)
    if c: dtc_canon.add(c)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing DTC: {i+1}/{len(dtc_smiles)}')
log.info(f'DTC canonical drugs: {len(dtc_canon)}')

bdb_smiles = bdb.ligand_smiles.unique()
bdb_overlap = set()
for i, s in enumerate(bdb_smiles):
    c = canonicalize(s)
    if c and c in dtc_canon:
        bdb_overlap.add(s)
    if (i + 1) % 50000 == 0: log.info(f'  Canonicalizing BDB: {i+1}/{len(bdb_smiles)}')
log.info(f'BDB overlap: {len(bdb_overlap)}/{len(bdb_smiles)} ({len(bdb_overlap)/len(bdb_smiles)*100:.1f}%)')

bdb_clean = bdb[~bdb.ligand_smiles.isin(bdb_overlap) & bdb.uniprot_id.isin(seqs)].copy()
log.info(f'BDB after exclusion: {len(bdb_clean)} interactions, {bdb_clean.uniprot_id.nunique()} proteins')

# ============================================================
# 4. Tanimoto-based anchor retrieval from DTC
# ============================================================
log.info("Building DTC anchors...")
dtc_drug_to_anchor = {}
dtc_drug_to_second = {}
for smi, grp in dtc_train.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    uid, pki = s.uniprot_id.values[0], s.pki.values[0]
    if pki >= 7.0 and uid in seqs:
        dtc_drug_to_anchor[smi] = (uid, pki)
        if len(s) > 1 and s.uniprot_id.values[1] in seqs:
            dtc_drug_to_second[smi] = (s.uniprot_id.values[1], s.pki.values[1])
log.info(f'DTC anchors (pKi>=7): {len(dtc_drug_to_anchor)}')

dtc_anchor_fps = {}
for s in dtc_drug_to_anchor:
    fp = smi_to_fp(s)
    if fp: dtc_anchor_fps[s] = fp
log.info(f'DTC anchor FPs: {len(dtc_anchor_fps)}')

bdb_drug_fps = {}
for s in bdb_clean.ligand_smiles.unique():
    fp = smi_to_fp(s)
    if fp: bdb_drug_fps[s] = fp
log.info(f'BDB drug FPs: {len(bdb_drug_fps)}')

dtc_fp_smiles = list(dtc_anchor_fps.keys())
dtc_fp_vals = list(dtc_anchor_fps.values())
log.info(f'Tanimoto search: {len(bdb_drug_fps)} BDB x {len(dtc_fp_vals)} DTC anchors')

nearest_dtc = {}
for i, (b_smi, b_fp) in enumerate(bdb_drug_fps.items()):
    sims = DataStructs.BulkTanimotoSimilarity(b_fp, dtc_fp_vals)
    best_idx = int(np.argmax(sims))
    nearest_dtc[b_smi] = (dtc_fp_smiles[best_idx], sims[best_idx])
    if (i + 1) % 10000 == 0:
        log.info(f'  Tanimoto: {i+1}/{len(bdb_drug_fps)}')
log.info(f'Nearest DTC computed for {len(nearest_dtc)} BDB drugs')

sims_vals = [v[1] for v in nearest_dtc.values()]
log.info(f'Tanimoto: Min={min(sims_vals):.3f} Med={np.median(sims_vals):.3f} Mean={np.mean(sims_vals):.3f} Max={max(sims_vals):.3f}')

# Self-anchor check
seq_to_uids = defaultdict(set)
for uid, seq in seqs.items():
    seq_to_uids[seq].add(uid)

# Build eval subset
rows, anc_uids, anc_pkis, tani_vals, anc_drugs = [], [], [], [], []
for i, row in bdb_clean.iterrows():
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if uid not in seqs: continue
    if smi not in nearest_dtc: continue
    dtc_smi, tani = nearest_dtc[smi]
    if dtc_smi not in dtc_drug_to_anchor: continue
    au, ap = dtc_drug_to_anchor[dtc_smi]
    # Self-anchor check
    query_seq = seqs.get(uid, '')
    dtc_equiv = seq_to_uids.get(query_seq, set())
    if au in dtc_equiv:
        if dtc_smi in dtc_drug_to_second:
            au, ap = dtc_drug_to_second[dtc_smi]
        else:
            continue
    if au not in seqs: continue
    rows.append(i)
    anc_uids.append(au); anc_pkis.append(ap)
    tani_vals.append(tani); anc_drugs.append(dtc_smi)

sub = bdb_clean.loc[rows].copy()
sub['anc_uid'] = anc_uids
sub['anc_pki'] = anc_pkis
sub['tanimoto'] = tani_vals
sub['anc_drug'] = anc_drugs
log.info(f'Eval set: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins')

# ============================================================
# 5. Protein families
# ============================================================
FAMILY_CACHE = 'results/bdb_protein_families.json'
if os.path.exists(FAMILY_CACHE):
    families = json.load(open(FAMILY_CACHE))
    log.info(f'Loaded {len(families)} families from cache')
else:
    families = {uid: 'Unknown' for uid in sub.uniprot_id.unique()}
    log.warning('No family cache — all proteins will be Unknown')

sub['family'] = sub['uniprot_id'].map(lambda u: families.get(u, 'Unknown'))

def collapse_family(f):
    if 'kinase' in f.lower(): return 'Protein kinase superfamily'
    elif 'G-protein coupled receptor 1' in f: return 'GPCR family 1'
    elif 'G-protein coupled receptor' in f: return 'GPCR (other)'
    return f

sub['family_group'] = sub.family.apply(collapse_family)
fam_counts = sub.groupby('family_group')['uniprot_id'].nunique()
top_families = set(fam_counts[fam_counts >= 10].index) - {'Unknown'}
sub['family_group'] = sub.family_group.apply(lambda f: f if f in top_families else 'Other')
log.info(f'Top families (>=10 proteins): {len(top_families)}')
for fam in sorted(top_families):
    log.info(f'  {fam}: {fam_counts.get(fam, 0)} proteins')

# Anchor quartiles
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
sub['anchor_q'] = pd.qcut(sub.anc_pki.rank(method='first'), 4, labels=quartile_labels)

# Save anchor metadata
os.makedirs('results', exist_ok=True)
sub[['uniprot_id', 'ligand_smiles', 'pki', 'anc_uid', 'anc_drug', 'anc_pki',
     'tanimoto', 'family', 'family_group', 'anchor_q']].to_csv(
    'results/bdb_new_models_anchor_metadata.csv', index=False)

# ============================================================
# 6. Load models
# ============================================================
from anchor_transfer.model.drugban import DrugBANModel
from anchor_transfer.model.anchor_drugban import AnchorDrugBAN

drugban = DrugBANModel(hidden_dim=128, dropout=0.2).to(device)
ckpt = torch.load('models/drugban_dtc/best_model.pt', map_location=device, weights_only=False)
drugban.load_state_dict(ckpt['model_state_dict']); drugban.eval()
log.info('Loaded DrugBAN')

anchor_drugban = AnchorDrugBAN(hidden_dim=128, dropout=0.2).to(device)
ckpt = torch.load('models/anchor_drugban_dtc/best_model.pt', map_location=device, weights_only=False)
anchor_drugban.load_state_dict(ckpt['model_state_dict']); anchor_drugban.eval()
log.info('Loaded AnchorDrugBAN')

# ============================================================
# 7. Predict DrugBAN
# ============================================================
log.info('Predicting DrugBAN on BDB...')
drugban_preds = [None] * len(sub)
batch_g, batch_p, batch_idx = [], [], []
for i, (_, row) in enumerate(sub.iterrows()):
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    if uid not in seqs: continue
    g = get_graph(smi)
    if g is None: continue
    batch_g.append(g); batch_p.append(enc_prot(seqs[uid])); batch_idx.append(i)
    if len(batch_g) >= 128:
        gb = Batch.from_data_list(batch_g).to(device)
        pt = torch.tensor(batch_p, dtype=torch.long, device=device)
        with torch.no_grad():
            p = drugban(gb, pt).cpu().tolist()
        for k, v in zip(batch_idx, p): drugban_preds[k] = v
        batch_g, batch_p, batch_idx = [], [], []
    if (i + 1) % 50000 == 0:
        log.info(f'  DrugBAN: {i+1}/{len(sub)}')
if batch_g:
    gb = Batch.from_data_list(batch_g).to(device)
    pt = torch.tensor(batch_p, dtype=torch.long, device=device)
    with torch.no_grad():
        p = drugban(gb, pt).cpu().tolist()
    for k, v in zip(batch_idx, p): drugban_preds[k] = v
log.info(f'DrugBAN: {sum(1 for x in drugban_preds if x is not None)} predictions')

# ============================================================
# 8. Predict AnchorDrugBAN
# ============================================================
log.info('Predicting AnchorDrugBAN on BDB...')
anchor_preds = [None] * len(sub)
batch_g, batch_a, batch_q, batch_idx = [], [], [], []
for i, (_, row) in enumerate(sub.iterrows()):
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    au = row['anc_uid']
    if uid not in seqs or au not in seqs: continue
    g = get_graph(smi)
    if g is None: continue
    batch_g.append(g)
    batch_a.append(enc_prot(seqs[au]))
    batch_q.append(enc_prot(seqs[uid]))
    batch_idx.append(i)
    if len(batch_g) >= 128:
        gb = Batch.from_data_list(batch_g).to(device)
        at = torch.tensor(batch_a, dtype=torch.long, device=device)
        qt = torch.tensor(batch_q, dtype=torch.long, device=device)
        with torch.no_grad():
            p = anchor_drugban(gb, at, qt).cpu().tolist()
        for k, v in zip(batch_idx, p): anchor_preds[k] = v
        batch_g, batch_a, batch_q, batch_idx = [], [], [], []
    if (i + 1) % 50000 == 0:
        log.info(f'  AnchorDrugBAN: {i+1}/{len(sub)}')
if batch_g:
    gb = Batch.from_data_list(batch_g).to(device)
    at = torch.tensor(batch_a, dtype=torch.long, device=device)
    qt = torch.tensor(batch_q, dtype=torch.long, device=device)
    with torch.no_grad():
        p = anchor_drugban(gb, at, qt).cpu().tolist()
    for k, v in zip(batch_idx, p): anchor_preds[k] = v
log.info(f'AnchorDrugBAN: {sum(1 for x in anchor_preds if x is not None)} predictions')

# ============================================================
# 9. Enforce common subset + metrics
# ============================================================
sub['drugban_pred'] = drugban_preds
sub['anchor_drugban_pred'] = anchor_preds
pred_cols = ['drugban_pred', 'anchor_drugban_pred']
sub = sub[sub[pred_cols].notna().all(axis=1)].copy()
log.info(f'Common subset: {len(sub)} interactions, {sub.uniprot_id.nunique()} proteins')

model_info = [('DrugBAN', 'drugban_pred'), ('AnchorDrugBAN', 'anchor_drugban_pred')]

# ============================================================
# 10. Per-protein metrics by family
# ============================================================
MIN_INTERACTIONS = 5
pp_rows = []
for uid, grp in sub.groupby('uniprot_id'):
    t = grp.pki.values
    if len(t) < MIN_INTERACTIONS: continue
    fam = grp.family_group.values[0]
    for name, col in model_info:
        p = grp[col].values
        pp_rows.append({
            'uniprot_id': uid, 'family': fam, 'model': name,
            'ci': ci_fn(t, p),
            'rmse': float(np.sqrt(mean_squared_error(t, p))),
            'n': len(t),
            'mean_anc_pki': grp.anc_pki.mean(),
            'mean_tanimoto': grp.tanimoto.mean(),
        })
pp = pd.DataFrame(pp_rows)
pp.to_csv('results/bdb_new_models_per_protein.csv', index=False)
log.info(f'Per-protein metrics: {len(pp)} rows, {pp.uniprot_id.nunique()} proteins')

# Per-family summary
log.info(f'\n=== PER-FAMILY SUMMARY ===')
log.info(f'{"Family":<40} {"Model":<18} {"n_prot":<8} {"CI med":<8} {"CI mean":<8} {"RMSE med":<9}')
log.info('-' * 100)
for fam in sorted(pp.family.unique()):
    for name, _ in model_info:
        d = pp[(pp.family == fam) & (pp.model == name)]
        if len(d) == 0: continue
        ci_v = d.ci.dropna()
        log.info(f'{fam:<40} {name:<18} {len(d):<8} {ci_v.median():<8.3f} {ci_v.mean():<8.3f} {d.rmse.median():<9.3f}')

# Overall
log.info(f'\n=== OVERALL ===')
for name, _ in model_info:
    d = pp[pp.model == name]
    ci_v = d.ci.dropna()
    log.info(f'{name:<18} n={len(d)} proteins  CI: {ci_v.mean():.4f} (med {ci_v.median():.4f})  RMSE: {d.rmse.mean():.4f} (med {d.rmse.median():.4f})')

# Quartile summary
log.info(f'\n=== ANCHOR QUARTILE SUMMARY ===')
for q in quartile_labels:
    qsub = sub[sub.anchor_q == q]
    if len(qsub) < 10: continue
    t = qsub.pki.values
    for name, col in model_info:
        p = qsub[col].values
        log.info(f'{q:<16} {name:<18} n={len(qsub):<7} CI={ci_fn(t,p):.4f} RMSE={np.sqrt(np.mean((t-p)**2)):.4f}')

# Per-protein quartile metrics
pp_q_rows = []
for q in quartile_labels:
    qsub = sub[sub.anchor_q == q]
    for uid, grp in qsub.groupby('uniprot_id'):
        t = grp.pki.values
        if len(t) < MIN_INTERACTIONS: continue
        for name, col in model_info:
            p = grp[col].values
            pp_q_rows.append({
                'quartile': q, 'uniprot_id': uid, 'model': name,
                'ci': ci_fn(t, p), 'rmse': float(np.sqrt(np.mean((t - p) ** 2))),
                'n': len(t),
            })
pp_q = pd.DataFrame(pp_q_rows)
pp_q.to_csv('results/bdb_new_models_per_protein_quartile.csv', index=False)

# ============================================================
# 11. Violin plots by family
# ============================================================
colors = {'DrugBAN': '#ff7f0e', 'AnchorDrugBAN': '#9467bd'}
model_names = [n for n, _ in model_info]

fam_order = pp.groupby('family')['uniprot_id'].nunique().sort_values(ascending=False).index.tolist()
if 'Other' in fam_order:
    fam_order.remove('Other'); fam_order.append('Other')

n_families = len(fam_order)
n_models = len(model_names)
width = 0.8 / n_models

for metric, ylabel, fname in [
    ('ci', 'Per-Protein CI', 'ci'),
    ('rmse', 'Per-Protein RMSE', 'rmse'),
]:
    fig, ax = plt.subplots(figsize=(max(14, n_families * 2.5), 7))
    for j, m in enumerate(model_names):
        positions, data = [], []
        for i, fam in enumerate(fam_order):
            vals = pp[(pp.family == fam) & (pp.model == m)][metric].dropna().values
            if len(vals) >= 3:
                data.append(vals)
                positions.append(i + (j - n_models / 2 + 0.5) * width)
        if data:
            parts = ax.violinplot(data, positions=positions, widths=width * 0.9,
                                  showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(colors[m]); pc.set_alpha(0.7)
            parts['cmedians'].set_color('black')
            parts['cmeans'].set_color('red')

    ax.set_xticks(range(n_families))
    ax.set_xticklabels([f[:25] for f in fam_order], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'BindingDB: {ylabel} by Protein Family (DrugBAN vs AnchorDrugBAN)',
                 fontsize=14, fontweight='bold')
    if metric == 'ci': ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[m], alpha=0.7) for m in model_names]
    ax.legend(handles, model_names, loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs('paper/figures', exist_ok=True)
    plt.savefig(f'paper/figures/fig_bdb_new_models_{fname}_family.png', dpi=300, bbox_inches='tight')
    log.info(f'Saved: paper/figures/fig_bdb_new_models_{fname}_family.png')
    plt.close()

# Anchor quartile boxplots
for metric, ylabel, title_m in [('ci', 'Per-Protein CI', 'CI'), ('rmse', 'Per-Protein RMSE', 'RMSE')]:
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    group_width = 0.78
    box_width = group_width / n_models
    positions = np.arange(len(quartile_labels)) * 1.6
    for mi, mname in enumerate(model_names):
        mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
        box_data = []
        for q in quartile_labels:
            vals = pp_q[(pp_q.model == mname) & (pp_q.quartile == q)][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.9,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[mname]); patch.set_alpha(0.78); patch.set_edgecolor('#333')
        for key in ('whiskers', 'caps', 'medians'):
            for artist in bp[key]:
                artist.set_color('#333'); artist.set_linewidth(1.0)
    ax.set_xticks(positions); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.set_xlabel('Anchor Quartile'); ax.set_ylabel(ylabel)
    if metric == 'ci': ax.set_ylim(0.0, 1.0)
    ax.set_title(f'BindingDB: {title_m} by Anchor Quartile', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    legend_handles = [plt.Line2D([0], [0], color=colors[m], lw=6, label=m) for m in model_names]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(f'paper/figures/fig_bdb_new_models_{metric}_quartile.png', dpi=220, bbox_inches='tight')
    log.info(f'Saved: paper/figures/fig_bdb_new_models_{metric}_quartile.png')
    plt.close()

log.info('\n=== DONE ===')
