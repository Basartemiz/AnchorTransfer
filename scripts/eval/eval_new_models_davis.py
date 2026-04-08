"""Evaluate DrugBAN + AnchorDrugBAN on Davis with anchor quartile + per-protein analysis.

Follows the same structure as eval_anchor_quartiles_vs_baselines.py:
  1. Load Davis, build dataset-internal anchors
  2. Predict with DrugBAN (pairwise) and AnchorDrugBAN (anchor-based)
  3. Per-protein CI/RMSE distributions by anchor quartile
  4. Generate PNG plots (line + boxplot)
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
from torch_geometric.data import Data, Batch

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

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

def auroc_safe(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))

# ============================================================
# 1. Load data
# ============================================================
log.info("Loading data...")
davis_raw = pd.read_csv(DATA_DIR / 'raw' / 'davis' / 'davis_benchmark.csv')
davis = davis_raw.rename(columns={'protein_name': 'uniprot_id', 'drug_smiles': 'ligand_smiles', 'pKd': 'pki'})
seqs = {}
if 'protein_sequence' in davis_raw.columns:
    for _, r in davis_raw.drop_duplicates('protein_name').iterrows():
        seqs[r['protein_name']] = r['protein_sequence']
# Also load DTC sequences
merged_seq_path = DATA_DIR / 'processed' / 'merged_sequences.json'
if merged_seq_path.exists():
    seqs.update(json.load(open(merged_seq_path)))
log.info(f'Davis: {len(davis)} interactions, {davis.uniprot_id.nunique()} proteins')
log.info(f'Sequences: {len(seqs)}')

# Load graph cache
GRAPH_CACHE = Path('data/processed/drugban_graph_cache.pt')
if not GRAPH_CACHE.exists():
    GRAPH_CACHE = Path('drugban_graph_cache.pt')
if GRAPH_CACHE.exists():
    raw = torch.load(GRAPH_CACHE, map_location='cpu', weights_only=False)
    graph_cache = {smi: Data(x=raw['x'][smi], edge_index=raw['edge_index'][smi]) for smi in raw['x']}
    log.info(f'Graph cache: {len(graph_cache)} drugs')
else:
    log.warning('No graph cache! Building on the fly...')
    from anchor_transfer.model.drug_encoder import smiles_to_graph
    graph_cache = {}

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
# 2. DTC train split + anchors from DTC
# ============================================================
dtc = pd.read_csv(DATA_DIR / 'processed' / 'dtc_training_interactions.csv')
dtc_valid = dtc[dtc.uniprot_id.isin(seqs)]
dtc_prots = sorted(set(dtc_valid.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
dtc_train_prots = set(dtc_prots[nt + nv:])
dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(dtc_train_prots)]
log.info(f'DTC train: {len(dtc_train)} interactions, {dtc_train.uniprot_id.nunique()} proteins')

# DTC train overlap filtering for Davis
train_seqs = {seqs[uid] for uid in dtc_train_prots if uid in seqs}
train_drugs = set(dtc_train.ligand_smiles.unique())

# Remove DTC-overlapping proteins and drugs from Davis
davis_filt = davis.copy()
overlap_prots = set()
for uid in davis_filt.uniprot_id.unique():
    if uid in dtc_train_prots or (uid in seqs and seqs[uid] in train_seqs):
        overlap_prots.add(uid)
davis_filt = davis_filt[~davis_filt.uniprot_id.isin(overlap_prots)]
davis_filt = davis_filt[~davis_filt.ligand_smiles.isin(train_drugs)]
davis_filt = davis_filt[davis_filt.uniprot_id.isin(seqs)]
log.info(f'Davis after overlap filtering: {len(davis_filt)} interactions, {davis_filt.uniprot_id.nunique()} proteins')

# ============================================================
# 3. Build dataset-internal anchors
# ============================================================
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
    rows.append(i)
    anchor_uids.append(au)
    anchor_pkis.append(ap)

subset = davis_filt.loc[rows].copy()
subset['anchor_uid'] = anchor_uids
subset['anchor_pki'] = anchor_pkis
log.info(f'Anchored subset: {len(subset)} interactions, {subset.uniprot_id.nunique()} proteins')

# Assign quartiles
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
subset['anchor_q'] = pd.qcut(
    subset.anchor_pki.rank(method='first'), 4, labels=quartile_labels
)

# ============================================================
# 4. Load models
# ============================================================
from anchor_transfer.model.drugban import DrugBANModel
from anchor_transfer.model.anchor_drugban import AnchorDrugBAN

drugban = DrugBANModel(hidden_dim=128, dropout=0.2).to(device)
ckpt = torch.load('models/drugban_dtc/best_model.pt', map_location=device, weights_only=False)
drugban.load_state_dict(ckpt['model_state_dict']); drugban.eval()
log.info(f'Loaded DrugBAN (epoch {ckpt.get("epoch", "?")})')

anchor_drugban = AnchorDrugBAN(hidden_dim=128, dropout=0.2).to(device)
ckpt = torch.load('models/anchor_drugban_dtc/best_model.pt', map_location=device, weights_only=False)
anchor_drugban.load_state_dict(ckpt['model_state_dict']); anchor_drugban.eval()
log.info(f'Loaded AnchorDrugBAN (epoch {ckpt.get("epoch", "?")})')

# ============================================================
# 5. Predict DrugBAN (pairwise — ignores anchor)
# ============================================================
log.info("Predicting DrugBAN...")
drugban_preds = [None] * len(subset)
batch_g, batch_p, batch_idx = [], [], []
for i, (_, row) in enumerate(subset.iterrows()):
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
if batch_g:
    gb = Batch.from_data_list(batch_g).to(device)
    pt = torch.tensor(batch_p, dtype=torch.long, device=device)
    with torch.no_grad():
        p = drugban(gb, pt).cpu().tolist()
    for k, v in zip(batch_idx, p): drugban_preds[k] = v
log.info(f'DrugBAN: {sum(1 for x in drugban_preds if x is not None)} predictions')

# ============================================================
# 6. Predict AnchorDrugBAN (anchor-based)
# ============================================================
log.info("Predicting AnchorDrugBAN...")
anchor_preds = [None] * len(subset)
batch_g, batch_a, batch_q, batch_idx = [], [], [], []
for i, (_, row) in enumerate(subset.iterrows()):
    uid, smi = row['uniprot_id'], row['ligand_smiles']
    au = row['anchor_uid']
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
if batch_g:
    gb = Batch.from_data_list(batch_g).to(device)
    at = torch.tensor(batch_a, dtype=torch.long, device=device)
    qt = torch.tensor(batch_q, dtype=torch.long, device=device)
    with torch.no_grad():
        p = anchor_drugban(gb, at, qt).cpu().tolist()
    for k, v in zip(batch_idx, p): anchor_preds[k] = v
log.info(f'AnchorDrugBAN: {sum(1 for x in anchor_preds if x is not None)} predictions')

# ============================================================
# 7. Store predictions, enforce common subset
# ============================================================
subset['drugban_pred'] = drugban_preds
subset['anchor_drugban_pred'] = anchor_preds
pred_cols = ['drugban_pred', 'anchor_drugban_pred']
subset = subset[subset[pred_cols].notna().all(axis=1)].copy()
log.info(f'Common subset: {len(subset)} interactions, {subset.uniprot_id.nunique()} proteins')

# ============================================================
# 8. Overall metrics
# ============================================================
model_info = [
    ('DrugBAN', 'drugban_pred'),
    ('AnchorDrugBAN', 'anchor_drugban_pred'),
]

log.info(f'\n=== OVERALL DAVIS RESULTS ===')
for name, col in model_info:
    t, p = subset.pki.values, subset[col].values
    ci = ci_fn(t, p)
    rmse = np.sqrt(mean_squared_error(t, p))
    auroc = auroc_safe(t, p)
    r = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0
    log.info(f'{name:<20} CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(t)}')

# ============================================================
# 9. Anchor quartile analysis
# ============================================================
log.info(f'\n=== ANCHOR QUARTILE ANALYSIS ===')
log.info(f'{"Quartile":<16} {"Anchor pKi":<16} {"n":<7} | {"DrugBAN CI":<12} {"AncDrugBAN CI":<14} | {"DrugBAN RMSE":<13} {"AncDrugBAN RMSE":<15}')
log.info('-' * 105)

quartile_results = []
for q in quartile_labels:
    sub = subset[subset.anchor_q == q]
    if len(sub) == 0: continue
    lo, hi = sub.anchor_pki.min(), sub.anchor_pki.max()
    t = sub.pki.values
    row = {'quartile': q, 'anchor_range': f'[{lo:.1f}-{hi:.1f}]', 'n': len(sub)}
    for name, col in model_info:
        p = sub[col].values
        row[f'{name}_ci'] = ci_fn(t, p)
        row[f'{name}_rmse'] = float(np.sqrt(np.mean((t - p) ** 2)))
        row[f'{name}_auroc'] = auroc_safe(t, p)
    log.info(f'{q:<16} [{lo:.1f}-{hi:.1f}]{"":<6} {row["n"]:<7} | {row["DrugBAN_ci"]:<12.4f} {row["AnchorDrugBAN_ci"]:<14.4f} | {row["DrugBAN_rmse"]:<13.4f} {row["AnchorDrugBAN_rmse"]:<15.4f}')
    quartile_results.append(row)

# ============================================================
# 10. Per-protein metrics by anchor quartile
# ============================================================
pp_rows = []
for q in quartile_labels:
    sub = subset[subset.anchor_q == q]
    for uid, grp in sub.groupby('uniprot_id'):
        t = grp.pki.values
        if len(t) < 5: continue
        for name, col in model_info:
            p = grp[col].values
            pp_rows.append({
                'quartile': q, 'uniprot_id': uid, 'model': name,
                'n': len(t), 'ci': ci_fn(t, p),
                'rmse': float(np.sqrt(np.mean((t - p) ** 2))),
            })

pp_df = pd.DataFrame(pp_rows)
log.info(f'\nPer-protein metrics: {len(pp_df)} rows, {pp_df.uniprot_id.nunique()} proteins')

# Overall per-protein summary
log.info(f'\n=== PER-PROTEIN SUMMARY ===')
for name, _ in model_info:
    d = pp_df[pp_df.model == name]
    ci_vals = d.ci.dropna()
    log.info(f'{name:<20} n={len(d)} proteins  CI: mean={ci_vals.mean():.4f} med={ci_vals.median():.4f}  RMSE: mean={d.rmse.mean():.4f} med={d.rmse.median():.4f}')

# ============================================================
# 11. Save results
# ============================================================
os.makedirs('results', exist_ok=True)
qr_df = pd.DataFrame(quartile_results)
qr_df.to_csv('results/davis_new_models_quartiles.csv', index=False)
pp_df.to_csv('results/davis_new_models_per_protein.csv', index=False)
subset.to_csv('results/davis_new_models_predictions.csv', index=False)
log.info('Saved CSVs to results/')

# ============================================================
# 12. Plots
# ============================================================
MODEL_COLORS = {
    'DrugBAN': '#ff7f0e',
    'AnchorDrugBAN': '#9467bd',
}

# --- Figure 1: Quartile line plot (CI, RMSE, AUROC) ---
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
x = np.arange(len(quartile_labels))

for ax, (title, suffix) in zip(axes, [('AUROC', 'auroc'), ('CI', 'ci'), ('RMSE', 'rmse')]):
    for name, _ in model_info:
        col = f'{name}_{suffix}'
        ys = [qr_df[qr_df.quartile == q].iloc[0][col] if len(qr_df[qr_df.quartile == q]) > 0 else np.nan for q in quartile_labels]
        ax.plot(x, ys, marker='o', linewidth=2, markersize=6, color=MODEL_COLORS[name], label=name)
    ax.set_xticks(x); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.set_title(title); ax.grid(axis='y', alpha=0.25)
    if suffix in ('auroc', 'ci'): ax.set_ylim(0.0, 1.0)

axes[0].set_ylabel('Metric Value')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=4, frameon=False)
fig.suptitle('Davis: Anchor Quartile Analysis (DrugBAN vs AnchorDrugBAN)', fontsize=14, fontweight='bold', y=1.08)
fig.tight_layout()
os.makedirs('paper/figures', exist_ok=True)
fig.savefig('paper/figures/fig_davis_new_models_quartile.png', dpi=220, bbox_inches='tight')
log.info('Saved: paper/figures/fig_davis_new_models_quartile.png')
plt.close()

# --- Figure 2: Per-protein CI distribution by quartile (boxplot) ---
for metric, ylabel, title_metric in [('ci', 'Per-Protein CI', 'CI'), ('rmse', 'Per-Protein RMSE', 'RMSE')]:
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    model_order = [n for n, _ in model_info]
    group_width = 0.78
    box_width = group_width / len(model_order)
    positions = np.arange(len(quartile_labels)) * 1.6

    for mi, mname in enumerate(model_order):
        mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
        box_data = []
        for q in quartile_labels:
            vals = pp_df[(pp_df.model == mname) & (pp_df.quartile == q)][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.9,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        color = MODEL_COLORS[mname]
        for patch in bp['boxes']:
            patch.set_facecolor(color); patch.set_alpha(0.78); patch.set_edgecolor('#333')
        for key in ('whiskers', 'caps', 'medians'):
            for artist in bp[key]:
                artist.set_color('#333'); artist.set_linewidth(1.0)

    ax.set_xticks(positions); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.set_xlabel('Anchor Quartile'); ax.set_ylabel(ylabel)
    if metric == 'ci': ax.set_ylim(0.0, 1.0)
    ax.set_title(f'Davis: {title_metric} Distribution by Anchor Quartile', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    legend_handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], lw=6, label=m) for m in model_order]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(f'paper/figures/fig_davis_new_models_{metric}_distribution.png', dpi=220, bbox_inches='tight')
    log.info(f'Saved: paper/figures/fig_davis_new_models_{metric}_distribution.png')
    plt.close()

log.info('\n=== DONE ===')
