"""BDB-trained cross-dataset plots: Davis + GLASS2."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

OUT = 'paper/figures/bdb_cross_dataset'
os.makedirs(OUT, exist_ok=True)

COLORS = {'CoNCISE-BDB': '#2ca02c', 'ConciseAnchor-BDB': '#d62728', 'Retrieval': '#7f7f7f'}
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]

def ci_fn(y, f):
    n = len(y)
    if n < 2: return 0.5
    y, f = np.array(y), np.array(f)
    idx = np.triu_indices(n, k=1)
    i, j = idx[0], idx[1]
    dt = y[i] - y[j]; dp = f[i] - f[j]; t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

# Load data
davis = pd.read_csv('results/bdb_to_davis_predictions.csv')
glass = pd.read_csv('results/bdb_to_glass_predictions.csv')

def per_protein_ci(df, pred_col, min_n=3):
    """Compute per-protein CI, assign quartile from majority of interactions."""
    results = []
    for uid, grp in df.groupby('uniprot_id'):
        tv, pv = grp.pki.values, grp[pred_col].values
        if len(tv) < min_n:
            continue
        q = grp.anchor_q.mode().iloc[0]  # majority quartile
        results.append({'anchor_q': q, 'uid': uid, 'ci': ci_fn(tv, pv), 'n': len(tv)})
    return pd.DataFrame(results)

# ============================================================
# 1. Davis: per-protein CI boxplot by quartile
# ============================================================
davis_concise = per_protein_ci(davis, 'concise_pred')
davis_concise['model'] = 'CoNCISE-BDB'
davis_anchor = per_protein_ci(davis, 'anchor_pred')
davis_anchor['model'] = 'ConciseAnchor-BDB'
davis_pp = pd.concat([davis_concise, davis_anchor])

fig, ax = plt.subplots(figsize=(10, 5.5))
models = ['CoNCISE-BDB', 'ConciseAnchor-BDB']
group_width = 0.6; box_width = group_width / 2
positions = np.arange(len(quartile_labels)) * 1.6

for mi, m in enumerate(models):
    mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
    box_data = []
    for q in quartile_labels:
        vals = davis_pp[(davis_pp.model == m) & (davis_pp.anchor_q == q)].ci.dropna().tolist()
        box_data.append(vals if vals else [np.nan])
    bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.85,
                    patch_artist=True, showfliers=False, manage_ticks=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.78); patch.set_edgecolor('#333')
    for key in ('whiskers', 'caps', 'medians'):
        for artist in bp[key]:
            artist.set_color('#333'); artist.set_linewidth(1.0)

ax.set_xticks(positions)
ax.set_xticklabels(['Q1\n(weakest)', 'Q2', 'Q3', 'Q4\n(strongest)'], fontsize=11)
ax.set_xlabel('Anchor Quartile', fontsize=12)
ax.set_ylabel('Per-Protein CI', fontsize=12)
ax.set_ylim(0.0, 1.0)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
ax.set_title('Davis (BDB-trained): Per-Protein CI by Anchor Quartile', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.25)
handles = [plt.Line2D([0], [0], color=COLORS[m], lw=8, label=m, alpha=0.78) for m in models]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_bdb_to_davis_ci_quartile.png', dpi=300, bbox_inches='tight')
print(f'Saved: {OUT}/fig_bdb_to_davis_ci_quartile.png')
plt.close()

# ============================================================
# 2. GLASS2: per-protein CI boxplot by quartile
# ============================================================
glass_concise = per_protein_ci(glass, 'concise_pred')
glass_concise['model'] = 'CoNCISE-BDB'
glass_anchor = per_protein_ci(glass, 'anchor_pred')
glass_anchor['model'] = 'ConciseAnchor-BDB'
glass_pp = pd.concat([glass_concise, glass_anchor])

fig, ax = plt.subplots(figsize=(10, 5.5))
for mi, m in enumerate(models):
    mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
    box_data = []
    for q in quartile_labels:
        vals = glass_pp[(glass_pp.model == m) & (glass_pp.anchor_q == q)].ci.dropna().tolist()
        box_data.append(vals if vals else [np.nan])
    bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.85,
                    patch_artist=True, showfliers=False, manage_ticks=False)
    for patch in bp['boxes']:
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.78); patch.set_edgecolor('#333')
    for key in ('whiskers', 'caps', 'medians'):
        for artist in bp[key]:
            artist.set_color('#333'); artist.set_linewidth(1.0)

ax.set_xticks(positions)
ax.set_xticklabels(['Q1\n(weakest)', 'Q2', 'Q3', 'Q4\n(strongest)'], fontsize=11)
ax.set_xlabel('Anchor Quartile', fontsize=12)
ax.set_ylabel('Per-Protein CI', fontsize=12)
ax.set_ylim(0.0, 1.0)
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
ax.set_title('GLASS2 GPCRs (BDB-trained): Per-Protein CI by Anchor Quartile', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.25)
handles = [plt.Line2D([0], [0], color=COLORS[m], lw=8, label=m, alpha=0.78) for m in models]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=11)
fig.tight_layout()
fig.savefig(f'{OUT}/fig_bdb_to_glass_ci_quartile.png', dpi=300, bbox_inches='tight')
print(f'Saved: {OUT}/fig_bdb_to_glass_ci_quartile.png')
plt.close()

# ============================================================
# 3. Grouped bar chart: CI + AUROC for both benchmarks
# ============================================================
from sklearn.metrics import roc_auc_score

def auroc_safe(t, p):
    b = t >= 7.0; nb = t <= 5.0; m = b | nb
    if m.sum() == 0 or b[m].sum() == 0 or nb[m].sum() == 0: return float('nan')
    return float(roc_auc_score(b[m].astype(int), p[m]))

bar_data = {}
for name, df in [('Davis', davis), ('GLASS2', glass)]:
    t = df.pki.values
    bar_data[name] = {
        'Retrieval': {'CI': ci_fn(t, df.anchor_pki.values), 'AUROC': auroc_safe(t, df.anchor_pki.values)},
        'CoNCISE-BDB': {'CI': ci_fn(t, df.concise_pred.values), 'AUROC': auroc_safe(t, df.concise_pred.values)},
        'ConciseAnchor-BDB': {'CI': ci_fn(t, df.anchor_pred.values), 'AUROC': auroc_safe(t, df.anchor_pred.values)},
    }

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bar_models = ['Retrieval', 'CoNCISE-BDB', 'ConciseAnchor-BDB']
bar_colors = [COLORS[m] for m in bar_models]
x = np.arange(2)
width = 0.22

for ai, metric in enumerate(['CI', 'AUROC']):
    ax = axes[ai]
    for mi, m in enumerate(bar_models):
        vals = [bar_data[bench][m][metric] for bench in ['Davis', 'GLASS2']]
        bars = ax.bar(x + (mi - 1) * width, vals, width * 0.9, label=m, color=bar_colors[mi], alpha=0.82, edgecolor='#333', linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Davis\n(108 proteins)', 'GLASS2\n(144 GPCRs)'], fontsize=11)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'BDB→Cross-Dataset {metric}', fontsize=13, fontweight='bold')
    ax.set_ylim(0.35, 0.95)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.25)
    if ai == 0:
        ax.legend(frameon=False, fontsize=9, loc='upper left')

fig.tight_layout()
fig.savefig(f'{OUT}/fig_bdb_cross_dataset_bar.png', dpi=300, bbox_inches='tight')
print(f'Saved: {OUT}/fig_bdb_cross_dataset_bar.png')
plt.close()

print('\nAll plots generated.')
