"""CoNCISE vs ConciseAnchor comparison plots."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('paper/figures', exist_ok=True)

COLORS = {'CoNCISE': '#2ca02c', 'ConciseAnchor': '#d62728'}
MODELS = ['CoNCISE', 'ConciseAnchor']
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]

# Load Davis
d_c = pd.read_csv('results/davis_concise_per_protein.csv')
d_a = pd.read_csv('results/davis_concise_anchor_per_protein.csv')
davis = pd.concat([d_c, d_a], ignore_index=True)

# Load BDB
b_c = pd.read_csv('results/bdb_concise_per_protein.csv')
b_a = pd.read_csv('results/bdb_concise_anchor_per_protein.csv')
bdb = pd.concat([b_c, b_a], ignore_index=True)

# ---- Davis: CI + RMSE quartile boxplots ----
for metric, ylabel, title_m in [('ci', 'Per-Protein CI', 'CI'), ('rmse', 'Per-Protein RMSE', 'RMSE')]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    group_width = 0.6
    box_width = group_width / 2
    positions = np.arange(len(quartile_labels)) * 1.6

    for mi, m in enumerate(MODELS):
        mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
        box_data = []
        for q in quartile_labels:
            vals = davis[(davis.model == m) & (davis.quartile == q)][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.85,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.78); patch.set_edgecolor('#333')
        for key in ('whiskers', 'caps', 'medians'):
            for artist in bp[key]:
                artist.set_color('#333'); artist.set_linewidth(1.0)

    ax.set_xticks(positions); ax.set_xticklabels(['Q1\n(weakest)', 'Q2', 'Q3', 'Q4\n(strongest)'], fontsize=11)
    ax.set_xlabel('Anchor Quartile', fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    if metric == 'ci':
        ax.set_ylim(0.0, 1.0); ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_title(f'Davis: {title_m} — CoNCISE vs ConciseAnchor', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    handles = [plt.Line2D([0], [0], color=COLORS[m], lw=8, label=m, alpha=0.78) for m in MODELS]
    ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f'paper/figures/fig_davis_concise_vs_anchor_{metric}.png', dpi=300, bbox_inches='tight')
    print(f'Saved: paper/figures/fig_davis_concise_vs_anchor_{metric}.png')
    plt.close()

# ---- Davis: Quartile line plot ----
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(len(quartile_labels))
for ax, (metric, ylabel) in zip(axes, [('ci', 'Per-Protein CI (mean)'), ('rmse', 'Per-Protein RMSE (mean)')]):
    for m in MODELS:
        ys = [davis[(davis.model == m) & (davis.quartile == q)][metric].dropna().mean() for q in quartile_labels]
        ax.plot(x, ys, marker='o', linewidth=2.5, markersize=8, color=COLORS[m], label=m)
    ax.set_xticks(x); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4']); ax.set_xlabel('Anchor Quartile')
    ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.25)
    if metric == 'ci': ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)
fig.suptitle('Davis: CoNCISE vs ConciseAnchor by Anchor Quartile', fontsize=14, fontweight='bold', y=1.06)
fig.tight_layout()
fig.savefig('paper/figures/fig_davis_concise_vs_anchor_lines.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/fig_davis_concise_vs_anchor_lines.png')
plt.close()

# ---- BDB: Violin by family ----
fam_order = bdb.groupby('family')['uniprot_id'].nunique().sort_values(ascending=False).index.tolist()
if 'Other' in fam_order: fam_order.remove('Other'); fam_order.append('Other')
n_fam = len(fam_order)
width = 0.8 / 2

for metric, ylabel in [('ci', 'Per-Protein CI'), ('rmse', 'Per-Protein RMSE')]:
    fig, ax = plt.subplots(figsize=(max(14, n_fam * 2.5), 7))
    for j, m in enumerate(MODELS):
        positions, data = [], []
        for i, fam in enumerate(fam_order):
            vals = bdb[(bdb.family == fam) & (bdb.model == m)][metric].dropna().values
            if len(vals) >= 3:
                data.append(vals); positions.append(i + (j - 0.5) * width)
        if data:
            parts = ax.violinplot(data, positions=positions, widths=width * 0.9, showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(COLORS[m]); pc.set_alpha(0.7)
            parts['cmedians'].set_color('black'); parts['cmeans'].set_color('red')
    ax.set_xticks(range(n_fam))
    ax.set_xticklabels([f[:28] for f in fam_order], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'BindingDB: {ylabel} by Protein Family', fontsize=14, fontweight='bold')
    if metric == 'ci': ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=COLORS[m], alpha=0.7) for m in MODELS]
    ax.legend(handles, MODELS, loc='upper right', fontsize=11); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'paper/figures/fig_bdb_concise_vs_anchor_{metric}_family.png', dpi=300, bbox_inches='tight')
    print(f'Saved: paper/figures/fig_bdb_concise_vs_anchor_{metric}_family.png')
    plt.close()

# ---- BDB: Overall bar chart ----
fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
for ax, (metric, ylabel, title) in zip(axes, [('ci', 'Per-Protein CI', 'CI'), ('rmse', 'Per-Protein RMSE', 'RMSE')]):
    means, colors = [], []
    for m in MODELS:
        vals = bdb[bdb.model == m][metric].dropna()
        means.append(vals.mean()); colors.append(COLORS[m])
    x = np.arange(2)
    ax.bar(x, means, color=colors, alpha=0.8, edgecolor='#333', width=0.5)
    ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=11)
    ax.set_ylabel(ylabel); ax.set_title(f'BDB: {title}', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    if metric == 'ci': ax.set_ylim(0.4, 0.65); ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    for i, v in enumerate(means): ax.text(i, v + 0.003, f'{v:.3f}', ha='center', fontsize=11)
fig.tight_layout()
fig.savefig('paper/figures/fig_bdb_concise_vs_anchor_bar.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/fig_bdb_concise_vs_anchor_bar.png')
plt.close()

print('\n=== DONE ===')
