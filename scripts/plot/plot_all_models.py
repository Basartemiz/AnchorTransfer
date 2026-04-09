"""Generate combined plots for all 4 models: DrugBAN, AnchorDrugBAN, CoNCISE, ConciseAnchor.

Reads per-protein CSVs from results/, produces combined figures.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

MODEL_COLORS = {
    'DrugBAN': '#ff7f0e',
    'AnchorDrugBAN': '#9467bd',
    'CoNCISE': '#2ca02c',
    'ConciseAnchor': '#d62728',
}
MODEL_ORDER = ['DrugBAN', 'AnchorDrugBAN', 'CoNCISE', 'ConciseAnchor']

os.makedirs('paper/figures', exist_ok=True)

# ============================================================
# 1. DAVIS: load all per-protein results
# ============================================================
davis_drugban = pd.read_csv('results/davis_new_models_per_protein.csv')
davis_concise = pd.read_csv('results/davis_concise_per_protein.csv')
davis_concise_anchor = pd.read_csv('results/davis_concise_anchor_per_protein.csv')

davis_all = pd.concat([davis_drugban, davis_concise, davis_concise_anchor], ignore_index=True)
print(f"Davis: {len(davis_all)} rows, models: {davis_all.model.unique()}")

# ============================================================
# 2. DAVIS: Per-protein CI boxplot by anchor quartile
# ============================================================
quartile_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]

for metric, ylabel, title_m, fname in [
    ('ci', 'Per-Protein CI', 'CI', 'ci'),
    ('rmse', 'Per-Protein RMSE', 'RMSE', 'rmse'),
]:
    fig, ax = plt.subplots(figsize=(14, 6))
    n_models = len(MODEL_ORDER)
    group_width = 0.82
    box_width = group_width / n_models
    positions = np.arange(len(quartile_labels)) * 1.8

    for mi, mname in enumerate(MODEL_ORDER):
        mpos = positions - group_width / 2 + box_width / 2 + mi * box_width
        box_data = []
        for q in quartile_labels:
            vals = davis_all[(davis_all.model == mname) & (davis_all.quartile == q)][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(box_data, positions=mpos, widths=box_width * 0.85,
                        patch_artist=True, showfliers=False, manage_ticks=False)
        color = MODEL_COLORS[mname]
        for patch in bp['boxes']:
            patch.set_facecolor(color); patch.set_alpha(0.78); patch.set_edgecolor('#333')
        for key in ('whiskers', 'caps', 'medians'):
            for artist in bp[key]:
                artist.set_color('#333'); artist.set_linewidth(1.0)

    ax.set_xticks(positions); ax.set_xticklabels(['Q1\n(weakest)', 'Q2', 'Q3', 'Q4\n(strongest)'], fontsize=11)
    ax.set_xlabel('Anchor Quartile', fontsize=12); ax.set_ylabel(ylabel, fontsize=12)
    if metric == 'ci':
        ax.set_ylim(0.0, 1.0)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    ax.set_title(f'Davis: Per-Protein {title_m} by Anchor Quartile', fontsize=15, fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    legend_handles = [plt.Line2D([0], [0], color=MODEL_COLORS[m], lw=8, label=m, alpha=0.78) for m in MODEL_ORDER]
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(f'paper/figures/fig_davis_all_{fname}_quartile.png', dpi=300, bbox_inches='tight')
    print(f'Saved: paper/figures/fig_davis_all_{fname}_quartile.png')
    plt.close()

# ============================================================
# 3. DAVIS: Quartile line plot (CI mean per quartile)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, (metric, ylabel) in zip(axes, [('ci', 'Per-Protein CI (mean)'), ('rmse', 'Per-Protein RMSE (mean)')]):
    x = np.arange(len(quartile_labels))
    for mname in MODEL_ORDER:
        ys = []
        for q in quartile_labels:
            vals = davis_all[(davis_all.model == mname) & (davis_all.quartile == q)][metric].dropna()
            ys.append(vals.mean() if len(vals) > 0 else np.nan)
        ax.plot(x, ys, marker='o', linewidth=2.5, markersize=7, color=MODEL_COLORS[mname], label=mname)
    ax.set_xticks(x); ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
    ax.set_xlabel('Anchor Quartile'); ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.25)
    if metric == 'ci':
        ax.set_ylim(0.3, 0.9)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=False, fontsize=11)
fig.suptitle('Davis: Anchor Quartile Analysis (All Models)', fontsize=15, fontweight='bold', y=1.06)
fig.tight_layout()
fig.savefig('paper/figures/fig_davis_all_quartile_lines.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/fig_davis_all_quartile_lines.png')
plt.close()

# ============================================================
# 4. DAVIS: Overall per-protein CI bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (metric, ylabel, title) in zip(axes, [
    ('ci', 'Per-Protein CI', 'Concordance Index'),
    ('rmse', 'Per-Protein RMSE', 'RMSE'),
]):
    means, meds, colors = [], [], []
    for mname in MODEL_ORDER:
        vals = davis_all[davis_all.model == mname][metric].dropna()
        means.append(vals.mean()); meds.append(vals.median())
        colors.append(MODEL_COLORS[mname])
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='#333', linewidth=0.8)
    ax.scatter(x, meds, color='black', zorder=5, s=40, marker='D', label='median')
    ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(f'Davis: {title}', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    if metric == 'ci':
        ax.set_ylim(0, 0.85)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    for i, v in enumerate(means):
        ax.text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)

fig.tight_layout()
fig.savefig('paper/figures/fig_davis_all_overall_bar.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/fig_davis_all_overall_bar.png')
plt.close()

# ============================================================
# 5. BDB: load all per-protein results
# ============================================================
bdb_drugban = pd.read_csv('results/bdb_new_models_per_protein.csv')
bdb_concise_anchor = pd.read_csv('results/bdb_concise_anchor_per_protein.csv')
bdb_concise = pd.read_csv('results/bdb_concise_per_protein.csv')

bdb_all = pd.concat([bdb_drugban, bdb_concise, bdb_concise_anchor], ignore_index=True)
print(f"\nBDB: {len(bdb_all)} rows, models: {bdb_all.model.unique()}")

# ============================================================
# 6. BDB: Violin plots by protein family
# ============================================================
fam_order = bdb_all.groupby('family')['uniprot_id'].nunique().sort_values(ascending=False).index.tolist()
if 'Other' in fam_order:
    fam_order.remove('Other'); fam_order.append('Other')

n_families = len(fam_order)
n_models = len(MODEL_ORDER)
width = 0.8 / n_models

for metric, ylabel, fname in [('ci', 'Per-Protein CI', 'ci'), ('rmse', 'Per-Protein RMSE', 'rmse')]:
    fig, ax = plt.subplots(figsize=(max(16, n_families * 2.8), 7.5))
    for j, m in enumerate(MODEL_ORDER):
        positions, data = [], []
        for i, fam in enumerate(fam_order):
            vals = bdb_all[(bdb_all.family == fam) & (bdb_all.model == m)][metric].dropna().values
            if len(vals) >= 3:
                data.append(vals)
                positions.append(i + (j - n_models / 2 + 0.5) * width)
        if data:
            parts = ax.violinplot(data, positions=positions, widths=width * 0.9,
                                  showmeans=True, showmedians=True)
            for pc in parts['bodies']:
                pc.set_facecolor(MODEL_COLORS[m]); pc.set_alpha(0.7)
            parts['cmedians'].set_color('black')
            parts['cmeans'].set_color('red')

    ax.set_xticks(range(n_families))
    ax.set_xticklabels([f[:30] for f in fam_order], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'BindingDB: {ylabel} by Protein Family', fontsize=15, fontweight='bold')
    if metric == 'ci': ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=MODEL_COLORS[m], alpha=0.7) for m in MODEL_ORDER]
    ax.legend(handles, MODEL_ORDER, loc='upper right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'paper/figures/fig_bdb_all_{fname}_family.png', dpi=300, bbox_inches='tight')
    print(f'Saved: paper/figures/fig_bdb_all_{fname}_family.png')
    plt.close()

# ============================================================
# 7. BDB: Overall per-protein bar chart
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (metric, ylabel, title) in zip(axes, [
    ('ci', 'Per-Protein CI', 'Concordance Index'),
    ('rmse', 'Per-Protein RMSE', 'RMSE'),
]):
    means, meds, colors = [], [], []
    for mname in MODEL_ORDER:
        vals = bdb_all[bdb_all.model == mname][metric].dropna()
        means.append(vals.mean() if len(vals) > 0 else 0)
        meds.append(vals.median() if len(vals) > 0 else 0)
        colors.append(MODEL_COLORS[mname])
    x = np.arange(len(MODEL_ORDER))
    bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor='#333', linewidth=0.8)
    ax.scatter(x, meds, color='black', zorder=5, s=40, marker='D', label='median')
    ax.set_xticks(x); ax.set_xticklabels(MODEL_ORDER, rotation=15, ha='right', fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(f'BindingDB: {title}', fontweight='bold')
    ax.grid(axis='y', alpha=0.25)
    if metric == 'ci':
        ax.set_ylim(0, 0.7)
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    for i, v in enumerate(means):
        ax.text(i, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

fig.tight_layout()
fig.savefig('paper/figures/fig_bdb_all_overall_bar.png', dpi=300, bbox_inches='tight')
print('Saved: paper/figures/fig_bdb_all_overall_bar.png')
plt.close()

print('\n=== ALL PLOTS DONE ===')
