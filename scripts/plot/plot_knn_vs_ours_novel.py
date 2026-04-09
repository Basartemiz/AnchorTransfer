"""Plot: ConciseAnchor vs kNN baselines on truly novel GLASS2 proteins.

Shows per-protein CI distributions by anchor pKi quartile.
Panel A: Overall comparison (box + strip)
Panel B: Quartile breakdown
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
from pathlib import Path

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE = Path("results")

# ── Compute per-protein CI ──────────────────────────────────────

def per_protein_ci(df, col):
    results = {}
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        if len(yt) < 3 or yp.std() < 1e-8:
            if len(yt) >= 3: results[uid] = 0.5
            continue
        c = d = t = 0
        for i, j in combinations(range(len(yt)), 2):
            if yt[i] == yt[j]: continue
            if (yp[i]>yp[j]) == (yt[i]>yt[j]): c += 1
            elif yp[i] == yp[j]: t += 1
            else: d += 1
        tot = c+d+t
        results[uid] = (c+0.5*t)/tot if tot else 0.5
    return results

# ── Load and merge ──────────────────────────────────────────────

ours = pd.read_csv(BASE / "bdb_to_glass_prot_only_predictions.csv")
knn = pd.read_csv(BASE / "knn_glass_glass2_novel_30.csv")

merged = knn.merge(
    ours[["uniprot_id", "ligand_smiles", "concise_pred", "anchor_pred", "anchor_q"]],
    on=["uniprot_id", "ligand_smiles"], how="inner"
)
print(f"Merged: {len(merged)} interactions, {merged.uniprot_id.nunique()} proteins")

# Methods to compare
methods = {
    "ConciseAnchor": "concise_pred",
    "Prot-kNN (k=1)": "prot_knn_k1",
    "Prot-kNN (k=5)": "prot_knn_k5",
    "Joint-kNN (k=10)": "joint_knn_k10",
    "Anchor-kNN": "anchor_knn",
}

# Per-protein CI
pp = {}
for label, col in methods.items():
    pp[label] = per_protein_ci(merged, col)

# Assign quartile per protein by median pKi
prot_median = merged.groupby("uniprot_id").pki.median()
q_edges = np.quantile(prot_median.values, [0, 0.25, 0.5, 0.75, 1.0])
q_labels = ["Q1\n(weakest)", "Q2", "Q3", "Q4\n(strongest)"]
prot_q = pd.cut(prot_median, bins=q_edges, labels=q_labels, include_lowest=True).to_dict()

# Colors
colors = {
    "ConciseAnchor": "#2E86AB",
    "Prot-kNN (k=1)": "#E8998D",
    "Prot-kNN (k=5)": "#D4A373",
    "Joint-kNN (k=10)": "#A8DADC",
    "Anchor-kNN": "#CDB4DB",
}

# ── Figure ──────────────────────────────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                gridspec_kw={"width_ratios": [1, 2.2]})

# ── Panel A: Overall box + strip ────────────────────────────────

method_labels = list(methods.keys())
box_data = []
for label in method_labels:
    vals = list(pp[label].values())
    box_data.append(vals)

bp = ax1.boxplot(box_data, positions=range(len(method_labels)),
                 widths=0.5, patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.5))

for patch, label in zip(bp["boxes"], method_labels):
    patch.set_facecolor(colors[label])
    patch.set_alpha(0.7)

# Strip plot
for i, label in enumerate(method_labels):
    vals = list(pp[label].values())
    jitter = np.random.uniform(-0.12, 0.12, len(vals))
    ax1.scatter(i + jitter, vals, s=18, color=colors[label],
                edgecolor="black", linewidth=0.3, alpha=0.8, zorder=3)

ax1.set_xticks(range(len(method_labels)))
ax1.set_xticklabels(method_labels, rotation=35, ha="right")
ax1.set_ylabel("Per-protein CI")
ax1.set_title("A) Overall — Novel GLASS2 proteins")
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax1.set_ylim(0.15, 1.05)

# Add mean annotations
for i, label in enumerate(method_labels):
    vals = list(pp[label].values())
    mean_ci = np.mean(vals)
    ax1.text(i, 1.02, f"{mean_ci:.3f}", ha="center", va="bottom",
             fontsize=7, fontweight="bold", color=colors[label])

# ── Panel B: Quartile grouped bar ───────────────────────────────

n_methods = len(method_labels)
n_quartiles = len(q_labels)
bar_width = 0.14
x_base = np.arange(n_quartiles)

for mi, label in enumerate(method_labels):
    means = []
    sems = []
    for ql in q_labels:
        # Get proteins in this quartile
        prots_in_q = [uid for uid, q in prot_q.items() if q == ql and uid in pp[label]]
        vals = [pp[label][uid] for uid in prots_in_q]
        if vals:
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        else:
            means.append(0)
            sems.append(0)

    offset = (mi - n_methods / 2 + 0.5) * bar_width
    bars = ax2.bar(x_base + offset, means, bar_width * 0.9,
                   yerr=sems, capsize=2,
                   color=colors[label], edgecolor="black", linewidth=0.3,
                   alpha=0.8, label=label, error_kw={"linewidth": 0.8})

    # Value labels on bars
    for xi, (m, s) in enumerate(zip(means, sems)):
        if m > 0:
            ax2.text(x_base[xi] + offset, m + s + 0.015, f"{m:.2f}",
                     ha="center", va="bottom", fontsize=6, rotation=0)

ax2.set_xticks(x_base)
ax2.set_xticklabels(q_labels)
ax2.set_xlabel("Anchor pKi quartile")
ax2.set_ylabel("Mean per-protein CI")
ax2.set_title("B) By anchor pKi quartile — Novel GLASS2 proteins")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax2.set_ylim(0, 1.0)
ax2.legend(loc="upper left", framealpha=0.9, edgecolor="gray")

fig.suptitle("ConciseAnchor vs. Retrieval Baselines on Truly Novel Proteins\n"
             "(GLASS2, <30% seq. identity to training, drug-masked)",
             fontsize=11, fontweight="bold", y=1.02)

plt.tight_layout()
out = BASE / "knn_vs_concise_novel_quartile.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out}")
plt.close()
