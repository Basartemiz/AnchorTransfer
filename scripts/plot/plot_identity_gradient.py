"""Sequence identity gradient: ConciseAnchor vs kNN baselines on GLASS2.

Shows how per-protein CI changes as max sequence identity to training decreases.
ConciseAnchor maintains performance where kNN baselines collapse.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.fontsize": 9, "figure.dpi": 300,
    "savefig.dpi": 300, "savefig.bbox": "tight",
})

df = pd.read_csv("results/seq_identity_gradient_full.csv")

# Define bins
bins_def = [
    (0.40, 0.60, "40-60%"),
    (0.60, 0.70, "60-70%"),
    (0.70, 0.80, "70-80%"),
    (0.80, 0.90, "80-90%"),
    (0.90, 1.01, "90-100%"),
]

methods = {
    "ConciseAnchor": {"col": "ConciseAnchor", "color": "#2E86AB", "marker": "o"},
    "Prot-kNN (k=1)": {"col": "Prot_kNN_k1", "color": "#E8998D", "marker": "s"},
    "Prot-kNN (k=5)": {"col": "Prot_kNN_k5", "color": "#D4A373", "marker": "D"},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                gridspec_kw={"width_ratios": [2, 1]})

# ── Panel A: Line plot with CI vs identity bin ──

x_pos = np.arange(len(bins_def))
x_labels = [b[2] for b in bins_def]

for label, cfg in methods.items():
    means, sems, ns = [], [], []
    for lo, hi, _ in bins_def:
        mask = (df.max_seq_id >= lo) & (df.max_seq_id < hi)
        vals = df.loc[mask, cfg["col"]].dropna().values
        if len(vals) > 0:
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            ns.append(len(vals))
        else:
            means.append(np.nan)
            sems.append(0)
            ns.append(0)

    means = np.array(means, dtype=float)
    sems = np.array(sems, dtype=float)

    # Plot line with error bars
    valid = ~np.isnan(means)
    ax1.errorbar(x_pos[valid], means[valid], yerr=sems[valid],
                 marker=cfg["marker"], markersize=8, linewidth=2.5,
                 color=cfg["color"], label=label, capsize=4, capthick=1.5,
                 zorder=3)

    # Annotate with n
    for i in range(len(means)):
        if not np.isnan(means[i]) and ns[i] > 0:
            ax1.annotate(f"n={ns[i]}", (x_pos[i], means[i]),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=7, ha="center", color=cfg["color"])

ax1.set_xticks(x_pos)
ax1.set_xticklabels(x_labels)
ax1.set_xlabel("Max sequence identity to BDB training proteins")
ax1.set_ylabel("Mean per-protein CI")
ax1.set_title("A) Performance vs. Sequence Identity to Training")
ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="Random (CI=0.5)")
ax1.set_ylim(0.35, 0.85)
ax1.legend(loc="upper left", framealpha=0.9)

# Add arrow annotation for the key finding
ax1.annotate("ConciseAnchor\nmaintains CI",
            xy=(1, 0.71), xytext=(0.3, 0.80),
            fontsize=8, fontweight="bold", color="#2E86AB",
            arrowprops=dict(arrowstyle="->", color="#2E86AB", lw=1.5),
            ha="center")
ax1.annotate("kNN collapses",
            xy=(3, 0.48), xytext=(3.5, 0.40),
            fontsize=8, fontweight="bold", color="#E8998D",
            arrowprops=dict(arrowstyle="->", color="#E8998D", lw=1.5),
            ha="center")

# ── Panel B: Scatter plot — per-protein CI comparison ──

# Only proteins with both ConciseAnchor and kNN predictions
both = df.dropna(subset=["ConciseAnchor", "Prot_kNN_k1"])
if len(both) > 0:
    colors = both.max_seq_id.values
    sc = ax2.scatter(both.Prot_kNN_k1, both.ConciseAnchor,
                     c=colors, cmap="RdYlBu_r", s=50, edgecolor="black",
                     linewidth=0.5, alpha=0.8, vmin=0.4, vmax=1.0)
    cb = plt.colorbar(sc, ax=ax2, shrink=0.8)
    cb.set_label("Max seq. identity to BDB", fontsize=8)

    # Diagonal
    ax2.plot([0.3, 1], [0.3, 1], "k--", linewidth=0.8, alpha=0.5)
    ax2.fill_between([0.3, 1], [0.3, 1], [1, 1], alpha=0.05, color="blue")
    ax2.fill_between([0.3, 1], [0.3, 1], [0.3, 0.3], alpha=0.05, color="red")

    # Count wins
    ca_wins = (both.ConciseAnchor > both.Prot_kNN_k1).sum()
    knn_wins = (both.Prot_kNN_k1 > both.ConciseAnchor).sum()
    ax2.text(0.35, 0.92, f"ConciseAnchor\nwins: {ca_wins}/{len(both)}",
             fontsize=8, color="#2E86AB", fontweight="bold")
    ax2.text(0.7, 0.35, f"kNN wins:\n{knn_wins}/{len(both)}",
             fontsize=8, color="#E8998D", fontweight="bold")

ax2.set_xlabel("Prot-kNN (k=1) CI")
ax2.set_ylabel("ConciseAnchor CI")
ax2.set_title("B) Per-protein CI comparison")
ax2.set_xlim(0.3, 1.0)
ax2.set_ylim(0.3, 1.0)
ax2.set_aspect("equal")

fig.suptitle("Sequence Identity Gradient: GLASS2 Cross-Dataset Evaluation\n"
             "(BDB retrieval pool, proteins colored by max identity to training)",
             fontsize=12, fontweight="bold", y=1.03)

plt.tight_layout()
out = "results/seq_identity_gradient.pdf"
fig.savefig(out, bbox_inches="tight")
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
print(f"Saved {out}")
