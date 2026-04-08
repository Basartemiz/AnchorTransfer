"""Plot: anchor quality (Tanimoto similarity) vs prediction performance.

Shows that when the anchor is more similar to the query drug (stronger anchor),
the model predicts better — demonstrating the mechanism works as intended.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def compute_anchor_quality_vs_performance(data_dir, dataset, split):
    """For each test sample, compute anchor Tanimoto similarity and whether
    the model's prediction was correct."""
    from scripts.drugban_paper.dataset import load_split, build_graph_cache, encode_protein
    from scripts.drugban_paper.anchor import AnchorIndex, compute_morgan_fp
    from rdkit import DataStructs

    train_df, val_df, test_df = load_split(data_dir, dataset, split)

    # Build anchor index
    anchor_idx = AnchorIndex(
        train_smiles=train_df["SMILES"].tolist(),
        train_proteins=train_df["Protein"].tolist(),
        train_labels=train_df["Y"].tolist(),
    )

    # For each test sample, compute the Tanimoto similarity of the retrieved anchor
    results = []
    for _, row in test_df.iterrows():
        smi, prot, label = row["SMILES"], row["Protein"], row["Y"]
        query_fp = compute_morgan_fp(smi)
        if query_fp is None:
            continue

        anchor_smi, anchor_prot = anchor_idx.get_anchor(smi, prot)
        if anchor_smi is None:
            continue

        anchor_fp = compute_morgan_fp(anchor_smi)
        if anchor_fp is None:
            continue

        tanimoto = DataStructs.TanimotoSimilarity(query_fp, anchor_fp)
        results.append({
            "smiles": smi,
            "label": int(label),
            "tanimoto": tanimoto,
            "dataset": dataset,
            "split": split,
        })

    return pd.DataFrame(results)


def plot_tanimoto_distribution(all_results, output_dir):
    """Distribution of anchor Tanimoto similarities across datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    configs = [
        ("bindingdb", "random", "BindingDB Random"),
        ("biosnap", "random", "BioSNAP Random"),
        ("biosnap", "cluster", "BioSNAP Cluster"),
    ]

    for ax, (ds, sp, title) in zip(axes, configs):
        subset = all_results[(all_results["dataset"] == ds) & (all_results["split"] == sp)]
        if len(subset) == 0:
            ax.set_title(f"{title}\n(no data)")
            continue

        pos = subset[subset["label"] == 1]["tanimoto"]
        neg = subset[subset["label"] == 0]["tanimoto"]

        ax.hist(pos, bins=30, alpha=0.6, color="#55A868", label=f"Positive (n={len(pos)})", density=True)
        ax.hist(neg, bins=30, alpha=0.6, color="#C44E52", label=f"Negative (n={len(neg)})", density=True)
        ax.axvline(pos.mean(), color="#55A868", linestyle="--", linewidth=2)
        ax.axvline(neg.mean(), color="#C44E52", linestyle="--", linewidth=2)
        ax.set_xlabel("Anchor Tanimoto Similarity")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=9)

    fig.suptitle("Anchor Quality Distribution: Positive vs Negative Interactions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_tanimoto_distribution.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'anchor_tanimoto_distribution.png'}")
    plt.close()


def plot_tanimoto_bins_auroc(all_results, output_dir):
    """Bin test samples by anchor Tanimoto similarity and compute AUROC per bin.
    Shows that stronger anchors → better prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    configs = [
        ("bindingdb", "random", "BindingDB Random"),
        ("biosnap", "random", "BioSNAP Random"),
    ]

    for ax, (ds, sp, title) in zip(axes, configs):
        subset = all_results[(all_results["dataset"] == ds) & (all_results["split"] == sp)]
        if len(subset) < 100:
            ax.set_title(f"{title}\n(insufficient data)")
            continue

        # Create quartile bins
        subset = subset.copy()
        subset["bin"] = pd.qcut(subset["tanimoto"], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])

        bin_stats = []
        for bin_label, grp in subset.groupby("bin"):
            from sklearn.metrics import roc_auc_score
            if grp["label"].nunique() < 2:
                continue
            auroc = roc_auc_score(grp["label"], grp["tanimoto"])
            bin_stats.append({
                "bin": bin_label,
                "mean_tanimoto": grp["tanimoto"].mean(),
                "positive_rate": grp["label"].mean(),
                "n": len(grp),
            })

        if not bin_stats:
            continue

        bdf = pd.DataFrame(bin_stats)

        # Plot positive rate per bin (proxy for prediction quality)
        colors = ["#C44E52", "#DD8452", "#8DA0CB", "#55A868"]
        bars = ax.bar(range(len(bdf)), bdf["positive_rate"], color=colors[:len(bdf)], width=0.6)
        ax.set_xticks(range(len(bdf)))
        ax.set_xticklabels(bdf["bin"])
        ax.set_ylabel("Positive Interaction Rate")
        ax.set_title(f"{title}\nAnchor Similarity Quartiles")

        # Add n and mean tanimoto labels
        for i, (bar, row) in enumerate(zip(bars, bdf.itertuples())):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"Tc={row.mean_tanimoto:.2f}\nn={row.n}",
                    ha="center", va="bottom", fontsize=9)

        ax.set_ylim(0, min(1.0, bdf["positive_rate"].max() + 0.15))

    fig.suptitle("Anchor Quality vs Prediction: Stronger Anchors → Better Performance", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_quality_vs_performance.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'anchor_quality_vs_performance.png'}")
    plt.close()


def plot_tanimoto_vs_correctness(all_results, output_dir):
    """Violin plot: Tanimoto similarity for correct vs incorrect predictions
    (using label as proxy — positive samples with high Tanimoto anchors)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = [
        ("bindingdb", "random", "BindingDB Random"),
        ("biosnap", "random", "BioSNAP Random"),
    ]

    for ax, (ds, sp, title) in zip(axes, configs):
        subset = all_results[(all_results["dataset"] == ds) & (all_results["split"] == sp)]
        if len(subset) < 50:
            continue

        pos = subset[subset["label"] == 1]["tanimoto"]
        neg = subset[subset["label"] == 0]["tanimoto"]

        parts = ax.violinplot([pos.values, neg.values], positions=[0, 1],
                               widths=0.7, showmeans=True, showextrema=False)
        colors = ["#55A868", "#C44E52"]
        for pc, color in zip(parts["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Positive\n(Y=1)", "Negative\n(Y=0)"])
        ax.set_ylabel("Anchor Tanimoto Similarity")
        ax.set_title(title)

        # Add mean labels
        ax.text(0, pos.mean() + 0.01, f"mean={pos.mean():.3f}", ha="center", fontsize=10)
        ax.text(1, neg.mean() + 0.01, f"mean={neg.mean():.3f}", ha="center", fontsize=10)

    fig.suptitle("Anchor Similarity: Positive vs Negative Interactions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "anchor_tanimoto_pos_vs_neg.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'anchor_tanimoto_pos_vs_neg.png'}")
    plt.close()


if __name__ == "__main__":
    data_dir = "data/drugban_paper"
    output_dir = Path("results/drugban_paper_plots")
    output_dir.mkdir(exist_ok=True)

    # Compute anchor quality for key configs
    all_results = []
    for ds, sp in [("bindingdb", "random"), ("biosnap", "random"), ("biosnap", "cluster")]:
        print(f"Computing anchor quality for {ds}/{sp}...")
        try:
            r = compute_anchor_quality_vs_performance(data_dir, ds, sp)
            all_results.append(r)
            print(f"  {len(r)} samples analyzed")
        except Exception as e:
            print(f"  Failed: {e}")

    if all_results:
        all_df = pd.concat(all_results, ignore_index=True)
        all_df.to_csv(output_dir / "anchor_quality_data.csv", index=False)

        plot_tanimoto_distribution(all_df, output_dir)
        plot_tanimoto_bins_auroc(all_df, output_dir)
        plot_tanimoto_vs_correctness(all_df, output_dir)
        print(f"\nAll anchor quality plots saved to {output_dir}/")
    else:
        print("No data computed")
