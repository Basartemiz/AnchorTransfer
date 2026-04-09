"""Plot DrugBAN paper replication results."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def plot_bar_comparison(df, output_dir):
    """Bar chart comparing DrugBAN vs AnchorDrugBAN vs Oracle across datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    datasets_splits = [
        ("bindingdb", "random", "BindingDB (Random)"),
        ("biosnap", "random", "BioSNAP (Random)"),
        ("human", "random", "Human (Random)"),
    ]

    models_show = ["drugban", "anchor_drugban", "anchor_drugban_oracle"]
    labels = ["DrugBAN", "AnchorDrugBAN", "Oracle Anchor"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (ds, sp, title) in zip(axes, datasets_splits):
        grp = df[(df["dataset"] == ds) & (df["split"] == sp)]
        means, stds = [], []
        valid_labels, valid_colors = [], []
        for md, lbl, col in zip(models_show, labels, colors):
            mgrp = grp[grp["model"] == md]
            if len(mgrp) == 0:
                continue
            means.append(mgrp["auroc"].mean())
            stds.append(mgrp["auroc"].std())
            valid_labels.append(lbl)
            valid_colors.append(col)

        x = np.arange(len(valid_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=valid_colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim(0.8, 1.0) if sp == "random" else ax.set_ylim(0.4, 0.8)

        # Value labels
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.003,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("In-Domain Performance: DrugBAN vs AnchorDrugBAN", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "indomain_comparison.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'indomain_comparison.png'}")
    plt.close()


def plot_cross_domain(df, output_dir):
    """Bar chart for cross-domain cluster splits."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    datasets = [
        ("bindingdb", "BindingDB (Cluster)"),
        ("biosnap", "BioSNAP (Cluster)"),
    ]

    models_show = ["drugban", "anchor_drugban", "anchor_drugban_oracle"]
    labels = ["DrugBAN", "AnchorDrugBAN", "Oracle Anchor"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for ax, (ds, title) in zip(axes, datasets):
        grp = df[(df["dataset"] == ds) & (df["split"] == "cluster")]
        means, stds = [], []
        valid_labels, valid_colors = [], []
        for md, lbl, col in zip(models_show, labels, colors):
            mgrp = grp[grp["model"] == md]
            if len(mgrp) == 0:
                continue
            means.append(mgrp["auroc"].mean())
            stds.append(mgrp["auroc"].std())
            valid_labels.append(lbl)
            valid_colors.append(col)

        x = np.arange(len(valid_labels))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=valid_colors, width=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(valid_labels, rotation=15, ha="right", fontsize=10)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.set_ylim(0.3, 0.85)

        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Cross-Domain Performance: DrugBAN vs AnchorDrugBAN", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "crossdomain_comparison.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'crossdomain_comparison.png'}")
    plt.close()


def plot_oracle_gap(df, output_dir):
    """Show the gap between Tanimoto anchor, oracle, and DrugBAN."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs = [
        ("bindingdb", "random", "BDB Random"),
        ("biosnap", "random", "BSNAP Random"),
        ("bindingdb", "cluster", "BDB Cluster"),
        ("biosnap", "cluster", "BSNAP Cluster"),
    ]

    models = ["drugban", "anchor_drugban", "anchor_drugban_oracle"]
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    labels = ["DrugBAN", "Anchor (Tanimoto)", "Anchor (Oracle)"]

    x = np.arange(len(configs))
    width = 0.25

    for i, (md, col, lbl) in enumerate(zip(models, colors, labels)):
        means, stds = [], []
        for ds, sp, _ in configs:
            grp = df[(df["dataset"] == ds) & (df["split"] == sp) & (df["model"] == md)]
            if len(grp) > 0:
                means.append(grp["auroc"].mean())
                stds.append(grp["auroc"].std())
            else:
                means.append(0)
                stds.append(0)
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4, color=col, label=lbl)

    ax.set_xticks(x + width)
    ax.set_xticklabels([c[2] for c in configs])
    ax.set_ylabel("AUROC")
    ax.set_title("Oracle Gap Analysis: Tanimoto vs Perfect Anchor Retrieval")
    ax.legend()
    ax.set_ylim(0.3, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "oracle_gap_analysis.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'oracle_gap_analysis.png'}")
    plt.close()


def plot_per_seed(df, output_dir):
    """Per-seed scatter for each dataset/split."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    configs = [
        ("bindingdb", "random"), ("biosnap", "random"), ("human", "random"),
        ("bindingdb", "cluster"), ("biosnap", "cluster"), ("human", "cold"),
    ]

    for ax, (ds, sp) in zip(axes.flat, configs):
        grp = df[(df["dataset"] == ds) & (df["split"] == sp)]
        for md, color, marker in [
            ("drugban", "#4C72B0", "o"),
            ("anchor_drugban", "#DD8452", "s"),
            ("anchor_drugban_oracle", "#55A868", "^"),
        ]:
            mgrp = grp[grp["model"] == md]
            if len(mgrp) == 0:
                continue
            ax.scatter(mgrp["seed"], mgrp["auroc"], c=color, marker=marker,
                       s=60, label=md.replace("_", " "), alpha=0.8)
            ax.axhline(y=mgrp["auroc"].mean(), color=color, linestyle="--", alpha=0.4)

        ax.set_title(f"{ds} / {sp}")
        ax.set_xlabel("Seed")
        ax.set_ylabel("AUROC")
        ax.legend(fontsize=8)

    fig.suptitle("Per-Seed Results Across All Benchmarks", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "per_seed_scatter.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'per_seed_scatter.png'}")
    plt.close()


if __name__ == "__main__":
    results_path = Path("results/drugban_paper_all_results.csv")
    if not results_path.exists():
        # Try merging
        dfs = []
        for p in Path("results").glob("drugban_paper*.csv"):
            try:
                d = pd.read_csv(p)
                if "error" in d.columns:
                    d = d[d["error"].isna()].drop(columns=["error"])
                dfs.append(d)
            except Exception:
                pass
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df[df["auroc"].notna()]
            df.to_csv(results_path, index=False)
        else:
            print("No results found")
            sys.exit(1)
    else:
        df = pd.read_csv(results_path)

    output_dir = Path("results/drugban_paper_plots")
    output_dir.mkdir(exist_ok=True)

    plot_bar_comparison(df, output_dir)
    plot_cross_domain(df, output_dir)
    plot_oracle_gap(df, output_dir)
    plot_per_seed(df, output_dir)
    print(f"\nAll plots saved to {output_dir}/")
