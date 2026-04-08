"""Violin plots for DrugBAN paper replication — per-seed distribution."""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.size": 12, "figure.dpi": 150})


def plot_violin_all(df, output_dir):
    """Violin plots showing per-seed AUROC distribution for each config."""

    configs = [
        ("bindingdb", "random", "BindingDB\nRandom"),
        ("biosnap", "random", "BioSNAP\nRandom"),
        ("human", "random", "Human\nRandom"),
        ("human", "cold", "Human\nCold"),
        ("bindingdb", "cluster", "BindingDB\nCluster"),
        ("biosnap", "cluster", "BioSNAP\nCluster"),
    ]

    models = [
        ("drugban", "DrugBAN", "#4C72B0"),
        ("anchor_drugban", "AnchorDrugBAN", "#DD8452"),
        ("anchor_drugban_oracle", "Oracle", "#55A868"),
    ]

    fig, ax = plt.subplots(figsize=(16, 7))

    positions = []
    tick_positions = []
    tick_labels = []
    group_width = len(models) + 1.5  # space between groups

    for gi, (ds, sp, label) in enumerate(configs):
        grp = df[(df["dataset"] == ds) & (df["split"] == sp)]
        group_center = gi * group_width

        for mi, (md, md_label, color) in enumerate(models):
            mgrp = grp[grp["model"] == md]
            pos = group_center + mi * 0.8
            positions.append(pos)

            if len(mgrp) >= 2:
                parts = ax.violinplot(
                    mgrp["auroc"].values, positions=[pos],
                    widths=0.6, showmeans=True, showextrema=False,
                )
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.6)
                parts["cmeans"].set_color(color)

                # Overlay individual points
                jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(mgrp))
                ax.scatter(
                    [pos] * len(mgrp) + jitter[:len(mgrp)],
                    mgrp["auroc"].values,
                    c=color, s=30, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.5,
                )

                # Mean label
                mean_val = mgrp["auroc"].mean()
                ax.text(pos, mean_val + 0.008, f"{mean_val:.3f}",
                        ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

            elif len(mgrp) == 1:
                ax.scatter([pos], mgrp["auroc"].values, c=color, s=60, zorder=3,
                           edgecolors="white", linewidths=0.5)
                ax.text(pos, mgrp["auroc"].values[0] + 0.008, f"{mgrp['auroc'].values[0]:.3f}",
                        ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

        tick_positions.append(group_center + 0.8)
        tick_labels.append(label)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel("AUROC")
    ax.set_title("Per-Seed AUROC Distribution: DrugBAN vs AnchorDrugBAN vs Oracle")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, alpha=0.6, label=l) for _, l, c in models]
    ax.legend(handles=legend_elements, loc="lower left")

    fig.tight_layout()
    fig.savefig(output_dir / "violin_all_configs.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'violin_all_configs.png'}")
    plt.close()


def plot_violin_indomain(df, output_dir):
    """Violin plot focused on in-domain random splits with all 5 models."""

    configs = [
        ("bindingdb", "random", "BindingDB Random"),
        ("biosnap", "random", "BioSNAP Random"),
        ("human", "random", "Human Random"),
    ]

    models = [
        ("drugban", "DrugBAN", "#4C72B0"),
        ("drugban_anchor_subset", "DrugBAN\n(anchor sub)", "#8DA0CB"),
        ("anchor_drugban", "AnchorDrugBAN", "#DD8452"),
        ("drugban_oracle_subset", "DrugBAN\n(oracle sub)", "#A1D99B"),
        ("anchor_drugban_oracle", "Oracle", "#55A868"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, (ds, sp, title) in zip(axes, configs):
        grp = df[(df["dataset"] == ds) & (df["split"] == sp)]

        data_to_plot = []
        colors_plot = []
        labels_plot = []

        for md, md_label, color in models:
            mgrp = grp[grp["model"] == md]
            if len(mgrp) < 2:
                continue
            data_to_plot.append(mgrp["auroc"].values)
            colors_plot.append(color)
            labels_plot.append(md_label)

        if not data_to_plot:
            ax.set_title(title)
            continue

        parts = ax.violinplot(
            data_to_plot, positions=range(len(data_to_plot)),
            widths=0.7, showmeans=True, showextrema=False,
        )
        for pc, color in zip(parts["bodies"], colors_plot):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
        parts["cmeans"].set_color("black")

        # Scatter individual seeds
        for i, (vals, color) in enumerate(zip(data_to_plot, colors_plot)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
            ax.scatter([i] * len(vals) + jitter[:len(vals)], vals,
                       c=color, s=35, alpha=0.8, zorder=3, edgecolors="white", linewidths=0.5)
            ax.text(i, np.mean(vals) + 0.003, f"{np.mean(vals):.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

        ax.set_xticks(range(len(labels_plot)))
        ax.set_xticklabels(labels_plot, fontsize=9)
        ax.set_ylabel("AUROC")
        ax.set_title(title)

    fig.suptitle("In-Domain: Full Model Comparison (5 seeds each)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "violin_indomain_full.png", bbox_inches="tight")
    print(f"Saved {output_dir / 'violin_indomain_full.png'}")
    plt.close()


if __name__ == "__main__":
    results_path = Path("results/drugban_paper_all_results.csv")
    if not results_path.exists():
        dfs = []
        for p in Path("results").glob("drugban_paper*.csv"):
            try:
                d = pd.read_csv(p)
                if "error" in d.columns:
                    d = d[d["error"].isna()].drop(columns=["error"])
                dfs.append(d)
            except Exception:
                pass
        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        df = df[df["auroc"].notna()]
    else:
        df = pd.read_csv(results_path)

    output_dir = Path("results/drugban_paper_plots")
    output_dir.mkdir(exist_ok=True)

    plot_violin_all(df, output_dir)
    plot_violin_indomain(df, output_dir)
    print(f"\nViolin plots saved to {output_dir}/")
