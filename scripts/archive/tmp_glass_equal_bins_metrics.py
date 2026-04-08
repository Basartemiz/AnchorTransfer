import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_benchmark_filter_ci_panels import (  # noqa: E402
    AnchorTransferDTAv2,
    ConPlex,
    DeepDTAModel,
    EsmDTAModel,
    MODEL_COLORS,
    MODELS,
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    ci_fn,
    load_benchmarks,
    load_embeddings,
    load_model,
    predict_conplex,
    predict_deepdta,
    predict_esm_dta,
    predict_v2,
)


BIN_LABELS = ["7-9", "9-11", "11-13", "13+"]
BIN_EDGES = [7.0, 9.0, 11.0, 13.0, float("inf")]


def assign_anchor_bin(series):
    bins = pd.cut(series, bins=BIN_EDGES, labels=BIN_LABELS, right=False, include_lowest=True)
    return bins.astype(str).replace("nan", np.nan)


def per_protein_bin_metrics(pred_df):
    rows = []
    for anchor_bin in BIN_LABELS:
        subset = pred_df[pred_df["anchor_bin"] == anchor_bin]
        for uid, group in subset.groupby("uniprot_id"):
            rmse = math.sqrt(float(((group["pki"] - group["pred"]) ** 2).mean()))
            rows.append(
                {
                    "anchor_bin": anchor_bin,
                    "uniprot_id": uid,
                    "n": int(len(group)),
                    "ci": ci_fn(group["pki"].values, group["pred"].values),
                    "rmse": rmse,
                }
            )
    return pd.DataFrame(rows)


def draw_distribution_plot(protocol_df, protocol, metric, out_path):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    group_width = 0.78
    box_width = group_width / len(MODELS)
    bin_positions = np.arange(len(BIN_LABELS)) * 1.6
    legend_handles = []

    for model_idx, model_name in enumerate(MODELS):
        positions = bin_positions - group_width / 2 + box_width / 2 + model_idx * box_width
        box_data = []
        for anchor_bin in BIN_LABELS:
            vals = protocol_df[
                (protocol_df["model"] == model_name)
                & (protocol_df["anchor_bin"] == anchor_bin)
            ][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(
            box_data,
            positions=positions,
            widths=box_width * 0.9,
            patch_artist=True,
            showfliers=False,
            manage_ticks=False,
        )
        color = MODEL_COLORS[model_name]
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
            patch.set_edgecolor("#333333")
        for key in ("whiskers", "caps", "medians"):
            for artist in bp[key]:
                artist.set_color("#333333")
                artist.set_linewidth(1.0)
        legend_handles.append(Patch(facecolor=color, edgecolor="#333333", label=model_name))

    ax.set_xticks(bin_positions)
    ax.set_xticklabels(BIN_LABELS)
    ax.set_xlabel("Anchor pKi Bin")
    if metric == "ci":
        ax.set_ylabel("Per-Protein CI")
        ax.set_ylim(0.0, 1.0)
        title_metric = "CI"
    else:
        ax.set_ylabel("Per-Protein RMSE")
        title_metric = "RMSE"
    ax.set_title(
        f"GLASS ({protocol.title()}): Per-Protein {title_metric} by Equal Anchor pKi Bins",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs = build_sequences()
    emb = load_embeddings()
    dtc_ref = build_dtc_reference(seqs)
    bench = load_benchmarks()["GLASS"]

    v2 = load_model(AnchorTransferDTAv2(esm2_dim=480).to(device), "models/v2_dtc/best_model.pt", device)
    deepdta = load_model(DeepDTAModel().to(device), "models/deepdta_dtc/best_model.pt", device)
    conplex = load_model(ConPlex(esm2_dim=480).to(device), "models/conplex_dtc/best_model.pt", device)
    esm_dta = load_model(EsmDTAModel(esm2_dim=480).to(device), "models/esm_dta_dtc/best_model.pt", device)

    out_dir = Path("results/benchmark_filter_ci_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summary = []
    all_per_protein = []

    for protocol in ["filtered", "unfiltered"]:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        if anchor_df.empty:
            continue

        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()
        anchor_df["anchor_bin"] = assign_anchor_bin(anchor_df["anchor_pki"])
        anchor_df = anchor_df.dropna(subset=["anchor_bin"]).copy()

        def oracle(uid, smi):
            anc = strongest_uid.get(smi)
            if anc == uid:
                anc = second_uid.get(smi)
            return anc

        def weakest(uid, smi):
            anc = weakest_uid.get(smi)
            if anc == uid:
                anc = second_uid.get(smi)
            return anc

        rng = random.Random(seed)

        def rand_anchor(uid, _smi):
            choices = [protein for protein in all_uids if protein != uid]
            return rng.choice(choices) if choices else None

        predictions = {
            "V2_oracle": predict_v2(anchor_df, emb, v2, oracle, device),
            "V2_weakest": predict_v2(anchor_df, emb, v2, weakest, device),
            "V2_random": predict_v2(anchor_df, emb, v2, rand_anchor, device),
            "DeepDTA": predict_deepdta(anchor_df, seqs, deepdta, device),
            "ConPlex": predict_conplex(anchor_df, emb, conplex, device),
            "ESM-DTA": predict_esm_dta(anchor_df, emb, esm_dta, device),
        }

        for model_name, pred_df in predictions.items():
            pred_df = pred_df.copy()
            pred_df["anchor_bin"] = anchor_df.loc[pred_df.index, "anchor_bin"]
            prot_df = per_protein_bin_metrics(pred_df)
            prot_df["benchmark"] = "GLASS"
            prot_df["protocol"] = protocol
            prot_df["model"] = model_name
            all_per_protein.append(prot_df)

            summary = (
                prot_df.groupby("anchor_bin")
                .agg(
                    n_proteins=("uniprot_id", "nunique"),
                    n_groups=("uniprot_id", "size"),
                    mean_n=("n", "mean"),
                    mean_ci=("ci", "mean"),
                    median_ci=("ci", "median"),
                    mean_rmse=("rmse", "mean"),
                    median_rmse=("rmse", "median"),
                )
                .reset_index()
            )
            summary["benchmark"] = "GLASS"
            summary["protocol"] = protocol
            summary["model"] = model_name
            all_summary.append(summary)

    all_summary_df = pd.concat(all_summary, ignore_index=True)
    all_per_protein_df = pd.concat(all_per_protein, ignore_index=True)
    all_summary_df.to_csv(out_dir / "glass_equal_anchor_bins_summary.csv", index=False)
    all_per_protein_df.to_csv(out_dir / "glass_equal_anchor_bins_per_protein.csv", index=False)

    for protocol in ["filtered", "unfiltered"]:
        protocol_df = all_per_protein_df[all_per_protein_df["protocol"] == protocol].copy()
        if protocol_df.empty:
            continue
        draw_distribution_plot(
            protocol_df,
            protocol,
            "ci",
            out_dir / f"glass_{protocol}_equal_anchor_bins_ci_distribution.png",
        )
        draw_distribution_plot(
            protocol_df,
            protocol,
            "rmse",
            out_dir / f"glass_{protocol}_equal_anchor_bins_rmse_distribution.png",
        )
    print(all_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
