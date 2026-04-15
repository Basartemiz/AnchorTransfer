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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "plot"))

from generate_benchmark_filter_ci_panels import (  # noqa: E402
    AnchorTransferDTAv2,
    ConPlex,
    DeepDTAModel,
    EsmDTAModel,
    MODEL_COLORS,
    QUARTILES,
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


MODELS = ["V2_oracle", "DeepDTA", "ConPlex", "ESM-DTA"]
PROTOCOLS = ["filtered", "unfiltered"]


def compute_per_protein(df):
    rows = []
    for uid, group in df.groupby("uniprot_id"):
        rows.append(
            {
                "uniprot_id": uid,
                "n": int(len(group)),
                "ci": ci_fn(group["pki"].values, group["pred"].values),
                "rmse": math.sqrt(float(((group["pki"] - group["pred"]) ** 2).mean())),
            }
        )
    return pd.DataFrame(rows)


def draw_metric_plot(protocol_df, protocol, metric, out_path):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    group_width = 0.78
    box_width = group_width / len(MODELS)
    positions = np.arange(len(QUARTILES)) * 1.6
    legend_handles = []

    for model_idx, model_name in enumerate(MODELS):
        model_positions = positions - group_width / 2 + box_width / 2 + model_idx * box_width
        box_data = []
        for anchor_bin in QUARTILES:
            vals = protocol_df[
                (protocol_df["model"] == model_name)
                & (protocol_df["anchor_bin"] == anchor_bin)
            ][metric].dropna().tolist()
            box_data.append(vals if vals else [np.nan])
        bp = ax.boxplot(
            box_data,
            positions=model_positions,
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

    ax.set_xticks(positions)
    ax.set_xticklabels(QUARTILES)
    ax.set_xlabel("Anchor Bin (Equal Number of Pairs)")
    if metric == "ci":
        ax.set_ylabel("Per-Protein CI")
        ax.set_ylim(0.0, 1.0)
        title_metric = "CI"
    else:
        ax.set_ylabel("Per-Protein RMSE")
        title_metric = "RMSE"
    ax.set_title(
        f"GLASS ({protocol.title()}): Per-Protein {title_metric} by Equal-Frequency Anchor Bins",
        fontsize=16,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
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

    per_protein_frames = []
    summary_frames = []
    pair_bin_frames = []

    for protocol in PROTOCOLS:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        drug_counts = sdf.groupby("ligand_smiles").size()
        keep_drugs = set(drug_counts[drug_counts > 10].index.astype(str))
        sdf = sdf[sdf["ligand_smiles"].astype(str).isin(keep_drugs)].copy()

        strongest_uid, second_uid, weakest_uid, all_uids = None, None, None, None
        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        if anchor_df.empty:
            continue
        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()
        anchor_df["anchor_bin"] = pd.qcut(
            anchor_df["anchor_pki"],
            4,
            labels=QUARTILES,
            duplicates="drop",
        )
        anchor_df = anchor_df.dropna(subset=["anchor_bin"]).copy()

        pair_bin_summary = (
            anchor_df.groupby("anchor_bin", as_index=False)
            .agg(
                n_pairs=("ligand_smiles", "size"),
                n_drugs=("ligand_smiles", "nunique"),
                n_proteins=("uniprot_id", "nunique"),
                anchor_pki_min=("anchor_pki", "min"),
                anchor_pki_max=("anchor_pki", "max"),
                anchor_pki_mean=("anchor_pki", "mean"),
                anchor_pki_median=("anchor_pki", "median"),
            )
        )
        pair_bin_summary["protocol"] = protocol
        pair_bin_frames.append(pair_bin_summary)

        def oracle(uid, smi):
            anc = strongest_uid.get(smi)
            if anc == uid:
                anc = second_uid.get(smi)
            return anc

        predictions = {
            "V2_oracle": predict_v2(anchor_df, emb, v2, oracle, device),
            "DeepDTA": predict_deepdta(anchor_df, seqs, deepdta, device),
            "ConPlex": predict_conplex(anchor_df, emb, conplex, device),
            "ESM-DTA": predict_esm_dta(anchor_df, emb, esm_dta, device),
        }

        for model_name, pred_df in predictions.items():
            pred_df = pred_df.copy()
            pred_df["anchor_bin"] = anchor_df.loc[pred_df.index, "anchor_bin"]
            prot_rows = []
            for anchor_bin in QUARTILES:
                subset = pred_df[pred_df["anchor_bin"].astype(str) == anchor_bin]
                prot = compute_per_protein(subset)
                if prot.empty:
                    continue
                prot["anchor_bin"] = anchor_bin
                prot["protocol"] = protocol
                prot["model"] = model_name
                prot_rows.append(prot)
            if not prot_rows:
                continue
            prot_df = pd.concat(prot_rows, ignore_index=True)
            per_protein_frames.append(prot_df)

            summary = (
                prot_df.groupby("anchor_bin", as_index=False)
                .agg(
                    n_proteins=("uniprot_id", "nunique"),
                    mean_n=("n", "mean"),
                    mean_ci=("ci", "mean"),
                    median_ci=("ci", "median"),
                    mean_rmse=("rmse", "mean"),
                    median_rmse=("rmse", "median"),
                )
            )
            summary["protocol"] = protocol
            summary["model"] = model_name
            summary_frames.append(summary)

    per_protein_df = pd.concat(per_protein_frames, ignore_index=True)
    summary_df = pd.concat(summary_frames, ignore_index=True)
    pair_bins_df = pd.concat(pair_bin_frames, ignore_index=True)

    per_protein_df.to_csv(out_dir / "glass_restricted_anchor_bins_per_protein.csv", index=False)
    summary_df.to_csv(out_dir / "glass_restricted_anchor_bins_summary.csv", index=False)
    pair_bins_df.to_csv(out_dir / "glass_restricted_anchor_pair_bins.csv", index=False)

    for protocol in PROTOCOLS:
        protocol_df = per_protein_df[per_protein_df["protocol"] == protocol].copy()
        if protocol_df.empty:
            continue
        draw_metric_plot(
            protocol_df,
            protocol,
            "ci",
            out_dir / f"glass_{protocol}_restricted_anchor_bins_ci_distribution.png",
        )
        draw_metric_plot(
            protocol_df,
            protocol,
            "rmse",
            out_dir / f"glass_{protocol}_restricted_anchor_bins_rmse_distribution.png",
        )

    print(summary_df.to_string(index=False))
    print("\nPAIR BINS")
    print(pair_bins_df.to_string(index=False))


if __name__ == "__main__":
    main()
