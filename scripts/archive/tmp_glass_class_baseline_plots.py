import json
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
MIN_CLASS_PROTEINS = 10


def normalize_gpcr_class(value):
    if value is None:
        return "Unknown"
    if isinstance(value, list):
        if not value:
            return "Unknown"
        value = value[0]
    value = str(value).strip()
    return value if value else "Unknown"


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


def draw_metric_panels(df, metric, out_path):
    major_classes = (
        df[df["model"] == "V2_oracle"]
        .groupby(["protocol", "gpcr_class"])["uniprot_id"]
        .nunique()
        .reset_index(name="n_proteins")
    )
    major_classes = major_classes[major_classes["n_proteins"] >= MIN_CLASS_PROTEINS]
    protocols = []
    for protocol in PROTOCOLS:
        classes = major_classes[major_classes["protocol"] == protocol]["gpcr_class"].tolist()
        if classes:
            protocols.append((protocol, classes))
    if not protocols:
        return

    fig, axes = plt.subplots(1, len(protocols), figsize=(6.4 * len(protocols), 5.8), squeeze=False)
    axes = axes[0]
    legend_handles = [Patch(facecolor=MODEL_COLORS[m], edgecolor="#333333", label=m) for m in MODELS]

    for ax, (protocol, classes) in zip(axes, protocols):
        positions = np.arange(len(classes)) * 1.6
        group_width = 0.78
        box_width = group_width / len(MODELS)
        for model_idx, model_name in enumerate(MODELS):
            model_positions = positions - group_width / 2 + box_width / 2 + model_idx * box_width
            box_data = []
            for gpcr_class in classes:
                vals = df[
                    (df["protocol"] == protocol)
                    & (df["model"] == model_name)
                    & (df["gpcr_class"] == gpcr_class)
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

        ax.set_xticks(positions)
        ax.set_xticklabels(classes)
        ax.set_xlabel("GPCR Class")
        ax.set_title(protocol.title(), fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)
        if metric == "ci":
            ax.set_ylabel("Per-Protein CI")
            ax.set_ylim(0.0, 1.0)
        else:
            ax.set_ylabel("Per-Protein RMSE")

    fig.suptitle(
        "GLASS: Per-Protein {} by GPCR Class (>10 drug interactions, oracle anchor)".format(
            "CI" if metric == "ci" else "RMSE"
        ),
        fontsize=16,
        fontweight="bold",
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
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
    protein_meta = json.load(open("data/raw/glass/protein.json"))
    meta_df = pd.DataFrame(
        [
            {
                "uniprot_id": str(uid),
                "gpcr_class": normalize_gpcr_class(info.get("gpcr_class")) if isinstance(info, dict) else "Unknown",
            }
            for uid, info in protein_meta.items()
        ]
    )

    v2 = load_model(AnchorTransferDTAv2(esm2_dim=480).to(device), "models/v2_dtc/best_model.pt", device)
    deepdta = load_model(DeepDTAModel().to(device), "models/deepdta_dtc/best_model.pt", device)
    conplex = load_model(ConPlex(esm2_dim=480).to(device), "models/conplex_dtc/best_model.pt", device)
    esm_dta = load_model(EsmDTAModel(esm2_dim=480).to(device), "models/esm_dta_dtc/best_model.pt", device)

    out_dir = Path("results/benchmark_filter_ci_panels")
    out_dir.mkdir(parents=True, exist_ok=True)
    per_protein_frames = []

    for protocol in PROTOCOLS:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        drug_counts = sdf.groupby("ligand_smiles").size()
        keep_drugs = set(drug_counts[drug_counts > 10].index.astype(str))
        sdf = sdf[sdf["ligand_smiles"].astype(str).isin(keep_drugs)].copy()

        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        if anchor_df.empty:
            continue
        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()

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
            prot = compute_per_protein(pred_df)
            prot["protocol"] = protocol
            prot["model"] = model_name
            prot = prot.merge(meta_df, on="uniprot_id", how="left")
            prot["gpcr_class"] = prot["gpcr_class"].fillna("Unknown")
            per_protein_frames.append(prot)

    per_protein = pd.concat(per_protein_frames, ignore_index=True)
    class_summary = (
        per_protein.groupby(["protocol", "model", "gpcr_class"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_n=("n", "mean"),
            mean_ci=("ci", "mean"),
            median_ci=("ci", "median"),
            mean_rmse=("rmse", "mean"),
            median_rmse=("rmse", "median"),
        )
        .sort_values(["protocol", "gpcr_class", "model"])
    )

    per_protein.to_csv(out_dir / "glass_class_baseline_per_protein.csv", index=False)
    class_summary.to_csv(out_dir / "glass_class_baseline_summary.csv", index=False)

    draw_metric_panels(
        per_protein,
        "ci",
        out_dir / "glass_class_baseline_ci_distribution.png",
    )
    draw_metric_panels(
        per_protein,
        "rmse",
        out_dir / "glass_class_baseline_rmse_distribution.png",
    )
    print(class_summary.to_string(index=False))


if __name__ == "__main__":
    main()
