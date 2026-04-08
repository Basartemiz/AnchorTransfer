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
    MODELS,
    MODEL_COLORS,
    QUARTILES,
    AnchorTransferDTAv2,
    ConPlex,
    DeepDTAModel,
    EsmDTAModel,
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    load_benchmarks,
    load_embeddings,
    load_model,
    predict_conplex,
    predict_deepdta,
    predict_esm_dta,
    predict_v2,
)


def draw_quartile_rmse_distribution(protocol, quartile_df, out_path):
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    group_width = 0.78
    box_width = group_width / len(MODELS)
    quartile_positions = np.arange(len(QUARTILES)) * 1.6
    legend_handles = []

    for model_idx, model_name in enumerate(MODELS):
        positions = quartile_positions - group_width / 2 + box_width / 2 + model_idx * box_width
        box_data = []
        for quartile in QUARTILES:
            vals = quartile_df[
                (quartile_df["model"] == model_name)
                & (quartile_df["quartile"] == quartile)
            ]["rmse"].dropna().tolist()
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

    ax.set_xticks(quartile_positions)
    ax.set_xticklabels(QUARTILES)
    ax.set_ylabel("Per-Protein RMSE")
    ax.set_xlabel("Anchor Quartile")
    ax.set_title(
        f"GLASS ({protocol.title()}): Per-Protein RMSE by Anchor Quartile",
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

    for protocol in ["filtered", "unfiltered"]:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()
        anchor_df["anchor_quartile"] = pd.qcut(
            anchor_df["anchor_pki"], 4, labels=QUARTILES, duplicates="drop"
        )

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

        rows = []
        for model_name, pred_df in predictions.items():
            pred_df = pred_df.copy()
            pred_df["anchor_quartile"] = anchor_df.loc[pred_df.index, "anchor_quartile"]
            for quartile in QUARTILES:
                subset = pred_df[pred_df["anchor_quartile"].astype(str) == quartile]
                for uid, group in subset.groupby("uniprot_id"):
                    rmse = math.sqrt(float(((group["pki"] - group["pred"]) ** 2).mean()))
                    rows.append(
                        {
                            "model": model_name,
                            "quartile": quartile,
                            "uniprot_id": uid,
                            "n": int(len(group)),
                            "rmse": rmse,
                            "protocol": protocol,
                        }
                    )

        rmse_df = pd.DataFrame(rows)
        rmse_df.to_csv(out_dir / f"glass_{protocol}_quartile_per_protein_rmse.csv", index=False)
        draw_quartile_rmse_distribution(
            protocol,
            rmse_df,
            out_dir / f"glass_{protocol}_quartile_rmse_distribution.png",
        )


if __name__ == "__main__":
    main()
