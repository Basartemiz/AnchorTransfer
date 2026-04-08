import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_benchmark_filter_ci_panels import (  # noqa: E402
    AnchorTransferDTAv2,
    MIN_ANCHOR_PKI,
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    ci_fn,
    load_benchmarks,
    load_embeddings,
    load_model,
    predict_v2,
)


def normalize_gpcr_class(value):
    if value is None:
        return "Unknown"
    if isinstance(value, list):
        if not value:
            return "Unknown"
        value = value[0]
    value = str(value).strip()
    return value if value else "Unknown"


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

    model = load_model(AnchorTransferDTAv2(esm2_dim=480).to(device), "models/v2_dtc/best_model.pt", device)
    out_dir = Path("results/benchmark_filter_ci_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    protein_rows = []
    class_rows = []

    for protocol in ["filtered", "unfiltered"]:
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

        pred_df = predict_v2(anchor_df, emb, model, oracle, device)
        pred_df = pred_df.copy()

        prot = (
            pred_df.groupby("uniprot_id", as_index=False)
            .apply(
                lambda g: pd.Series(
                    {
                        "n": int(len(g)),
                        "ci": ci_fn(g["pki"].values, g["pred"].values),
                        "rmse": math.sqrt(float(((g["pki"] - g["pred"]) ** 2).mean())),
                    }
                ),
                include_groups=False,
            )
            .reset_index(drop=True)
        )
        prot["protocol"] = protocol
        prot["model"] = "V2_oracle"
        prot["min_anchor_pki"] = MIN_ANCHOR_PKI
        prot["min_drug_interactions"] = 11
        prot = prot.merge(meta_df, on="uniprot_id", how="left")
        prot["gpcr_class"] = prot["gpcr_class"].fillna("Unknown")
        protein_rows.append(prot)

        class_summary = (
            prot.groupby("gpcr_class", as_index=False)
            .agg(
                n_proteins=("uniprot_id", "nunique"),
                mean_n=("n", "mean"),
                mean_ci=("ci", "mean"),
                median_ci=("ci", "median"),
                mean_rmse=("rmse", "mean"),
                median_rmse=("rmse", "median"),
            )
            .sort_values("n_proteins", ascending=False)
        )
        class_summary["protocol"] = protocol
        class_summary["model"] = "V2_oracle"
        class_summary["min_anchor_pki"] = MIN_ANCHOR_PKI
        class_summary["min_drug_interactions"] = 11
        class_rows.append(class_summary)

    protein_df = pd.concat(protein_rows, ignore_index=True)
    class_df = pd.concat(class_rows, ignore_index=True)

    protein_df.to_csv(out_dir / "glass_v2_oracle_min10_per_protein.csv", index=False)
    class_df.to_csv(out_dir / "glass_v2_oracle_min10_gpcr_class_summary.csv", index=False)
    print(class_df.to_string(index=False))


if __name__ == "__main__":
    main()
