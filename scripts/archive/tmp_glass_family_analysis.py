import json
from pathlib import Path

import pandas as pd


def normalize_gpcr_class(value):
    if value is None:
        return "Unknown"
    if isinstance(value, list):
        if not value:
            return "Unknown"
        value = value[0]
    value = str(value).strip()
    return value if value else "Unknown"


def first_pfam(value):
    if value is None:
        return "Unknown"
    if isinstance(value, list):
        if not value:
            return "Unknown"
        return str(value[0])
    value = str(value).strip()
    return value if value else "Unknown"


def main():
    base = Path("results/benchmark_filter_ci_panels")
    per_protein = pd.read_csv(base / "glass_equal_anchor_bins_per_protein.csv")

    protein_meta = json.load(open("data/raw/glass/protein.json"))
    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    dtc_proteins = set(dtc["uniprot_id"].astype(str))
    dtc_sequences = set(json.load(open("data/processed/dtc_sequences.json")).values())

    meta_rows = []
    for uid, info in protein_meta.items():
        if not isinstance(info, dict):
            continue
        seq = info.get("sequence")
        exact = str(uid) in dtc_proteins
        seq_overlap = seq in dtc_sequences if seq else False
        if exact:
            overlap_group = "exact_uid_overlap"
        elif seq_overlap:
            overlap_group = "sequence_overlap_only"
        else:
            overlap_group = "novel_to_dtc"
        meta_rows.append(
            {
                "uniprot_id": str(uid),
                "gpcr_class": normalize_gpcr_class(info.get("gpcr_class")),
                "pfam_id": first_pfam(info.get("pfam_ids")),
                "overlap_group": overlap_group,
                "sequence_length": info.get("sequence_length"),
            }
        )
    meta_df = pd.DataFrame(meta_rows)

    # Aggregate the equal-bin per-protein metrics to one row per protein/model/protocol.
    overall = (
        per_protein.groupby(["protocol", "model", "uniprot_id"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "total_n": int(g["n"].sum()),
                    "weighted_ci": float((g["ci"] * g["n"]).sum() / g["n"].sum()),
                    "weighted_rmse": float((g["rmse"] * g["n"]).sum() / g["n"].sum()),
                    "n_bins": int(g["anchor_bin"].nunique()),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    overall = overall.merge(meta_df, on="uniprot_id", how="left")
    overall["gpcr_class"] = overall["gpcr_class"].fillna("Unknown")
    overall["pfam_id"] = overall["pfam_id"].fillna("Unknown")
    overall["overlap_group"] = overall["overlap_group"].fillna("Unknown")

    class_summary = (
        overall.groupby(["protocol", "model", "gpcr_class"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_total_n=("total_n", "mean"),
            mean_ci=("weighted_ci", "mean"),
            median_ci=("weighted_ci", "median"),
            mean_rmse=("weighted_rmse", "mean"),
            median_rmse=("weighted_rmse", "median"),
        )
        .sort_values(["protocol", "model", "n_proteins"], ascending=[True, True, False])
    )

    overlap_summary = (
        overall.groupby(["protocol", "model", "overlap_group"], as_index=False)
        .agg(
            n_proteins=("uniprot_id", "nunique"),
            mean_total_n=("total_n", "mean"),
            mean_ci=("weighted_ci", "mean"),
            median_ci=("weighted_ci", "median"),
            mean_rmse=("weighted_rmse", "mean"),
            median_rmse=("weighted_rmse", "median"),
        )
        .sort_values(["protocol", "model", "overlap_group"])
    )

    protein_inventory = (
        meta_df.groupby(["gpcr_class", "overlap_group"], as_index=False)
        .agg(n_proteins=("uniprot_id", "nunique"))
        .sort_values(["overlap_group", "n_proteins"], ascending=[True, False])
    )

    overall.to_csv(base / "glass_family_per_protein_overall.csv", index=False)
    class_summary.to_csv(base / "glass_family_gpcr_class_summary.csv", index=False)
    overlap_summary.to_csv(base / "glass_family_overlap_summary.csv", index=False)
    protein_inventory.to_csv(base / "glass_family_inventory.csv", index=False)

    print("INVENTORY")
    print(protein_inventory.to_string(index=False))
    print("\nOVERLAP SUMMARY V2_oracle")
    print(overlap_summary[overlap_summary["model"] == "V2_oracle"].to_string(index=False))
    print("\nCLASS SUMMARY V2_oracle")
    print(class_summary[class_summary["model"] == "V2_oracle"].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
