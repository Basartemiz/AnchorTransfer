import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_benchmark_filter_ci_panels import (
    QUARTILES,
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    load_benchmarks,
    load_embeddings,
)


def main():
    seqs = build_sequences()
    emb = load_embeddings()
    dtc_ref = build_dtc_reference(seqs)
    bench = load_benchmarks()["GLASS"]

    for protocol in ["unfiltered", "filtered"]:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()
        anchor_df["anchor_quartile"] = pd.qcut(
            anchor_df["anchor_pki"], 4, labels=QUARTILES, duplicates="drop"
        )
        print(f"PROTOCOL {protocol}")
        print(
            "summary",
            len(sdf),
            len(anchor_df),
            anchor_df["uniprot_id"].nunique(),
            anchor_df["ligand_smiles"].nunique(),
        )
        print("quantiles")
        print(anchor_df["anchor_pki"].quantile([0, 0.25, 0.5, 0.75, 1.0]).to_string())
        print("quartiles")
        print(
            anchor_df.groupby("anchor_quartile")["anchor_pki"]
            .agg(["count", "min", "max", "mean", "median"])
            .to_string()
        )
        print()


if __name__ == "__main__":
    main()
