import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_benchmark_filter_ci_panels import (  # noqa: E402
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    load_benchmarks,
    load_embeddings,
    save_anchor_artifacts,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=["filtered", "unfiltered"],
        choices=["filtered", "unfiltered"],
    )
    args = parser.parse_args()

    seqs = build_sequences()
    emb = load_embeddings()
    dtc_ref = build_dtc_reference(seqs)
    bench = load_benchmarks()["GLASS"]
    out_dir = Path("results/benchmark_filter_ci_panels")
    out_dir.mkdir(parents=True, exist_ok=True)

    for protocol in args.protocols:
        sdf = apply_protocol_filters(bench, protocol, seqs, emb, dtc_ref)
        strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
        anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
        if anchor_df.empty:
            continue
        anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
        anchor_df["anchor_quartile"] = pd.qcut(
            anchor_df["anchor_pki"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
        )
        anchor_df["max_tanimoto_to_dtc"] = float("nan")
        anchor_df["tanimoto_bin"] = np.nan
        save_anchor_artifacts(
            out_dir, "GLASS", protocol, sdf, anchor_df, strongest_uid, second_uid, weakest_uid
        )
        print(protocol, len(sdf), len(anchor_df), anchor_df["ligand_smiles"].nunique())


if __name__ == "__main__":
    main()
