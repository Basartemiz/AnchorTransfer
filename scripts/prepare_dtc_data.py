#!/usr/bin/env python3
"""Prepare DTC training data: filter, split, and save.

Downloads or reads DrugTargetCommons bulk CSV, filters to Ki interactions,
converts to pKi, performs protein-level train/val/test split, and saves
the resulting DataFrames.

Usage:
  python scripts/prepare_dtc_data.py \
    --dtc-csv data/raw/DTC_data.csv \
    --output-dir data/processed \
    --val-frac 0.1 --test-frac 0.1
"""
import argparse
import logging
import random
from pathlib import Path

import pandas as pd

from anchor_transfer.data.dtc_loader import filter_dtc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Prepare DTC training data")
    parser.add_argument("--dtc-csv", required=True, help="Path to DTC bulk CSV")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--exclude-proteins", default=None,
                        help="Text file with UniProt IDs to exclude (one per line)")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude = set()
    if args.exclude_proteins:
        with open(args.exclude_proteins) as f:
            exclude = {line.strip() for line in f if line.strip()}
        logger.info("Excluding %d benchmark proteins", len(exclude))

    df = filter_dtc(Path(args.dtc_csv), exclude_proteins=exclude or None)

    # Protein-level split
    proteins = sorted(df["uniprot_id"].unique())
    random.shuffle(proteins)
    n_test = max(1, int(len(proteins) * args.test_frac))
    n_val = max(1, int(len(proteins) * args.val_frac))
    test_proteins = set(proteins[:n_test])
    val_proteins = set(proteins[n_test:n_test + n_val])
    train_proteins = set(proteins[n_test + n_val:])

    train_df = df[df["uniprot_id"].isin(train_proteins)]
    val_df = df[df["uniprot_id"].isin(val_proteins)]
    test_df = df[df["uniprot_id"].isin(test_proteins)]

    train_df.to_csv(output_dir / "dtc_train.csv", index=False)
    val_df.to_csv(output_dir / "dtc_val.csv", index=False)
    test_df.to_csv(output_dir / "dtc_test.csv", index=False)

    # Also save full training set (train + val for final model)
    full_train = pd.concat([train_df, val_df])
    full_train.to_csv(output_dir / "dtc_training_interactions.csv", index=False)

    # Save protein sequences for ESM-2 extraction
    all_proteins_df = df[["uniprot_id"]].drop_duplicates()
    all_proteins_df.to_csv(output_dir / "dtc_protein_ids.txt", index=False, header=False)

    logger.info("Split: %d train, %d val, %d test proteins",
                len(train_proteins), len(val_proteins), len(test_proteins))
    logger.info("Interactions: %d train, %d val, %d test",
                len(train_df), len(val_df), len(test_df))
    logger.info("Saved to %s", output_dir)


if __name__ == "__main__":
    main()
