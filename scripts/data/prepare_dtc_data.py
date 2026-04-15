#!/usr/bin/env python3
"""Filter DTC bulk CSV to clean Ki/Kd interactions for training.

Wraps anchor_transfer.data.dtc_loader.filter_dtc() with a CLI.

Usage:
    python scripts/data/prepare_dtc_data.py \
        --dtc-csv data/raw/DTC_data.csv \
        --output-dir data/processed \
        --seed 42
"""
import argparse
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter DTC data for training")
    parser.add_argument("--dtc-csv", required=True, help="Path to raw DTC_data.csv")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--exclude-proteins", help="File with UniProt IDs to exclude (one per line)")
    parser.add_argument("--pki-min", type=float, default=3.0, help="Minimum pKi to keep")
    parser.add_argument("--pki-max", type=float, default=12.0, help="Maximum pKi to keep")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    from anchor_transfer.data.dtc_loader import filter_dtc

    exclude = None
    if args.exclude_proteins:
        exclude_path = Path(args.exclude_proteins)
        if exclude_path.exists():
            exclude = set(exclude_path.read_text().strip().splitlines())
            log.info("Excluding %d proteins from %s", len(exclude), exclude_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Filtering DTC data from %s", args.dtc_csv)
    df = filter_dtc(
        Path(args.dtc_csv),
        exclude_proteins=exclude,
        pki_min=args.pki_min,
        pki_max=args.pki_max,
    )

    out_path = out_dir / "dtc_training_interactions.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved %d interactions (%d proteins, %d drugs) to %s",
             len(df), df["uniprot_id"].nunique(), df["ligand_smiles"].nunique(), out_path)


if __name__ == "__main__":
    main()
