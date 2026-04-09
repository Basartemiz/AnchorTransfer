"""Orchestrate all DrugBAN paper replication experiments.

Usage:
    python -m scripts.drugban_paper.run_all [--data-dir data/drugban_paper]
                                             [--output-dir models/drugban_paper]
                                             [--seeds 0,1,2,3,4]
                                             [--datasets bindingdb,biosnap,human]
                                             [--models drugban,anchor_drugban]
                                             [--epochs 100]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.drugban_paper.fetch_data import fetch_all
from scripts.drugban_paper.train import train_one_run

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# All experiment configurations: (dataset, split)
ALL_CONFIGS = [
    ("bindingdb", "random"),
    ("biosnap", "random"),
    ("human", "random"),
    ("human", "cold"),
    ("bindingdb", "cluster"),
    ("biosnap", "cluster"),
]

# anchor_drugban runs first to generate subsets, then drugban_anchor_subset uses them
ALL_MODELS = ["drugban", "anchor_drugban", "drugban_anchor_subset", "anchor_drugban_oracle", "drugban_oracle_subset"]
DEFAULT_SEEDS = [0, 1, 2, 3, 4]


def main():
    parser = argparse.ArgumentParser(description="DrugBAN paper replication")
    parser.add_argument("--data-dir", default="data/drugban_paper")
    parser.add_argument("--output-dir", default="models/drugban_paper")
    parser.add_argument(
        "--results-csv", default="results/drugban_paper_replication.csv"
    )
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated seeds")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets to run (default: all)",
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits to run (default: all for each dataset)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated models (default: drugban,anchor_drugban)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",") if args.models else ALL_MODELS

    # Filter configs
    configs = ALL_CONFIGS
    if args.datasets:
        requested_ds = set(args.datasets.split(","))
        configs = [(d, s) for d, s in configs if d in requested_ds]
    if args.splits:
        requested_sp = set(args.splits.split(","))
        configs = [(d, s) for d, s in configs if s in requested_sp]

    # 1. Fetch data
    log.info("Fetching DrugBAN paper datasets...")
    fetch_all(args.data_dir)

    # 2. Run experiments
    all_results = []
    results_path = Path(args.results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing results
    if results_path.exists():
        existing = pd.read_csv(results_path)
        all_results = existing.to_dict("records")
        completed = set(
            (r["dataset"], r["split"], r["model"], int(r["seed"]))
            for r in all_results
            if "error" not in r or pd.isna(r.get("error"))
        )
        log.info(f"Resuming: {len(completed)} runs already completed")
    else:
        completed = set()

    total = len(configs) * len(models) * len(seeds)
    done = 0

    for dataset, split in configs:
        for model_type in models:
            for seed in seeds:
                done += 1
                key = (dataset, split, model_type, seed)
                if key in completed:
                    log.info(
                        f"[{done}/{total}] SKIP (done): "
                        f"{dataset}/{split}/{model_type}/seed{seed}"
                    )
                    continue

                log.info(
                    f"[{done}/{total}] Running: "
                    f"{dataset}/{split}/{model_type}/seed{seed}"
                )
                try:
                    result = train_one_run(
                        dataset=dataset,
                        split=split,
                        model_type=model_type,
                        seed=seed,
                        data_dir=args.data_dir,
                        output_dir=args.output_dir,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        epochs=args.epochs,
                        patience=args.patience,
                    )
                    all_results.append(result)
                except Exception as e:
                    log.error(f"FAILED: {key}: {e}", exc_info=True)
                    all_results.append(
                        {
                            "dataset": dataset,
                            "split": split,
                            "model": model_type,
                            "seed": seed,
                            "error": str(e),
                        }
                    )

                # Save after each run (crash-safe) and print running summary
                pd.DataFrame(all_results).to_csv(results_path, index=False)
                _print_summary(all_results, results_path)

    # 3. Print final summary table
    _print_summary(all_results, results_path)


def _print_summary(all_results: list[dict], results_path: Path):
    """Print formatted summary table matching DrugBAN paper style."""
    df = pd.DataFrame(all_results)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")

    if df.empty:
        print("No successful runs to summarize.")
        return

    print("\n" + "=" * 90)
    print("DRUGBAN PAPER REPLICATION RESULTS")
    print("=" * 90)

    metrics = ["auroc", "auprc", "accuracy", "sensitivity", "specificity"]

    for (dataset, split), grp in df.groupby(["dataset", "split"]):
        print(f"\n--- {dataset} / {split} ---")
        header = f"  {'Model':20s} "
        for m in metrics:
            header += f"{m:>12s}  "
        print(header)
        print("  " + "-" * 86)

        for model_name, mgrp in grp.groupby("model"):
            line = f"  {model_name:20s} "
            for m in metrics:
                if m in mgrp.columns:
                    mean = mgrp[m].mean()
                    std = mgrp[m].std()
                    line += f"{mean:.3f}±{std:.3f}  "
                else:
                    line += f"{'N/A':>12s}  "
            print(line)

    print(f"\nFull results saved to: {results_path}")


if __name__ == "__main__":
    main()
