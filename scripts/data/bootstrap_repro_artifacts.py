#!/usr/bin/env python3
"""Search local filesystem for missing reproduction artifacts.

Fallback used by reproduce/00_fetch_artifacts.sh when Zenodo downloads fail
or when manually-obtained files aren't in the expected locations.

Usage:
    python scripts/data/bootstrap_repro_artifacts.py \
        --repo-root /path/to/repo \
        --search-root "$HOME/Desktop/IDP work" \
        --search-root "$HOME/Downloads"
"""
import argparse
import logging
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# Files to search for: (filename_pattern, target_relative_path)
ARTIFACT_MAP = {
    # Raw data
    "DTC_data.csv": "data/raw/DTC_data.csv",
    "dtc_proteins.csv": "data/raw/dtc_proteins.csv",
    "benchmark_proteins.csv": "data/raw/benchmark_proteins.csv",
    "benchmark_exclude.txt": "data/raw/benchmark_exclude.txt",
    # Processed data
    "dtc_training_interactions.csv": "data/processed/dtc_training_interactions.csv",
    "bindingdb_interactions.csv": "data/processed/bindingdb_interactions.csv",
    "merged_sequences.json": "data/processed/merged_sequences.json",
    # ESM embeddings (handle legacy filename)
    "esm2_35m_dtc_proteins_full.pt": "data/processed/esm2_35m_dtc.pt",
    "esm2_35m_dtc.pt": "data/processed/esm2_35m_dtc.pt",
    "esm2_650m_dtc.pt": "data/processed/esm2_650m_dtc.pt",
    "esm2_35m_benchmark.pt": "data/processed/esm2_35m_benchmark.pt",
    "esm2_650m_benchmark.pt": "data/processed/esm2_650m_benchmark.pt",
    # Benchmark datasets
    "davis_benchmark.csv": "data/raw/davis_benchmark.csv",
    "glass2_ki_interactions.csv": "data/raw/glass/glass2_ki_interactions.csv",
    "glass2_sequences.json": "data/raw/glass/glass2_sequences.json",
    # Raygun embeddings
    "raygun_bdb_embeddings.pt": "results/raygun_bdb_embeddings.pt",
}


def search_and_copy(search_roots: list[Path], repo_root: Path) -> int:
    copied = 0
    for filename, rel_target in ARTIFACT_MAP.items():
        target = repo_root / rel_target
        if target.exists():
            continue

        for root in search_roots:
            matches = list(root.rglob(filename))
            if matches:
                source = matches[0]
                target.parent.mkdir(parents=True, exist_ok=True)
                log.info("Copying %s → %s", source, target)
                shutil.copy2(source, target)
                copied += 1
                break

    return copied


def main():
    parser = argparse.ArgumentParser(description="Bootstrap reproduction artifacts from local filesystem")
    parser.add_argument("--repo-root", required=True, help="Repository root directory")
    parser.add_argument("--search-root", action="append", dest="search_roots",
                        help="Directory to search for artifacts (can be repeated)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    search_roots = [Path(r) for r in (args.search_roots or []) if Path(r).is_dir()]

    if not search_roots:
        log.warning("No valid search roots provided")
        return

    log.info("Searching %d roots for missing artifacts...", len(search_roots))
    copied = search_and_copy(search_roots, repo_root)
    log.info("Copied %d artifacts from local search roots", copied)


if __name__ == "__main__":
    main()
