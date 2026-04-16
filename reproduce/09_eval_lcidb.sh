#!/usr/bin/env bash
# Reproduce: Cross-dataset evaluation on LCIdb benchmark.
#
# Downloads LCIdb_v2.csv (~1.3GB) from Zenodo, removes all DTC training overlap
# (protein + drug + pair), computes ESM-2 embeddings for novel proteins, then
# evaluates V2-650M, V2-35M, DeepDTA, ESM-DTA on the clean LCIdb subset.
#
# Requires: GPU, ~10GB disk, ~16GB RAM.
#
# Usage:
#   bash reproduce/09_eval_lcidb.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source reproduce/common.sh
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
echo "=== LCIdb Cross-Dataset Evaluation (device=$DEVICE) ==="

# ── Download LCIdb if needed ──
LCIDB_PATH="data/LCIdb_v2.csv"
if [[ ! -f "$LCIDB_PATH" ]]; then
    echo "Downloading LCIdb_v2.csv (~1.3GB)..."
    mkdir -p data
    curl -L -o "$LCIDB_PATH" \
        "https://zenodo.org/api/records/12178118/files/LCIdb_v2.csv/content"
    echo "  Done: $(wc -l < "$LCIDB_PATH") rows"
fi

# ── Run evaluation ──
echo "Running LCIdb evaluation..."
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPRO_PYTHON" scripts/eval/eval_lcidb.py \
    --device "$DEVICE" \
    --lcidb-path "$LCIDB_PATH" \
    --results-dir results \
    2>&1 | tee results/lcidb_eval_results.txt

echo "=== Done. Results in results/lcidb_eval_results.txt ==="
