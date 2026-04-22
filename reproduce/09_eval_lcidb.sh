#!/usr/bin/env bash
# Reproduce: Cross-dataset evaluation on LCIdb benchmark.
#
# Evaluates CoNCISE (pretrained), ConciseAnchor (MooDeng binary checkpoint),
# and Prot-kNN on LCIdb with zero MooDeng v1 training overlap.
# Binary: pKi >= 7 positive, pKi <= 5 negative, ambiguous excluded.
#
# Requires: GPU, ~15GB disk, ~16GB RAM.
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

# ── Download MooDeng v1 if needed ──
MOODENG_V1="data/moodeng-v1"
if [[ ! -f "$MOODENG_V1/train.csv" ]]; then
    echo "Downloading MooDengDB v1..."
    mkdir -p data
    wget -q "https://zenodo.org/records/15368729/files/moodeng-v1.tar.gz" -O data/moodeng-v1.tar.gz
    tar xzf data/moodeng-v1.tar.gz --no-same-owner -C data/
    rm data/moodeng-v1.tar.gz
    echo "  Done: $(wc -l < $MOODENG_V1/train.csv) train rows"
fi

# ── Check model checkpoint ──
ANCHOR_CKPT="models/concise_anchor_moodeng/best_model.pt"
if [[ ! -f "$ANCHOR_CKPT" ]]; then
    echo "ERROR: ConciseAnchor MooDeng checkpoint not found at $ANCHOR_CKPT"
    echo "Run 00_fetch_artifacts.sh first to download from Zenodo."
    exit 1
fi
echo "ConciseAnchor checkpoint: $ANCHOR_CKPT"

# ── Run evaluation ──
echo "Running LCIdb evaluation..."
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPRO_PYTHON" scripts/eval/eval_lcidb.py \
    --device "$DEVICE" \
    --anchor-ckpt "$ANCHOR_CKPT" \
    --lcidb-path "$LCIDB_PATH" \
    --moodeng-dir "$MOODENG_V1" \
    --results-dir results \
    2>&1 | tee results/lcidb_eval_results.txt

echo "=== Done. Results in results/lcidb_eval_results.txt ==="
