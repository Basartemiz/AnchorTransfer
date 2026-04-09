#!/usr/bin/env bash
# Reproduce: MooDengDB IID + OOD evaluation of ConciseAnchor vs pretrained CoNCISE.
# Requires: GPU, ~50GB disk, ~16GB RAM.
#
# Usage:
#   bash reproduce/07_eval_moodeng.sh
#
# Outputs:
#   results/moodeng_reproduce_results.txt  — full results table
#   results/raygun_moodeng_embeddings.pt   — cached Raygun embeddings (reused if exists)
#   results/raygun_ood_embeddings.pt       — cached OOD Raygun embeddings
#   results/morgan_moodeng_fp.pkl          — cached Morgan fingerprints

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

source reproduce/common.sh
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
echo "=== MooDengDB Evaluation (device=$DEVICE) ==="

# ── Download MooDengDB if needed ──
MOODENG_V1="data/moodeng-v1"
MOODENG_V2="data/moodeng-v2-extended"

if [[ ! -f "$MOODENG_V1/train.csv" ]]; then
    echo "Downloading MooDengDB v1..."
    mkdir -p data
    wget -q "https://zenodo.org/records/15368729/files/moodeng-v1.tar.gz" -O data/moodeng-v1.tar.gz
    tar xzf data/moodeng-v1.tar.gz --no-same-owner -C data/
    rm data/moodeng-v1.tar.gz
    echo "  Done: $(wc -l < $MOODENG_V1/train.csv) train rows"
fi

if [[ ! -f "$MOODENG_V2/test.csv" ]]; then
    echo "Downloading MooDengDB v2-extended (OOD test)..."
    mkdir -p data
    wget -q "https://zenodo.org/records/15368729/files/moodeng-v2-extended.tar.gz" -O data/moodeng-v2-extended.tar.gz
    tar xzf data/moodeng-v2-extended.tar.gz --no-same-owner -C data/
    rm data/moodeng-v2-extended.tar.gz
    echo "  Done: $(wc -l < $MOODENG_V2/test.csv) OOD test rows"
fi

# ── Check model checkpoint ──
ANCHOR_CKPT="models/concise_anchor_moodeng/best_model.pt"
if [[ ! -f "$ANCHOR_CKPT" ]]; then
    echo "ERROR: ConciseAnchor checkpoint not found at $ANCHOR_CKPT"
    echo "Train it first with: python scripts/train/train_moodeng_anchor_eval.py"
    exit 1
fi
echo "ConciseAnchor checkpoint: $ANCHOR_CKPT"

# ── Run evaluation ──
echo "Running evaluation..."
PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$REPRO_PYTHON" reproduce/eval_moodeng_reproduce.py \
    --device "$DEVICE" \
    --anchor-ckpt "$ANCHOR_CKPT" \
    --moodeng-dir "$MOODENG_V1" \
    --ood-test "$MOODENG_V2/test.csv" \
    --results-dir results \
    2>&1 | tee results/moodeng_reproduce_results.txt

echo "=== Done. Results in results/moodeng_reproduce_results.txt ==="
