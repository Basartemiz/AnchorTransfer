#!/usr/bin/env bash
# reproduce/train_affinity_anchor.sh
# Step 1: Pre-compute anchor embeddings for all ordered BindingDB proteins
# Step 2: Train affinity model on frozen anchor embeddings
# Step 3: Evaluate on the 203-protein benchmark
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

MODEL_CHECKPOINT="${1:-models/tm08_e03-08/best_model.pt}"
OUTPUT_DIR="${2:-results/anchor_affinity}"

echo "=== Step 1: Pre-compute anchor embeddings ==="
python scripts/precompute_affinity_anchors.py \
    --model-path "${MODEL_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}/precompute" \
    --tm-threshold 0.4 \
    --plddt-threshold 70.0 \
    --n-conformations 10 \
    --resume \
    --use-esm2

echo "=== Step 2: Train affinity model ==="
python scripts/train_affinity_anchor.py \
    --embeddings "${OUTPUT_DIR}/precompute/anchor_embeddings.pt" \
    --output-dir "${OUTPUT_DIR}/model" \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --patience 10 \
    --amp

echo "=== Step 3: Evaluate on benchmark ==="
python scripts/evaluate_affinity_benchmark.py \
    --model-path "${OUTPUT_DIR}/model/best_model.pt" \
    --output-dir "${OUTPUT_DIR}/eval" \
    --device cuda

echo "=== Done ==="
