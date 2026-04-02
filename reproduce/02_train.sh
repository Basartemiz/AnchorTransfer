#!/usr/bin/env bash
# 02_train.sh — Train all models reported in the paper
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED=42

echo "=== Training V1 (AnchorTransferDTA, ESM-2 35M) ==="
python scripts/train_anchor_transfer.py \
    --graph data/processed/esm2_35m_dtc.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/v1_35m \
    --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --esm2-dim 480 --seed "$SEED"

echo "=== Training V2 (AnchorTransferDTAv2, ESM-2 35M) ==="
python scripts/train_anchor_transfer_v2.py \
    --graph data/processed/esm2_35m_dtc.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/v2_35m \
    --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --proj-dim 256 --seed "$SEED"

echo "=== Training V2 (AnchorTransferDTAv2, ESM-2 650M) ==="
python scripts/train_anchor_transfer_v2.py \
    --graph data/processed/esm2_650m_dtc.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/v2_650m \
    --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
    --proj-dim 256 --seed "$SEED"

echo "=== All models trained ==="
