#!/usr/bin/env bash
# 02_train.sh — Train all models reported in the paper
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
EPOCHS="${EPOCHS:-100}"
V1_EPOCHS="${V1_EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"
SEED=42
START_MODEL="${START_MODEL:-v1_35m}"

if [[ ! -f "data/processed/dtc_training_interactions.csv" ]] || [[ ! -f "data/processed/esm2_35m_dtc.pt" ]] || [[ ! -f "data/processed/esm2_650m_dtc.pt" ]]; then
    echo "Training inputs are missing. Run bash reproduce/01_prepare_data.sh first." >&2
fi

require_file "data/processed/dtc_training_interactions.csv"
require_file "data/processed/esm2_35m_dtc.pt"
require_file "data/processed/esm2_650m_dtc.pt"

run_v1=true
run_v2_35m=true
run_v2_650m=true

case "$START_MODEL" in
    v1_35m)
        ;;
    v2_35m)
        run_v1=false
        ;;
    v2_650m)
        run_v1=false
        run_v2_35m=false
        ;;
    *)
        echo "Unknown START_MODEL: $START_MODEL" >&2
        exit 1
        ;;
esac

if [[ "$run_v1" == true ]]; then
    echo "=== Training V1 (AnchorTransferDTA, ESM-2 35M) ==="
    "$REPRO_PYTHON" scripts/train_anchor_transfer.py \
        --graph data/processed/esm2_35m_dtc.pt \
        --interactions data/processed/dtc_training_interactions.csv \
        --output-dir models/v1_35m \
        --device "$DEVICE" --epochs "$V1_EPOCHS" --batch-size "$BATCH_SIZE" \
        --esm2-dim 480 --seed "$SEED"
else
    echo "=== Skipping V1 (START_MODEL=$START_MODEL) ==="
fi

if [[ "$run_v2_35m" == true ]]; then
    echo "=== Training V2 (AnchorTransferDTAv2, ESM-2 35M) ==="
    "$REPRO_PYTHON" scripts/train_anchor_transfer_v2.py \
        --graph data/processed/esm2_35m_dtc.pt \
        --interactions data/processed/dtc_training_interactions.csv \
        --output-dir models/v2_35m \
        --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --proj-dim 256 --seed "$SEED"
else
    echo "=== Skipping V2-35M (START_MODEL=$START_MODEL) ==="
fi

if [[ "$run_v2_650m" == true ]]; then
    echo "=== Training V2 (AnchorTransferDTAv2, ESM-2 650M) ==="
    "$REPRO_PYTHON" scripts/train_anchor_transfer_v2.py \
        --graph data/processed/esm2_650m_dtc.pt \
        --interactions data/processed/dtc_training_interactions.csv \
        --output-dir models/v2_650m \
        --device "$DEVICE" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" \
        --proj-dim 256 --seed "$SEED"
else
    echo "=== Skipping V2-650M (START_MODEL=$START_MODEL) ==="
fi

echo "=== All models trained ==="
