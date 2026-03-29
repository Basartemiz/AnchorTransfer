#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "${ROOT_DIR}"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi
source reproduce/config.sh

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

HOPS="${AFFINITY_RANK_HOPS:-2}"
SUBGRAPHS_PATH="${ROOT_DIR}/data/graphs/${GRAPH_TAG}/protein_subgraphs_${HOPS}hop.pt"
MODEL_DIR="${ROOT_DIR}/models/${GRAPH_TAG}/affinity_dyngat_${HOPS}hop_rank_v2"
LOG_DIR="${ROOT_DIR}/logs/${GRAPH_TAG}"
LOG_PATH="${LOG_DIR}/affinity_rank_v2_${HOPS}hop_a100.log"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2048}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
TRAIN_DRUG_CACHE_WORKERS="${TRAIN_DRUG_CACHE_WORKERS:-32}"
TRAIN_RANKING_WEIGHT="${TRAIN_RANKING_WEIGHT:-1.0}"
TRAIN_RANKING_MARGIN="${TRAIN_RANKING_MARGIN:-0.2}"
TRAIN_MAX_RANKING_PAIRS="${TRAIN_MAX_RANKING_PAIRS:-4096}"

mkdir -p "${LOG_DIR}"

echo "=== A100 affinity rank_v2 training (${GRAPH_TAG}, ${HOPS}-hop) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Subgraphs: ${SUBGRAPHS_PATH}"
echo "Model dir: ${MODEL_DIR}"
echo "Log: ${LOG_PATH}"
echo "Ranking: weight=${TRAIN_RANKING_WEIGHT} margin=${TRAIN_RANKING_MARGIN} max_pairs=${TRAIN_MAX_RANKING_PAIRS}"
echo "Loader: batch=${TRAIN_BATCH_SIZE} workers=${TRAIN_NUM_WORKERS} prefetch=${TRAIN_PREFETCH_FACTOR}"
echo "Drug-cache workers: ${TRAIN_DRUG_CACHE_WORKERS}"

if [ ! -f "${SUBGRAPHS_PATH}" ]; then
  echo "=== Extracting ${HOPS}-hop subgraphs ==="
  python scripts/extract_subgraphs.py \
    --graph-dir "data/graphs/${GRAPH_TAG}" \
    --hops "${HOPS}" \
    --output "${SUBGRAPHS_PATH}"
else
  echo "Subgraphs already exist, skipping extraction."
fi

echo "=== Launching training ==="
python -u scripts/train_affinity.py \
  --graph-dir "data/graphs/${GRAPH_TAG}" \
  --subgraphs-path "${SUBGRAPHS_PATH}" \
  --model-version rank_v2 \
  --device cuda \
  --amp \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --num-workers "${TRAIN_NUM_WORKERS}" \
  --prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  --drug-cache-workers "${TRAIN_DRUG_CACHE_WORKERS}" \
  --persistent-workers \
  --pin-memory \
  --epochs "${EPOCHS}" \
  --lr 1e-4 \
  --weight-decay 5e-6 \
  --dropout 0.05 \
  --head-dropout 0.1 \
  --drug-dropout 0.1 \
  --ranking-weight "${TRAIN_RANKING_WEIGHT}" \
  --ranking-margin "${TRAIN_RANKING_MARGIN}" \
  --max-ranking-pairs-per-protein "${TRAIN_MAX_RANKING_PAIRS}" \
  --patience 50 \
  "$@" 2>&1 | tee "${LOG_PATH}"

echo "=== Done ==="
echo "Best model: ${MODEL_DIR}/best_model.pt"
