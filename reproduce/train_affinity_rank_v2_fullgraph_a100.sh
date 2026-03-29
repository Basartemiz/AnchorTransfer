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

MODEL_DIR="${ROOT_DIR}/models/${GRAPH_TAG}/affinity_dyngat_fullgraph_rank_v2"
LOG_DIR="${ROOT_DIR}/logs/${GRAPH_TAG}"
LOG_PATH="${LOG_DIR}/affinity_rank_v2_fullgraph_a100.log"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-2}"
TRAIN_DRUG_CACHE_WORKERS="${TRAIN_DRUG_CACHE_WORKERS:-32}"
TRAIN_ANCHOR_CACHE_WORKERS="${TRAIN_ANCHOR_CACHE_WORKERS:-8}"
TRAIN_FOLDSEEK_THREADS="${TRAIN_FOLDSEEK_THREADS:-8}"
TRAIN_RANKING_WEIGHT="${TRAIN_RANKING_WEIGHT:-1.0}"
TRAIN_RANKING_MARGIN="${TRAIN_RANKING_MARGIN:-0.2}"
TRAIN_MAX_RANKING_PAIRS="${TRAIN_MAX_RANKING_PAIRS:-4096}"
PREBUILD_ANCHOR_CACHE="${PREBUILD_ANCHOR_CACHE:-1}"
EPOCHS="${EPOCHS:-50}"

mkdir -p "${LOG_DIR}"

EXTRA_ARGS=("$@")
TRAIN_ARGS=()
for arg in "${EXTRA_ARGS[@]}"; do
  case "${arg}" in
    --rebuild-anchor-cache|--build-anchor-cache-only)
      ;;
    *)
      TRAIN_ARGS+=("${arg}")
      ;;
  esac
done

echo "=== A100 affinity rank_v2 full-graph training (${GRAPH_TAG}) ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Graph dir: ${ROOT_DIR}/data/graphs/${GRAPH_TAG}"
echo "Model dir: ${MODEL_DIR}"
echo "Log: ${LOG_PATH}"
echo "Ranking: weight=${TRAIN_RANKING_WEIGHT} margin=${TRAIN_RANKING_MARGIN} max_pairs=${TRAIN_MAX_RANKING_PAIRS}"
echo "Loader: batch=${TRAIN_BATCH_SIZE} workers=${TRAIN_NUM_WORKERS} prefetch=${TRAIN_PREFETCH_FACTOR}"
echo "Drug-cache workers: ${TRAIN_DRUG_CACHE_WORKERS}"
echo "Anchor cache: workers=${TRAIN_ANCHOR_CACHE_WORKERS} foldseek_threads=${TRAIN_FOLDSEEK_THREADS} prebuild=${PREBUILD_ANCHOR_CACHE}"

COMMON_ARGS=(
  --graph-dir "data/graphs/${GRAPH_TAG}"
  --model-version rank_v2
  --amp
  --dropout 0.05
  --head-dropout 0.1
  --drug-dropout 0.1
  --ranking-weight "${TRAIN_RANKING_WEIGHT}"
  --ranking-margin "${TRAIN_RANKING_MARGIN}"
  --max-ranking-pairs-per-protein "${TRAIN_MAX_RANKING_PAIRS}"
  --anchor-cache-workers "${TRAIN_ANCHOR_CACHE_WORKERS}"
  --foldseek-threads "${TRAIN_FOLDSEEK_THREADS}"
)

if [ "${PREBUILD_ANCHOR_CACHE}" = "1" ]; then
  echo "=== Prebuilding training anchor cache ==="
  python -u scripts/train_affinity_full_graph.py \
    "${COMMON_ARGS[@]}" \
    --device cpu \
    --build-anchor-cache-only \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${LOG_PATH}"
fi

echo "=== Launching training ==="
python -u scripts/train_affinity_full_graph.py \
  "${COMMON_ARGS[@]}" \
  --device cuda \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --num-workers "${TRAIN_NUM_WORKERS}" \
  --prefetch-factor "${TRAIN_PREFETCH_FACTOR}" \
  --drug-cache-workers "${TRAIN_DRUG_CACHE_WORKERS}" \
  --persistent-workers \
  --pin-memory \
  --epochs "${EPOCHS}" \
  --lr 1e-4 \
  --weight-decay 5e-6 \
  --patience 50 \
  "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_PATH}"

echo "=== Done ==="
echo "Best model: ${MODEL_DIR}/best_model.pt"
