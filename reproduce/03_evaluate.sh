#!/usr/bin/env bash
# 03_evaluate.sh — Evaluate all models on all benchmarks
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"

# Benchmarks: davis, glass, bdb_ki, idp
BENCHMARKS=("davis" "glass" "bdb_ki" "idp")

# Models to evaluate
declare -A MODELS
MODELS[v1_35m]="v1"
MODELS[v2_35m]="v2"
MODELS[v2_650m]="v2"

# ESM-2 dim mapping
declare -A ESM_TRAIN
ESM_TRAIN[v1_35m]="data/processed/esm2_35m_dtc.pt"
ESM_TRAIN[v2_35m]="data/processed/esm2_35m_dtc.pt"
ESM_TRAIN[v2_650m]="data/processed/esm2_650m_dtc.pt"

declare -A ESM_BENCH
ESM_BENCH[v1_35m]="data/processed/esm2_35m_benchmark.pt"
ESM_BENCH[v2_35m]="data/processed/esm2_35m_benchmark.pt"
ESM_BENCH[v2_650m]="data/processed/esm2_650m_benchmark.pt"

for model_name in "${!MODELS[@]}"; do
    version="${MODELS[$model_name]}"
    for bench in "${BENCHMARKS[@]}"; do
        bench_csv="data/raw/${bench}_benchmark.csv"
        if [ ! -f "$bench_csv" ]; then
            echo "Skipping $bench (file not found: $bench_csv)"
            continue
        fi

        out_dir="results/${model_name}/${bench}"
        echo "=== Evaluating ${model_name} on ${bench} ==="
        python scripts/evaluate_anchor_transfer.py \
            --model "models/${model_name}/best_model.pt" \
            --model-version "$version" \
            --esm2 "${ESM_TRAIN[$model_name]}" \
            --esm2-benchmark "${ESM_BENCH[$model_name]}" \
            --benchmark "$bench_csv" \
            --training data/processed/dtc_training_interactions.csv \
            --output-dir "$out_dir" \
            --device "$DEVICE"
    done
done

echo "=== All evaluations complete ==="
echo "Results saved under results/"
