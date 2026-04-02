#!/usr/bin/env bash
# 01_prepare_data.sh — Filter DTC data and extract ESM-2 embeddings
#
# Prerequisites:
#   - data/raw/DTC_data.csv (DrugTargetCommons bulk export)
#   - data/raw/dtc_proteins.csv (uniprot_id,sequence for all DTC proteins)
#   - data/raw/benchmark_proteins.csv (uniprot_id,sequence for benchmark proteins)
#   - (Optional) data/raw/benchmark_exclude.txt (UniProt IDs to exclude from training)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

DEVICE="${DEVICE:-cuda}"
ESM2_35M="esm2_t12_35M_UR50D"
ESM2_650M="esm2_t33_650M_UR50D"

echo "=== Step 1: Filter and split DTC data ==="
python scripts/prepare_dtc_data.py \
    --dtc-csv data/raw/DTC_data.csv \
    --output-dir data/processed \
    --exclude-proteins data/raw/benchmark_exclude.txt \
    --seed 42

echo "=== Step 2: Extract ESM-2 35M embeddings (training) ==="
python scripts/extract_esm2_embeddings.py \
    --input data/raw/dtc_proteins.csv \
    --output data/processed/esm2_35m_dtc.pt \
    --model "$ESM2_35M" --device "$DEVICE" --batch-size 8

echo "=== Step 3: Extract ESM-2 650M embeddings (training) ==="
python scripts/extract_esm2_embeddings.py \
    --input data/raw/dtc_proteins.csv \
    --output data/processed/esm2_650m_dtc.pt \
    --model "$ESM2_650M" --device "$DEVICE" --batch-size 4

echo "=== Step 4: Extract ESM-2 embeddings (benchmark proteins) ==="
python scripts/extract_esm2_embeddings.py \
    --input data/raw/benchmark_proteins.csv \
    --output data/processed/esm2_35m_benchmark.pt \
    --model "$ESM2_35M" --device "$DEVICE" --batch-size 8

python scripts/extract_esm2_embeddings.py \
    --input data/raw/benchmark_proteins.csv \
    --output data/processed/esm2_650m_benchmark.pt \
    --model "$ESM2_650M" --device "$DEVICE" --batch-size 4

echo "=== Data preparation complete ==="
