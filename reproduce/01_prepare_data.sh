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
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
ESM2_35M="esm2_t12_35M_UR50D"
ESM2_650M="esm2_t33_650M_UR50D"

missing_raw=()
for required_path in \
    "data/raw/DTC_data.csv" \
    "data/raw/dtc_proteins.csv" \
    "data/raw/benchmark_proteins.csv"
do
    if [[ ! -f "$required_path" ]]; then
        missing_raw+=("${required_path#$REPO_ROOT/}")
    fi
done

if (( ${#missing_raw[@]} > 0 )); then
    echo "Missing raw inputs: ${missing_raw[*]}" >&2
    echo "Run bash reproduce/00_fetch_artifacts.sh to download/generate them." >&2
    exit 1
fi

EXCLUDE_ARGS=()
if [[ -f "data/raw/benchmark_exclude.txt" ]]; then
    EXCLUDE_ARGS+=(--exclude-proteins "data/raw/benchmark_exclude.txt")
fi

echo "=== Step 1: Filter and split DTC data ==="
if [[ -f "data/processed/dtc_training_interactions.csv" ]] && [[ "${FORCE_DTC_PREP:-0}" != "1" ]]; then
    echo "Using existing data/processed/dtc_training_interactions.csv"
else
    "$REPRO_PYTHON" scripts/data/prepare_dtc_data.py \
        --dtc-csv data/raw/DTC_data.csv \
        --output-dir data/processed \
        "${EXCLUDE_ARGS[@]}" \
        --seed 42
fi

echo "=== Step 2: Extract ESM-2 35M embeddings (training) ==="
if [[ "${FORCE_EMBED_REGEN:-0}" != "1" ]] && \
   embedding_cache_complete_and_finite "data/processed/esm2_35m_dtc.pt" "data/raw/dtc_proteins.csv"; then
    echo "Using existing data/processed/esm2_35m_dtc.pt"
else
    if [[ -f "data/processed/esm2_35m_dtc.pt" ]]; then
        echo "Regenerating data/processed/esm2_35m_dtc.pt (cache is incomplete or contains non-finite embeddings)"
    fi
    "$REPRO_PYTHON" scripts/data/extract_esm2_embeddings.py \
        --input data/raw/dtc_proteins.csv \
        --output data/processed/esm2_35m_dtc.pt \
        --model "$ESM2_35M" --device "$DEVICE" --batch-size 8
fi

echo "=== Step 3: Extract ESM-2 650M embeddings (training) ==="
if [[ "${FORCE_EMBED_REGEN:-0}" != "1" ]] && \
   embedding_cache_complete_and_finite "data/processed/esm2_650m_dtc.pt" "data/raw/dtc_proteins.csv"; then
    echo "Using existing data/processed/esm2_650m_dtc.pt"
else
    if [[ -f "data/processed/esm2_650m_dtc.pt" ]]; then
        echo "Regenerating data/processed/esm2_650m_dtc.pt (cache is incomplete or contains non-finite embeddings)"
    fi
    "$REPRO_PYTHON" scripts/data/extract_esm2_embeddings.py \
        --input data/raw/dtc_proteins.csv \
        --output data/processed/esm2_650m_dtc.pt \
        --model "$ESM2_650M" --device "$DEVICE" --batch-size 4
fi

echo "=== Step 4: Extract ESM-2 embeddings (benchmark proteins) ==="
if [[ "${FORCE_EMBED_REGEN:-0}" != "1" ]] && \
   embedding_cache_complete_and_finite "data/processed/esm2_35m_benchmark.pt" "data/raw/benchmark_proteins.csv"; then
    echo "Using existing data/processed/esm2_35m_benchmark.pt"
else
    if [[ -f "data/processed/esm2_35m_benchmark.pt" ]]; then
        echo "Regenerating data/processed/esm2_35m_benchmark.pt (cache is incomplete or contains non-finite embeddings)"
    fi
    "$REPRO_PYTHON" scripts/data/extract_esm2_embeddings.py \
        --input data/raw/benchmark_proteins.csv \
        --output data/processed/esm2_35m_benchmark.pt \
        --model "$ESM2_35M" --device "$DEVICE" --batch-size 8
fi

if [[ "${FORCE_EMBED_REGEN:-0}" != "1" ]] && \
   embedding_cache_complete_and_finite "data/processed/esm2_650m_benchmark.pt" "data/raw/benchmark_proteins.csv"; then
    echo "Using existing data/processed/esm2_650m_benchmark.pt"
else
    if [[ -f "data/processed/esm2_650m_benchmark.pt" ]]; then
        echo "Regenerating data/processed/esm2_650m_benchmark.pt (cache is incomplete or contains non-finite embeddings)"
    fi
    "$REPRO_PYTHON" scripts/data/extract_esm2_embeddings.py \
        --input data/raw/benchmark_proteins.csv \
        --output data/processed/esm2_650m_benchmark.pt \
        --model "$ESM2_650M" --device "$DEVICE" --batch-size 4
fi

echo "=== Step 5: Validate embedding coverage ==="
"$REPRO_PYTHON" - <<'PY'
import pandas as pd
import torch
from pathlib import Path

dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
clean_prots = set(dtc[~dtc.uniprot_id.str.contains(",", na=False)].uniprot_id.unique())

for emb_path in ["data/processed/esm2_35m_dtc.pt", "data/processed/esm2_650m_dtc.pt"]:
    if not Path(emb_path).exists():
        continue
    emb = torch.load(emb_path, map_location="cpu", weights_only=False)
    emb_prots = set(emb.keys())
    missing = clean_prots - emb_prots
    coverage = len(clean_prots & emb_prots) / len(clean_prots) * 100
    print(f"{emb_path}: {len(emb_prots)} proteins, {coverage:.1f}% DTC coverage")
    if missing:
        print(f"  WARNING: {len(missing)} DTC proteins missing from embeddings!")
        print(f"  This will reduce training data and may affect reproducibility.")
        print(f"  To fix: ensure data/raw/dtc_proteins.csv has sequences for all DTC proteins,")
        print(f"  or copy the original esm2_35m_dtc_proteins_full.pt to data/processed/esm2_35m_dtc.pt")
PY

echo "=== Data preparation complete ==="
