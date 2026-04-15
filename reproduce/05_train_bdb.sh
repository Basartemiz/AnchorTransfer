#!/usr/bin/env bash
# 05_train_bdb.sh — Prepare BDB caches and train ConciseAnchor on BindingDB
#
# This reproduces the BDB-trained ConciseAnchor model used in the cross-dataset
# generalization experiments (BDB → Davis, BDB → GLASS2).
#
# Prerequisites:
#   - data/processed/bindingdb_interactions.csv (541K interactions)
#   - data/processed/merged_sequences.json
#   - fair-esm + molfeat installed (via 00_setup.sh)
#
# Outputs:
#   - results/esm2_bdb_embeddings.pt           (ESM-2 650M cache, ~7 GB)
#   - results/raygun_bdb_embeddings.pt          (Raygun cache, ~700 MB)
#   - results/concise_bdb_morgan_fp.pkl         (Morgan FP cache, ~2 GB)
#   - models/concise_anchor_bdb/best_model.pt   (ConciseAnchor-Bilinear, ~13 MB)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

require_file "data/processed/bindingdb_interactions.csv"
require_file "data/processed/merged_sequences.json"

echo "=== Step 1: Preparing BDB caches (ESM-2 → Raygun + Morgan FPs) ==="
echo "  First run computes ESM-2 650M embeddings (~30 min on GPU),"
echo "  Raygun encoding (~5 min), and Morgan fingerprints (~10 min)."
echo "  Subsequent runs reuse cached results/."
"$REPRO_PYTHON" scripts/data/prepare_bdb_caches.py

echo ""
if [[ -f "models/concise_anchor_bdb/best_model.pt" ]] && [[ "${FORCE_RETRAIN:-0}" != "1" ]]; then
    echo "=== Step 2: Using existing ConciseAnchor-Bilinear checkpoint ==="
    echo "  models/concise_anchor_bdb/best_model.pt already exists (e.g. from Zenodo)."
    echo "  Set FORCE_RETRAIN=1 to retrain from scratch."
else
    echo "=== Step 2: Training ConciseAnchor-Bilinear on BindingDB (5 epochs) ==="
    echo "  Reuses cached Raygun embeddings and Morgan FPs."
    "$REPRO_PYTHON" scripts/train/train_concise_anchor_bdb.py
fi

echo ""
echo "=== BDB training complete ==="
echo "Model saved to: models/concise_anchor_bdb/best_model.pt"
echo "Caches: results/raygun_bdb_embeddings.pt, results/concise_bdb_morgan_fp.pkl"
