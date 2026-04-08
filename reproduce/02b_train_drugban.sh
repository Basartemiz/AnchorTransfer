#!/usr/bin/env bash
# 02b_train_drugban.sh — Train AnchorDrugBAN on DTC
#
# Prerequisites:
#   - data/processed/dtc_training_interactions.csv
#   - data/processed/merged_sequences.json
#   - torch-geometric and rdkit installed (via 00_setup.sh)
#
# Outputs:
#   - models/anchor_drugban_dtc/best_model.pt  (AnchorDrugBAN, ~2 MB)
#   - data/processed/drugban_graph_cache.pt    (molecular graph cache, ~450 MB, first run)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

require_file "data/processed/dtc_training_interactions.csv"
require_file "data/processed/merged_sequences.json"

echo "=== Training AnchorDrugBAN on DTC ==="
echo "  First run builds molecular graph cache (~5 min)."
echo "  Subsequent runs reuse data/processed/drugban_graph_cache.pt."
"$REPRO_PYTHON" scripts/train/train_anchor_drugban.py

echo ""
echo "=== AnchorDrugBAN training complete ==="
echo "Model saved to: models/anchor_drugban_dtc/best_model.pt"
