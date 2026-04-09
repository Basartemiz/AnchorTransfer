#!/usr/bin/env bash
# 06_evaluate_bdb_cross_dataset.sh — Evaluate BDB-trained models on Davis + GLASS2
#
# Cross-dataset evaluation: models trained on BindingDB are evaluated on
# Davis (kinase benchmark) and GLASS2 (GPCR benchmark) after excluding all
# canonical drug and protein overlap with the BDB training set.
#
# Tanimoto-nearest anchors are retrieved from BDB training set (pKi >= 7).
#
# Prerequisites:
#   - models/concise_bdb/best_model.pt
#   - models/concise_anchor_bdb/best_model.pt
#   - results/raygun_bdb_embeddings.pt (from 05_train_bdb.sh)
#   - data/raw/davis/davis_benchmark.csv
#   - data/raw/glass/glass2_ki_interactions.csv
#   - data/raw/glass/glass2_sequences.json
#
# Outputs:
#   - results/bdb_to_davis_predictions.csv   (per-interaction predictions)
#   - results/bdb_to_glass_predictions.csv   (per-interaction predictions)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

require_file "models/concise_bdb/best_model.pt"
require_file "models/concise_anchor_bdb/best_model.pt"
require_file "results/raygun_bdb_embeddings.pt"
require_file "data/raw/davis/davis_benchmark.csv"

echo "=== Evaluating BDB-trained models on Davis ==="
"$REPRO_PYTHON" scripts/eval/eval_bdb_to_davis.py

if [ -f "data/raw/glass/glass2_ki_interactions.csv" ]; then
    echo ""
    echo "=== Evaluating BDB-trained models on GLASS2 GPCRs ==="
    "$REPRO_PYTHON" scripts/eval/eval_bdb_to_glass.py
else
    echo "Skipping GLASS2 (data/raw/glass/glass2_ki_interactions.csv not found)"
    echo "Download from: https://zhanglab.comp.nus.edu.sg/GLASS/download.html"
fi

echo ""
echo "=== Generating cross-dataset plots ==="
"$REPRO_PYTHON" scripts/plot/plot_bdb_cross_dataset.py

echo ""
echo "=== Cross-dataset evaluation complete ==="
echo "Results:"
echo "  results/bdb_to_davis_predictions.csv"
echo "  results/bdb_to_glass_predictions.csv"
echo "Figures:"
echo "  paper/figures/bdb_cross_dataset/"
