#!/usr/bin/env bash
# 08_knn_baselines_dtc.sh — Train ConciseAnchor + evaluate prot-kNN baselines on DTC
#
# Reproduces Table tab:dtc_knn in the paper:
#   1. Compute Morgan FPs and ESM-2 → Raygun embeddings (cached)
#   2. Train ConciseAnchor-Bilinear for 10 epochs on DTC cold-protein split
#   3. Evaluate prot-kNN k=1,5 on the same test set
#   4. Compare on the common anchor-covered subset (apples-to-apples)
#
# Outputs:
#   results/concise_anchor_dtc_test.csv    — ConciseAnchor predictions
#   results/prot_knn_dtc_test.csv          — prot-kNN predictions
#   results/knn_vs_concise_common_subset.csv — merged common subset
#   results/knn_vs_concise_summary.csv     — summary table
#   results/knn_vs_concise_quartiles.{png,pdf}
#   results/knn_vs_concise_overall.{png,pdf}
#   results/knn_vs_concise_scatter.{png,pdf}
#
# Prerequisites:
#   - data/processed/dtc_training_interactions.csv (or embeddings_model_files/)
#   - data/processed/merged_sequences.json
#   - embeddings_model_files/esm2_650m_dtc.pt
#   - GPU with >= 16GB VRAM (for ESM-2 + Raygun + ConciseAnchor training)
#   - pip: fair-esm einops lightning biopython rdkit torch scikit-learn matplotlib
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

echo "=========================================="
echo "  kNN baselines vs ConciseAnchor on DTC"
echo "=========================================="

# Step 1: Train ConciseAnchor (computes Raygun + Morgan FPs if not cached)
echo ""
echo "[Step 1/3] Training ConciseAnchor-Bilinear (10 epochs)..."
"$REPRO_PYTHON" -u scripts/train/train_eval_concise_dtc.py 2>&1 | tee results/concise_anchor_train_eval.log

# Step 2: Run prot-kNN baselines
echo ""
echo "[Step 2/3] Running prot-kNN k=1,5 baselines..."
"$REPRO_PYTHON" -u scripts/eval/eval_knn_prot_only.py 2>&1 | tee results/prot_knn_dtc.log

# Step 3: Compare on common subset + generate plots
echo ""
echo "[Step 3/3] Comparing on common subset + generating plots..."
"$REPRO_PYTHON" -u scripts/compare/compare_knn_vs_concise.py 2>&1 | tee results/knn_vs_concise.log

echo ""
echo "=========================================="
echo "  DONE — results in results/"
echo "=========================================="
