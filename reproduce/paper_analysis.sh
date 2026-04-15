#!/usr/bin/env bash
# paper_analysis.sh — Run supplemental analysis scripts used by the manuscript.
#
# This wrapper sits on top of the core numbered flow. It is intentionally
# defensive: some paper figures require extra baseline checkpoints or processed
# benchmark files that are not produced by reproduce/02_train.sh alone.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
RESULTS_DIR="$REPO_ROOT/results"
PAPER_FIG_DIR="$REPO_ROOT/paper/figures"
PANELS_DIR="$RESULTS_DIR/benchmark_filter_ci_panels"
mkdir -p "$RESULTS_DIR" "$PAPER_FIG_DIR" "$PANELS_DIR"

run_step() {
    local label="$1"
    shift
    echo
    echo "=== ${label} ==="
    "$@"
}

copy_if_exists() {
    local src="$1"
    local dst="$2"
    if [ -f "$src" ]; then
        cp "$src" "$dst"
        echo "Copied $(basename "$src") -> ${dst#$REPO_ROOT/}"
    else
        echo "Missing expected artifact: ${src#$REPO_ROOT/}"
    fi
}

have_all() {
    for path in "$@"; do
        if [ ! -f "$path" ]; then
            return 1
        fi
    done
    return 0
}

ensure_parent() {
    mkdir -p "$(dirname "$1")"
}

ensure_symlink() {
    local src="$1"
    local dst="$2"
    if [ -e "$dst" ]; then
        return 0
    fi
    if [ -e "$src" ]; then
        ensure_parent "$dst"
        ln -s "$src" "$dst"
        echo "Linked ${dst#$REPO_ROOT/} -> ${src#$REPO_ROOT/}"
    fi
}

merge_pt_dicts() {
    local out="$1"
    shift
    if [ -f "$out" ]; then
        return 0
    fi
    "$REPRO_PYTHON" - "$out" "$@" <<'PY'
from pathlib import Path
import sys
import torch

out = Path(sys.argv[1])
inputs = [Path(p) for p in sys.argv[2:] if Path(p).exists()]
if not inputs:
    raise SystemExit(0)

merged = {}
for path in inputs:
    data = torch.load(path, map_location="cpu", weights_only=False)
    merged.update(data)

out.parent.mkdir(parents=True, exist_ok=True)
torch.save(merged, out)
print(f"merged {len(inputs)} files -> {out}")
PY
}

CORE_MODELS=(
    "$REPO_ROOT/models/v2_dtc/best_model.pt"
    "$REPO_ROOT/models/v2_650m_dtc/best_model.pt"
)
BASELINE_MODELS=(
    "$REPO_ROOT/models/deepdta_dtc/best_model.pt"
    "$REPO_ROOT/models/conplex_dtc/best_model.pt"
    "$REPO_ROOT/models/esm_dta_dtc/best_model.pt"
)

echo "=== Core paper analyses assume reproduce/01_prepare_data.sh and reproduce/02_train.sh are already complete ==="
echo "=== Preparing compatibility aliases for analysis scripts ==="
ensure_symlink "$REPO_ROOT/models/v2_35m" "$REPO_ROOT/models/v2_dtc"
ensure_symlink "$REPO_ROOT/models/v2_650m" "$REPO_ROOT/models/v2_650m_dtc"
ensure_symlink "$REPO_ROOT/data/raw/davis_benchmark.csv" "$REPO_ROOT/data/raw/davis/davis_benchmark.csv"
ensure_symlink "$REPO_ROOT/data/processed/esm2_35m_dtc.pt" "$REPO_ROOT/data/processed/esm2_35m_dtc_proteins.pt"
ensure_symlink "$REPO_ROOT/data/processed/esm2_35m_dtc.pt" "$REPO_ROOT/data/processed/esm2_35m_dtc_proteins_full.pt"
merge_pt_dicts \
    "$REPO_ROOT/data/processed/esm2_650m_all.pt" \
    "$REPO_ROOT/data/processed/esm2_650m_dtc.pt" \
    "$REPO_ROOT/data/processed/esm2_650m_benchmark.pt"

if have_all "${CORE_MODELS[@]}"; then
    run_step "Davis realistic retrieval analysis" \
        "$REPRO_PYTHON" scripts/eval/eval_robust_davis_v2.py \
        > "$RESULTS_DIR/eval_robust_davis_v2.log" 2>&1 || {
            echo "Davis realistic retrieval analysis failed; see results/eval_robust_davis_v2.log"
            exit 1
        }
    echo "Saved log: results/eval_robust_davis_v2.log"
else
    echo "Skipping Davis realistic retrieval analysis: missing V2 checkpoints from reproduce/02_train.sh"
fi

if have_all "${BASELINE_MODELS[@]}"; then
    run_step "Filtered vs unfiltered Davis/Metz benchmark panels" \
        "$REPRO_PYTHON" scripts/plot/generate_benchmark_filter_ci_panels.py --benchmarks Davis Metz

    copy_if_exists "$PANELS_DIR/davis_unfiltered_heatmap_ci.png" "$PAPER_FIG_DIR/fig_davis_unfiltered_heatmap_ci.png"
    copy_if_exists "$PANELS_DIR/davis_filtered_heatmap_ci.png" "$PAPER_FIG_DIR/fig_davis_filtered_heatmap_ci.png"
    copy_if_exists "$PANELS_DIR/metz_unfiltered_heatmap_ci.png" "$PAPER_FIG_DIR/fig_metz_unfiltered_heatmap_ci.png"
    copy_if_exists "$PANELS_DIR/metz_filtered_heatmap_ci.png" "$PAPER_FIG_DIR/fig_metz_filtered_heatmap_ci.png"
    copy_if_exists "$PANELS_DIR/davis_unfiltered_quartile_ci_distribution.png" "$PAPER_FIG_DIR/fig_davis_unfiltered_quartile_ci_distribution.png"
    copy_if_exists "$PANELS_DIR/davis_filtered_quartile_ci_distribution.png" "$PAPER_FIG_DIR/fig_davis_filtered_quartile_ci_distribution.png"
    copy_if_exists "$PANELS_DIR/metz_unfiltered_quartile_ci_distribution.png" "$PAPER_FIG_DIR/fig_metz_unfiltered_quartile_ci_distribution.png"
    copy_if_exists "$PANELS_DIR/metz_filtered_quartile_ci_distribution.png" "$PAPER_FIG_DIR/fig_metz_filtered_quartile_ci_distribution.png"
else
    echo "Skipping Davis/Metz panel generation: missing DTC baseline checkpoints (DeepDTA, ConPlex, ESM-DTA)"
fi

if have_all "${BASELINE_MODELS[@]}" "$REPO_ROOT/data/raw/glass/glass2_reg_major.csv" "$REPO_ROOT/data/raw/glass/ligands.tsv"; then
    run_step "GLASS stress-test supplementary panels" \
        "$REPRO_PYTHON" scripts/eval/eval_glass_anchor_bins_baselines.py

    copy_if_exists "$PANELS_DIR/glass_unfiltered_restricted_anchor_bins_ci_distribution.png" "$PAPER_FIG_DIR/fig_glass_unfiltered_restricted_anchor_bins_ci_distribution.png"
    copy_if_exists "$PANELS_DIR/glass_filtered_restricted_anchor_bins_ci_distribution.png" "$PAPER_FIG_DIR/fig_glass_filtered_restricted_anchor_bins_ci_distribution.png"
    copy_if_exists "$PANELS_DIR/glass_unfiltered_restricted_anchor_bins_rmse_distribution.png" "$PAPER_FIG_DIR/fig_glass_unfiltered_restricted_anchor_bins_rmse_distribution.png"
    copy_if_exists "$PANELS_DIR/glass_filtered_restricted_anchor_bins_rmse_distribution.png" "$PAPER_FIG_DIR/fig_glass_filtered_restricted_anchor_bins_rmse_distribution.png"
else
    echo "Skipping GLASS supplementary analysis: missing GLASS raw files and/or DTC baseline checkpoints"
fi

if have_all "${BASELINE_MODELS[@]}" "$REPO_ROOT/data/processed/bindingdb_interactions.csv"; then
    run_step "BindingDB family analysis" \
        "$REPRO_PYTHON" scripts/eval/eval_bdb_family.py
else
    echo "Skipping BindingDB family analysis: missing bindingdb_interactions.csv and/or DTC baseline checkpoints"
fi

echo
echo "=== Paper analysis complete ==="
echo "Generated figures live under paper/figures/ and analysis outputs under results/."
