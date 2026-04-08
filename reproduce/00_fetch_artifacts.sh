#!/usr/bin/env bash
# 00_fetch_artifacts.sh — Download precomputed artifacts from Zenodo
#
# Zenodo record: https://zenodo.org/records/19453090
# Falls back to local search if download fails.
# Files the pipeline needs but Zenodo doesn't have → missing/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ZENODO_BASE="https://zenodo.org/records/19453090/files"

# ── Zenodo file → local path mapping ──────────────────────────────
# Format: "zenodo_filename:local_path"
ZENODO_FILES=(
    "dtc_training_interactions.csv:embeddings_model_files/dtc_training_interactions.csv"
    "merged_sequences.json:data/processed/merged_sequences.json"
    "esm2_35m_dtc_proteins_full.pt:embeddings_model_files/esm2_35m_dtc_proteins_full.pt"
    "esm2_650m_dtc.pt:embeddings_model_files/esm2_650m_dtc.pt"
    "esm2_35m_benchmark.pt:embeddings_model_files/esm2_35m_benchmark.pt"
    "esm2_650m_benchmark.pt:embeddings_model_files/esm2_650m_benchmark.pt"
    "bindingdb_interactions.csv:embeddings_model_files/bindingdb_interactions.csv"
    "v2_35m_best_model.pt:embeddings_model_files/v2_35m_best_model.pt"
    "v2_650m_best_model.pt:embeddings_model_files/v2_650m_best_model.pt"
    "anchor_drugban_dtc_best_model.pt:embeddings_model_files/anchor_drugban_dtc_best_model.pt"
    "concise_anchor_bdb_best_model.pt:embeddings_model_files/concise_anchor_bdb_best_model.pt"
)

# ── Files needed but NOT on Zenodo → tracked in missing/ ─────────
REQUIRED_NOT_ON_ZENODO=(
    "data/raw/DTC_data.csv"
    "data/raw/dtc_proteins.csv"
    "data/raw/benchmark_proteins.csv"
)

echo "=========================================="
echo "  Downloading artifacts from Zenodo"
echo "  Record: 19453090"
echo "=========================================="

mkdir -p embeddings_model_files data/processed data/raw missing

downloaded=0
skipped=0
failed=0

for entry in "${ZENODO_FILES[@]}"; do
    zenodo_name="${entry%%:*}"
    local_path="${entry#*:}"

    if [[ -f "$local_path" ]]; then
        echo "  [skip] $local_path (exists)"
        ((skipped++))
        continue
    fi

    mkdir -p "$(dirname "$local_path")"
    echo "  [download] $zenodo_name → $local_path"

    if curl -fSL --progress-bar "${ZENODO_BASE}/${zenodo_name}?download=1" -o "$local_path"; then
        ((downloaded++))
    else
        echo "  [FAILED] Could not download $zenodo_name"
        rm -f "$local_path"
        ((failed++))
    fi
done

echo ""
echo "Downloaded: $downloaded  Skipped: $skipped  Failed: $failed"

# ── Check for files not on Zenodo ─────────────────────────────────
missing_count=0
for f in "${REQUIRED_NOT_ON_ZENODO[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "$f" >> missing/not_on_zenodo.txt
        ((missing_count++))
    fi
done

if [[ $missing_count -gt 0 ]]; then
    echo ""
    echo "WARNING: $missing_count required files not found and not on Zenodo."
    echo "See missing/not_on_zenodo.txt for the list."
    echo "These must be obtained manually (e.g., from DrugTargetCommons)."
fi

# ── Fallback: try local search for any missing files ──────────────
if [[ -f "scripts/data/bootstrap_repro_artifacts.py" ]] && [[ $failed -gt 0 || $missing_count -gt 0 ]]; then
    echo ""
    echo "Attempting local search fallback..."
    SEARCH_ROOTS=()
    if [[ -n "${ARTIFACT_SEARCH_ROOTS:-}" ]]; then
        IFS=':' read -r -a SEARCH_ROOTS <<<"$ARTIFACT_SEARCH_ROOTS"
    else
        SEARCH_ROOTS=(
            "$HOME/Desktop/IDP work"
            "$HOME/Downloads"
        )
    fi
    ARGS=()
    for root in "${SEARCH_ROOTS[@]}"; do
        if [[ -d "$root" ]]; then
            ARGS+=(--search-root "$root")
        fi
    done
    if [[ ${#ARGS[@]} -gt 0 ]]; then
        python3 scripts/data/bootstrap_repro_artifacts.py \
            --repo-root "$REPO_ROOT" \
            "${ARGS[@]}" || true
    fi
fi

echo ""
echo "=========================================="
echo "  Artifact fetch complete"
echo "=========================================="
echo "Next: bash reproduce/01_prepare_data.sh"
