#!/usr/bin/env bash
# 00_fetch_artifacts.sh — Download precomputed artifacts from Zenodo
#
# Zenodo record: https://zenodo.org/records/19481471
# Downloads embeddings, processed data, benchmark datasets, and model checkpoints.
# Falls back to local search if download fails.
# Files the pipeline needs but Zenodo doesn't have → missing/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ZENODO_RECORD="19481471"
ZENODO_BASE="https://zenodo.org/api/records/${ZENODO_RECORD}/files"

# ── Zenodo file → local path mapping ──────────────────────────────
# All 23 files from Zenodo v4 record.
# Format: "zenodo_filename:local_path"
ZENODO_FILES=(
    # --- Processed interaction data ---
    "dtc_training_interactions.csv:data/processed/dtc_training_interactions.csv"
    "bindingdb_interactions.csv:data/processed/bindingdb_interactions.csv"
    "merged_sequences.json:data/processed/merged_sequences.json"

    # --- ESM-2 precomputed embeddings ---
    "esm2_35m_dtc_proteins_full.pt:data/processed/esm2_35m_dtc.pt"
    "esm2_650m_dtc.pt:data/processed/esm2_650m_dtc.pt"
    "esm2_35m_benchmark.pt:data/processed/esm2_35m_benchmark.pt"
    "esm2_650m_benchmark.pt:data/processed/esm2_650m_benchmark.pt"

    # --- Raygun embeddings (BDB) ---
    "raygun_bdb_embeddings.pt:results/raygun_bdb_embeddings.pt"

    # --- Benchmark datasets ---
    "davis_benchmark.csv:data/raw/davis_benchmark.csv"
    "glass2_ki_interactions.csv:data/raw/glass/glass2_ki_interactions.csv"
    "glass2_sequences.json:data/raw/glass/glass2_sequences.json"

    # --- Core model checkpoints ---
    "v1_35m_best_model.pt:models/v1_35m/best_model.pt"
    "v2_35m_best_model.pt:models/v2_35m/best_model.pt"
    "v2_650m_best_model.pt:models/v2_650m/best_model.pt"
    "anchor_drugban_dtc_best_model.pt:models/anchor_drugban_dtc/best_model.pt"
    "concise_anchor_bdb_best_model.pt:models/concise_anchor_bdb/best_model.pt"

    # --- CoNCISE / Moodeng model checkpoints ---
    "concise_bdb_best_model.pt:models/concise_bdb/best_model.pt"
    "concise_moodeng_best_model.pt:models/concise_moodeng/best_model.pt"
    "concise_anchor_moodeng_best_model.pt:models/concise_anchor_moodeng/best_model.pt"

    # --- Baseline model checkpoints ---
    "deepdta_dtc_best_model.pt:models/deepdta_dtc/best_model.pt"
    "esm_dta_dtc_best_model.pt:models/esm_dta_dtc/best_model.pt"
    "conplex_dtc_best_model.pt:models/conplex_dtc/best_model.pt"
    "drugban_dtc_best_model.pt:models/drugban_dtc/best_model.pt"
)

# ── Files needed but NOT on Zenodo → tracked in missing/ ─────────
REQUIRED_NOT_ON_ZENODO=(
    "data/raw/DTC_data.csv"
    "data/raw/dtc_proteins.csv"
    "data/raw/benchmark_proteins.csv"
)

echo "=========================================="
echo "  Downloading artifacts from Zenodo"
echo "  Record: ${ZENODO_RECORD}"
echo "=========================================="

mkdir -p data/raw data/raw/glass data/processed models results missing

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

    if curl -fSL --progress-bar "${ZENODO_BASE}/${zenodo_name}/content" -o "$local_path"; then
        ((downloaded++))
    else
        echo "  [FAILED] Could not download $zenodo_name"
        rm -f "$local_path"
        ((failed++))
    fi
done

echo ""
echo "Downloaded: $downloaded  Skipped: $skipped  Failed: $failed"

# ── Symlink davis_benchmark.csv to alternate expected location ───
if [[ -f "data/raw/davis_benchmark.csv" ]] && [[ ! -e "data/raw/davis/davis_benchmark.csv" ]]; then
    mkdir -p data/raw/davis
    ln -sf "$REPO_ROOT/data/raw/davis_benchmark.csv" "data/raw/davis/davis_benchmark.csv"
    echo "  Linked data/raw/davis/davis_benchmark.csv → data/raw/davis_benchmark.csv"
fi

# ── Check for files not on Zenodo ─────────────────────────────────
rm -f missing/not_on_zenodo.txt
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
