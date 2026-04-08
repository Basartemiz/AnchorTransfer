#!/usr/bin/env bash
# 00_fetch_artifacts.sh — Bootstrap raw artifacts and reuse cached exact files
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

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

echo "=== Bootstrapping reproduction artifacts ==="
"$REPRO_PYTHON" scripts/data/bootstrap_repro_artifacts.py \
    --repo-root "$REPO_ROOT" \
    "${ARGS[@]}" \
    "$@"

echo "=== Artifact bootstrap complete ==="
echo "Next: bash reproduce/01_prepare_data.sh"
