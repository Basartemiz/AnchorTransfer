#!/usr/bin/env bash
# Shared helpers for the supported reproduction workflow.

if [[ -n "${ANCHOR_REPRO_COMMON_LOADED:-}" ]]; then
    return 0 2>/dev/null || exit 0
fi
ANCHOR_REPRO_COMMON_LOADED=1

resolve_bootstrap_python() {
    local candidate
    local candidates=()

    if [[ -n "${PYTHON_BIN:-}" ]]; then
        candidates+=("$PYTHON_BIN")
    fi
    if command -v python3.11 >/dev/null 2>&1; then
        candidates+=("$(command -v python3.11)")
    fi
    if command -v python3 >/dev/null 2>&1; then
        candidates+=("$(command -v python3)")
    fi

    for candidate in "${candidates[@]}"; do
        if "$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 11) else 1)
PY
        then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    echo "Unable to find a Python 3.11+ interpreter." >&2
    echo "Set PYTHON_BIN=/path/to/python3.11 and rerun reproduce/00_setup.sh." >&2
    return 1
}

init_repro_context() {
    REPO_ROOT="$1"
    REPRO_VENV_DIR="${REPRO_VENV_DIR:-$REPO_ROOT/.venv-repro}"
    export PYTHONNOUSERSITE=1
    export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
}

ensure_repro_venv() {
    local bootstrap_python
    bootstrap_python="$(resolve_bootstrap_python)" || return 1

    if [[ ! -x "$REPRO_VENV_DIR/bin/python" ]]; then
        echo "=== Creating local virtualenv at ${REPRO_VENV_DIR#$REPO_ROOT/} ==="
        "$bootstrap_python" -m venv "$REPRO_VENV_DIR"
    fi

    REPRO_PYTHON="$REPRO_VENV_DIR/bin/python"
    export REPRO_PYTHON
}

require_repro_venv() {
    if [[ ! -x "$REPRO_VENV_DIR/bin/python" ]]; then
        echo "Missing reproduction environment: ${REPRO_VENV_DIR#$REPO_ROOT/}/bin/python" >&2
        echo "Run bash reproduce/00_setup.sh first." >&2
        return 1
    fi

    REPRO_PYTHON="$REPRO_VENV_DIR/bin/python"
    export REPRO_PYTHON
}

default_device() {
    if [[ -n "${DEVICE:-}" ]]; then
        printf '%s\n' "$DEVICE"
        return 0
    fi
    if [[ -n "${REPRO_PYTHON:-}" ]] && [[ -x "${REPRO_PYTHON:-}" ]]; then
        if "$REPRO_PYTHON" - <<'PY' >/dev/null 2>&1
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
        then
            printf 'cuda\n'
            return 0
        fi
    fi
    printf 'cpu\n'
}

require_file() {
    local path="$1"
    if [[ ! -f "$path" ]]; then
        echo "Missing required file: ${path#$REPO_ROOT/}" >&2
        return 1
    fi
}

require_command() {
    local command_name="$1"
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        return 1
    fi
}

embedding_cache_complete_and_finite() {
    local cache_path="$1"
    local sequence_csv="$2"

    if [[ ! -f "$cache_path" ]]; then
        return 1
    fi

    "$REPRO_PYTHON" - "$cache_path" "$sequence_csv" <<'PY' >/dev/null 2>&1
import sys
import pandas as pd
import torch

cache_path, sequence_csv = sys.argv[1], sys.argv[2]
frame = pd.read_csv(sequence_csv, usecols=["uniprot_id"])
expected_ids = frame["uniprot_id"].astype(str).tolist()
embeddings = torch.load(cache_path, map_location="cpu", weights_only=False)

for protein_id in expected_ids:
    tensor = embeddings.get(protein_id)
    if tensor is None or not torch.isfinite(tensor).all():
        raise SystemExit(1)

raise SystemExit(0)
PY
}
