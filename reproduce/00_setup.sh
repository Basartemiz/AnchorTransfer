#!/usr/bin/env bash
# 00_setup.sh — Install the package and dependencies
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"

BOOTSTRAP_PYTHON="$(resolve_bootstrap_python)"
echo "=== Using bootstrap Python: $("$BOOTSTRAP_PYTHON" --version 2>&1) ==="
ensure_repro_venv

TORCH_SPEC="${REPRO_TORCH_SPEC:-}"
TORCH_INDEX_URL="${REPRO_TORCH_INDEX_URL:-}"

if [[ -z "$TORCH_SPEC" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9.]*\).*/\1/p' | head -n1)"
    case "$CUDA_VERSION" in
        12.4*)
            TORCH_SPEC="torch==2.6.0"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"
            ;;
    esac
fi

echo "=== Upgrading packaging tools ==="
"$REPRO_PYTHON" -m pip install --upgrade pip "setuptools<82" wheel

if [[ -n "$TORCH_SPEC" ]]; then
    echo "=== Installing compatible torch runtime: $TORCH_SPEC ==="
    INSTALL_TORCH_CMD=("$REPRO_PYTHON" -m pip install --upgrade)
    if [[ -n "$TORCH_INDEX_URL" ]]; then
        INSTALL_TORCH_CMD+=(--index-url "$TORCH_INDEX_URL")
        echo "Using torch index: $TORCH_INDEX_URL"
    fi
    INSTALL_TORCH_CMD+=("$TORCH_SPEC")
    "${INSTALL_TORCH_CMD[@]}"
fi

echo "=== Installing anchor-transfer-dta ==="
"$REPRO_PYTHON" -m pip install -e ".[dev]"

echo "=== Installing paper-evaluation chemistry dependency ==="
"$REPRO_PYTHON" -m pip install --upgrade rdkit

echo "=== Installing DrugBAN dependencies (torch-geometric) ==="
"$REPRO_PYTHON" -m pip install --upgrade torch-geometric

echo "=== Installing ConciseAnchor dependencies (molfeat, fair-esm, einops, lightning, biopython) ==="
"$REPRO_PYTHON" -m pip install --upgrade fair-esm molfeat einops lightning biopython

# concise-dti requires Python 3.12+. Install if available, skip with warning otherwise.
if "$REPRO_PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
    echo "=== Installing concise-dti (Python 3.12+ detected) ==="
    "$REPRO_PYTHON" -m pip install --upgrade "concise-dti>=1.0"
else
    echo "WARNING: concise-dti requires Python 3.12+. CoNCISE baseline training will be unavailable."
    echo "ConciseAnchor training and evaluation do NOT require concise-dti."
fi

echo "=== Creating directory structure ==="
mkdir -p data/raw data/processed models results

echo "=== Verifying runtime imports ==="
"$REPRO_PYTHON" - <<'PY'
import numpy
import pandas
import torch
import idrgat
from rdkit import Chem

# Optional imports — warn if missing but don't block setup
ok = ["torch " + torch.__version__, "cuda " + str(torch.cuda.is_available()), "rdkit " + Chem.rdBase.rdkitVersion]
for mod, label in [("torch_geometric", "torch-geometric"), ("esm", "fair-esm"), ("molfeat", "molfeat")]:
    try:
        __import__(mod)
        ok.append(label)
    except ImportError:
        print(f"WARNING: {label} not installed — {mod} import failed")
print("Imports OK:", ", ".join(ok))
PY

echo "=== Setup complete ==="
echo "Environment: ${REPRO_VENV_DIR#$REPO_ROOT/}"
echo "Next: run reproduce/00_fetch_artifacts.sh to bootstrap raw inputs, then reproduce/01_prepare_data.sh"
