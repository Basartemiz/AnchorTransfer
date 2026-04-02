#!/usr/bin/env bash
# 00_setup.sh — Install the package and dependencies
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== Installing anchor-transfer-dta ==="
pip install -e ".[dev]"

echo "=== Creating directory structure ==="
mkdir -p data/raw data/processed models results

echo "=== Setup complete ==="
echo "Next: download DTC data to data/raw/DTC_data.csv"
echo "  and benchmark CSVs to data/raw/"
