#!/usr/bin/env bash
# Download the exact DrugBAN paper datasets from their GitHub repo.
# Creates data/drugban_paper/ with all 6 dataset/split combinations.
set -euo pipefail

cd "$(dirname "$0")/../.."  # project root

echo "=== Fetching DrugBAN paper datasets ==="
python3 -m scripts.drugban_paper.fetch_data
echo "=== Data ready ==="

# Verify
echo ""
echo "Dataset sizes (lines including header):"
for f in data/drugban_paper/*/*/*.csv; do
    echo "  $(wc -l < "$f") $f"
done
