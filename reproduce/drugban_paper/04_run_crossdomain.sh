#!/usr/bin/env bash
# Run cross-domain (cluster split) and cold pair experiments.
# Models: DrugBAN, AnchorDrugBAN, DrugBAN (anchor subset), Oracle, DrugBAN (oracle subset)
# 5 seeds each, 10 epochs, patience 10.
set -euo pipefail

cd "$(dirname "$0")/../.."  # project root
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

EPOCHS=${EPOCHS:-10}
PATIENCE=${PATIENCE:-10}
SEEDS=${SEEDS:-"0,1,2,3,4"}
RESULTS_CSV="results/drugban_paper_crossdomain.csv"

echo "=== Cross-domain + cold experiments ==="
echo "Epochs: $EPOCHS, Patience: $PATIENCE, Seeds: $SEEDS"
echo "Results: $RESULTS_CSV"
echo ""

python3 -u -m scripts.drugban_paper.run_all \
    --datasets human,bindingdb,biosnap \
    --splits cold,cluster \
    --epochs "$EPOCHS" \
    --patience "$PATIENCE" \
    --seeds "$SEEDS" \
    --results-csv "$RESULTS_CSV"
