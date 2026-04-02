#!/usr/bin/env bash
# Run multi-seed training for repeated-run statistics.
# Trains each model 3 times with different seeds, collects test results.
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"

DEVICE="${1:-cuda}"
SEEDS=(42 123 7)
MODELS=("v2" "deepdta" "conplex" "esm_dta" "drug_anchor" "v2_attn")
ESM_PATH="data/processed/esm2_35m_dtc_proteins.pt"

echo "=== Multi-seed training: ${#SEEDS[@]} seeds × ${#MODELS[@]} models ==="

for model in "${MODELS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        out_dir="models/multiseed/${model}_seed${seed}"
        log_file="logs/multiseed_${model}_seed${seed}.log"
        mkdir -p "$(dirname "${log_file}")"

        if [ -f "${out_dir}/best_model.pt" ]; then
            echo "SKIP ${model} seed=${seed} (already trained)"
            continue
        fi

        echo "TRAIN ${model} seed=${seed} → ${out_dir}"
        python scripts/train_single_model.py \
            --model "${model}" \
            --dataset dtc \
            --device "${DEVICE}" \
            --seed "${seed}" \
            --esm-dim 480 \
            --esm-path "${ESM_PATH}" \
            --out-dir "${out_dir}" \
            2>&1 | tee "${log_file}"

        echo "DONE ${model} seed=${seed}"
        echo ""
    done
done

echo ""
echo "=== Collecting results ==="
python -c "
import re, glob, json
from collections import defaultdict

results = defaultdict(list)
for log in sorted(glob.glob('logs/multiseed_*.log')):
    model = log.split('multiseed_')[1].split('_seed')[0]
    seed = log.split('_seed')[1].replace('.log', '')
    with open(log) as f:
        text = f.read()
    # Extract test metrics
    m = re.search(r'TEST.*CI=([0-9.]+)\s+r=([0-9.]+)\s+AUROC=([0-9.]+)', text)
    g = re.search(r'TEST GLOBAL.*CI=([0-9.]+)\s+r=([0-9.]+)', text)
    if m and g:
        results[model].append({
            'seed': int(seed),
            'per_prot_ci': float(m.group(1)),
            'per_prot_r': float(m.group(2)),
            'per_prot_auroc': float(m.group(3)),
            'global_ci': float(g.group(1)),
            'global_r': float(g.group(2)),
        })

import numpy as np
print()
print('MODEL            | Per-prot AUROC      | Global CI           | Global r')
print('-' * 80)
for model in ['v2', 'deepdta', 'conplex', 'esm_dta', 'drug_anchor', 'v2_attn']:
    if model not in results: continue
    runs = results[model]
    aurocs = [r['per_prot_auroc'] for r in runs]
    cis = [r['global_ci'] for r in runs]
    rs = [r['global_r'] for r in runs]
    print(f'{model:16s} | {np.mean(aurocs):.3f} ± {np.std(aurocs):.3f} (n={len(runs)}) | {np.mean(cis):.3f} ± {np.std(cis):.3f} | {np.mean(rs):.3f} ± {np.std(rs):.3f}')

json.dump(dict(results), open('results/multiseed_results.json', 'w'), indent=2)
print()
print('Saved to results/multiseed_results.json')
"
