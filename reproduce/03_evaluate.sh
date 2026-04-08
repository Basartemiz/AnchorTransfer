#!/usr/bin/env bash
# 03_evaluate.sh — Evaluate paper checkpoints with the maintained Davis protocol
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
source "$REPO_ROOT/reproduce/common.sh"
init_repro_context "$REPO_ROOT"
require_repro_venv

DEVICE="$(default_device)"
RUN_GENERIC_BENCHMARKS="${RUN_GENERIC_BENCHMARKS:-0}"
BOOTSTRAP_SAMPLES="${BOOTSTRAP_SAMPLES:-1000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2048}"

if [[ -f "data/raw/davis_benchmark.csv" ]]; then
    DAVIS_BENCHMARK="data/raw/davis_benchmark.csv"
elif [[ -f "data/raw/davis/davis_benchmark.csv" ]]; then
    DAVIS_BENCHMARK="data/raw/davis/davis_benchmark.csv"
else
    DAVIS_BENCHMARK=""
fi

GENERIC_BENCHMARKS=("metz" "glass" "bdb_ki")

if [[ ! -f "data/processed/dtc_training_interactions.csv" ]] || [[ ! -f "data/processed/esm2_35m_benchmark.pt" ]] || [[ ! -f "data/processed/esm2_650m_benchmark.pt" ]] || [[ ! -f "data/raw/dtc_proteins.csv" ]] || [[ -z "$DAVIS_BENCHMARK" ]]; then
    echo "Evaluation inputs are missing. Run bash reproduce/01_prepare_data.sh first." >&2
fi

require_file "data/processed/dtc_training_interactions.csv"
require_file "data/processed/esm2_35m_benchmark.pt"
require_file "data/processed/esm2_650m_benchmark.pt"
require_file "data/raw/dtc_proteins.csv"
require_file "$DAVIS_BENCHMARK"

MODEL_NAMES=("v1_35m" "v2_35m" "v2_650m")

for model_name in "${MODEL_NAMES[@]}"; do
    case "$model_name" in
        v1_35m)
            version="v1"
            esm_train="data/processed/esm2_35m_dtc.pt"
            esm_bench="data/processed/esm2_35m_benchmark.pt"
            ;;
        v2_35m)
            version="v2"
            esm_train="data/processed/esm2_35m_dtc.pt"
            esm_bench="data/processed/esm2_35m_benchmark.pt"
            ;;
        v2_650m)
            version="v2"
            esm_train="data/processed/esm2_650m_dtc.pt"
            esm_bench="data/processed/esm2_650m_benchmark.pt"
            ;;
        *)
            echo "Unknown model alias: $model_name" >&2
            exit 1
            ;;
    esac

    if [[ ! -f "models/${model_name}/best_model.pt" ]]; then
        echo "Skipping ${model_name} (missing checkpoint: models/${model_name}/best_model.pt)"
        continue
    fi

    require_file "$esm_train"

    out_dir="results/${model_name}/davis"
    echo "=== Evaluating ${model_name} on davis (paper cross-dataset protocol) ==="
    "$REPRO_PYTHON" scripts/eval/evaluate_anchor_transfer_davis_paper.py \
        --model "models/${model_name}/best_model.pt" \
        --model-version "$version" \
        --esm2-train "$esm_train" \
        --esm2-benchmark "$esm_bench" \
        --training data/processed/dtc_training_interactions.csv \
        --benchmark "$DAVIS_BENCHMARK" \
        --dtc-proteins data/raw/dtc_proteins.csv \
        --output-dir "$out_dir" \
        --device "$DEVICE" \
        --batch-size "$EVAL_BATCH_SIZE" \
        --bootstrap-samples "$BOOTSTRAP_SAMPLES"

    if [[ "$RUN_GENERIC_BENCHMARKS" != "1" ]]; then
        continue
    fi

    for bench in "${GENERIC_BENCHMARKS[@]}"; do
        bench_csv="data/raw/${bench}_benchmark.csv"
        if [ ! -f "$bench_csv" ]; then
            echo "Skipping $bench (file not found: $bench_csv)"
            continue
        fi

        out_dir="results/${model_name}/${bench}"
        echo "=== Evaluating ${model_name} on ${bench} (generic protocol) ==="
        "$REPRO_PYTHON" scripts/eval/evaluate_anchor_transfer.py \
            --model "models/${model_name}/best_model.pt" \
            --model-version "$version" \
            --esm2 "$esm_train" \
            --esm2-benchmark "$esm_bench" \
            --benchmark "$bench_csv" \
            --training data/processed/dtc_training_interactions.csv \
            --output-dir "$out_dir" \
            --device "$DEVICE"
    done
done

# --- Homolog-filtered evaluation (novel proteins only) ---
if command -v mmseqs >/dev/null 2>&1 && [[ -n "$DAVIS_BENCHMARK" ]]; then
    echo ""
    echo "=== Computing homolog filtering (MMseqs2, >=50% identity) ==="
    "$REPRO_PYTHON" - "$DAVIS_BENCHMARK" <<'PY'
import json, random, subprocess, sys
import pandas as pd
from pathlib import Path

seqs = {}
merged = Path("data/processed/merged_sequences.json")
if merged.exists(): seqs.update(json.load(open(merged)))
davis = pd.read_csv(sys.argv[1])
if "protein_sequence" in davis.columns:
    for _, r in davis.drop_duplicates("protein_name").iterrows():
        seqs[r["protein_name"]] = r["protein_sequence"]

dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
dtc_prots = sorted(set(dtc.uniprot_id) & set(seqs.keys()))
dtc_prots = [p for p in dtc_prots if "," not in p]
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots)*0.1)); nv = max(1, int(len(dtc_prots)*0.1))
train_prots = set(dtc_prots[nt+nv:])
davis_prots = set(davis["protein_name"].unique()) & set(seqs.keys())

with open("/tmp/dtc_train.fasta","w") as f:
    for p in sorted(train_prots):
        if p in seqs: f.write(f">{p}\n{seqs[p]}\n")
with open("/tmp/davis.fasta","w") as f:
    for p in sorted(davis_prots):
        if p in seqs: f.write(f">{p}\n{seqs[p]}\n")

subprocess.run(["mmseqs","createdb","/tmp/dtc_train.fasta","/tmp/dtcDB"], capture_output=True)
subprocess.run(["mmseqs","createdb","/tmp/davis.fasta","/tmp/davisDB"], capture_output=True)
import shutil; shutil.rmtree("/tmp/mmseqs_tmp", ignore_errors=True)
subprocess.run(["mmseqs","search","/tmp/davisDB","/tmp/dtcDB","/tmp/resultDB","/tmp/mmseqs_tmp",
    "--min-seq-id","0.5","-c","0.8","--cov-mode","0"], capture_output=True)
subprocess.run(["mmseqs","convertalis","/tmp/davisDB","/tmp/dtcDB","/tmp/resultDB","/tmp/homologs.tsv"],
    capture_output=True)
import os; os.makedirs("results", exist_ok=True)
hits = pd.read_csv("/tmp/homologs.tsv", sep="\t", header=None)
homologs = set(hits[0].unique())
with open("results/davis_homologs_50.txt","w") as f:
    for p in sorted(homologs): f.write(p+"\n")
print(f"Homologs (>=50%): {len(homologs)}, Novel: {len(davis_prots - homologs)}")
PY

    if [[ -f "results/davis_homologs_50.txt" ]]; then
        for model_name in "${MODEL_NAMES[@]}"; do
            if [[ ! -f "models/${model_name}/best_model.pt" ]]; then continue; fi
            case "$model_name" in
                v1_35m) version="v1"; esm_train="data/processed/esm2_35m_dtc.pt"; esm_bench="data/processed/esm2_35m_benchmark.pt";;
                v2_35m) version="v2"; esm_train="data/processed/esm2_35m_dtc.pt"; esm_bench="data/processed/esm2_35m_benchmark.pt";;
                v2_650m) version="v2"; esm_train="data/processed/esm2_650m_dtc.pt"; esm_bench="data/processed/esm2_650m_benchmark.pt";;
            esac
            echo "=== Evaluating ${model_name} on Davis NOVEL proteins (<50% identity) ==="
            "$REPRO_PYTHON" scripts/eval/evaluate_anchor_transfer_davis_paper.py \
                --model "models/${model_name}/best_model.pt" \
                --model-version "$version" \
                --esm2-train "$esm_train" \
                --esm2-benchmark "$esm_bench" \
                --training data/processed/dtc_training_interactions.csv \
                --benchmark "$DAVIS_BENCHMARK" \
                --dtc-proteins data/raw/dtc_proteins.csv \
                --output-dir "results/${model_name}/davis_novel50" \
                --device "$DEVICE" \
                --homolog-exclude results/davis_homologs_50.txt \
                --batch-size "$EVAL_BATCH_SIZE"
        done
    fi
else
    echo "Skipping homolog filtering (mmseqs2 not installed). Install with: apt install mmseqs2"
fi

# --- AnchorDrugBAN evaluation ---
if [[ -f "models/anchor_drugban_dtc/best_model.pt" ]] && [[ -n "$DAVIS_BENCHMARK" ]]; then
    echo ""
    echo "=== Evaluating DrugBAN + AnchorDrugBAN on Davis ==="
    "$REPRO_PYTHON" scripts/eval/eval_new_models_davis.py || echo "DrugBAN eval skipped (missing dependencies or baseline model)"
fi

echo ""
echo "=== All evaluations complete ==="
echo "Results saved under results/"
