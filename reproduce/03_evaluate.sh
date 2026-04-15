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

# --- DeepDTA / ESM-DTA baselines ---
if [[ -n "$DAVIS_BENCHMARK" ]]; then
    echo ""
    echo "=== Evaluating DeepDTA + ESM-DTA baselines on Davis ==="
    "$REPRO_PYTHON" - "$DAVIS_BENCHMARK" <<'BASELINES'
import sys, json, logging, random, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Davis ---
davis = pd.read_csv(sys.argv[1])
log.info(f"Davis: {len(davis)} interactions, {davis.protein_name.nunique()} proteins")

# --- SMILES encoder (shared) ---
CHARSMILES = {c: i+1 for i, c in enumerate("CNOSFClBrIPHcs()[]=@+\\/#-1234567890")}
def enc_smi(s, ml=200):
    return [CHARSMILES.get(c, 0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

CHARPROT = {c: i+1 for i, c in enumerate("ACBEDGFIHKMLONQPSRUTWVYXZ")}
def enc_prot(s, ml=1000):
    return [CHARPROT.get(c, 0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

# --- DeepDTA model ---
class DeepDTA(nn.Module):
    def __init__(self):
        super().__init__()
        self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
        self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
        self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
        self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
        self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1)
        self.do = nn.Dropout(0.1)
    def forward(self, smi_tok, prot_tok):
        s = self.smiles_embed(smi_tok).transpose(1,2)
        s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s))
        s = F.adaptive_max_pool1d(s, 1).squeeze(-1)
        p = self.protein_embed(prot_tok).transpose(1,2)
        p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p))
        p = F.adaptive_max_pool1d(p, 1).squeeze(-1)
        x = torch.cat([s, p], dim=1)
        x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)

# --- ESM-DTA model ---
from anchor_transfer.model.esm_dta import EsmDTAModel

# --- CI function ---
def ci_fn(y, f):
    y, f = np.array(y), np.array(f)
    n = len(y)
    if n < 2: return 0.5
    idx = np.triu_indices(n, k=1); i, j = idx[0], idx[1]
    dt = y[i] - y[j]; dp = f[i] - f[j]; t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

# --- Evaluate each baseline ---
esm2_35m = None
results = []
for model_name, model_path in [("DeepDTA", "models/deepdta_dtc/best_model.pt"),
                                 ("ESM-DTA", "models/esm_dta_dtc/best_model.pt")]:
    if not Path(model_path).exists():
        log.info(f"Skipping {model_name} (checkpoint not found)")
        continue
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if model_name == "DeepDTA":
        model = DeepDTA().to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        log.info(f"Loaded {model_name} (epoch {ckpt.get('epoch', '?')})")
        preds = []
        for _, row in davis.iterrows():
            smi_t = torch.tensor([enc_smi(row.drug_smiles)]).to(device)
            prot_t = torch.tensor([enc_prot(row.protein_sequence)]).to(device)
            with torch.no_grad():
                preds.append(model(smi_t, prot_t).item())
    else:
        if esm2_35m is None:
            esm2_35m = torch.load("data/processed/esm2_35m_dtc.pt", map_location="cpu", weights_only=False)
            bench_emb = torch.load("data/processed/esm2_35m_benchmark.pt", map_location="cpu", weights_only=False)
            esm2_35m.update(bench_emb)
        model = EsmDTAModel(esm2_dim=480).to(device)
        model.load_state_dict(ckpt["model_state_dict"]); model.eval()
        log.info(f"Loaded {model_name} (epoch {ckpt.get('epoch', '?')})")
        preds = []
        for _, row in davis.iterrows():
            uid = row.protein_name
            if uid not in esm2_35m:
                preds.append(np.nan); continue
            smi_t = torch.tensor([enc_smi(row.drug_smiles)]).to(device)
            prot_emb = esm2_35m[uid].unsqueeze(0).to(device)
            with torch.no_grad():
                preds.append(model(smi_t, prot_emb).item())

    davis[f"{model_name}_pred"] = preds
    valid = davis[~davis[f"{model_name}_pred"].isna()]
    t, p = valid.pki.values, valid[f"{model_name}_pred"].values
    ci = ci_fn(t, p)
    rmse = np.sqrt(np.mean((t - p) ** 2))
    from sklearn.metrics import roc_auc_score
    binary = (t >= 7.0).astype(int)
    auroc = roc_auc_score(binary, p) if binary.sum() > 0 and binary.sum() < len(binary) else 0
    r = np.corrcoef(t, p)[0, 1] if len(t) > 1 else 0
    log.info(f"{model_name:20s} CI={ci:.4f} RMSE={rmse:.4f} AUROC={auroc:.4f} r={r:.4f} n={len(valid)}")

log.info("=== Baseline evaluation complete ===")
BASELINES
fi

echo ""
echo "=== All evaluations complete ==="
echo "Results saved under results/"
