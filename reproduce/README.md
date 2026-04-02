# Reproduction Guide

This directory now contains the single supported reproduction path for the
current anchor-transfer repo. Older duplicate scripts that referenced missing
files and legacy `src/idr_gat` code were removed.

## Numbered Flow

```bash
bash reproduce/00_setup.sh
bash reproduce/01_prepare_data.sh
bash reproduce/02_train.sh
bash reproduce/03_evaluate.sh
bash reproduce/04_build_paper.sh
```

## What Each Step Does

1. `00_setup.sh`
   Installs the package and creates `data/`, `models/`, and `results/`.
2. `01_prepare_data.sh`
   Filters DTC interactions and extracts ESM-2 embeddings for training and
   benchmark proteins.
3. `02_train.sh`
   Trains `V1-35M`, `V2-35M`, and `V2-650M`.
4. `03_evaluate.sh`
   Runs the trained checkpoints on Davis, GLASS, BDB-Ki, and IDP benchmarks.
5. `04_build_paper.sh`
   Compiles the LaTeX paper in `paper/`.

## Models Covered by the Supported Flow

| Model | Script |
|------|--------|
| `V1-35M` | `scripts/train_anchor_transfer.py` |
| `V2-35M` | `scripts/train_anchor_transfer_v2.py` |
| `V2-650M` | `scripts/train_anchor_transfer_v2.py` |

## Required Raw Data

Place these files in `data/raw/`:

| File | Description |
|------|-------------|
| `DTC_data.csv` | DrugTargetCommons bulk export |
| `dtc_proteins.csv` | DTC proteins with sequences |
| `benchmark_proteins.csv` | Benchmark proteins with sequences |
| `benchmark_exclude.txt` | Optional training exclusion list |
| `davis_benchmark.csv` | Davis benchmark |
| `glass_benchmark.csv` | GLASS benchmark |
| `bdb_ki_benchmark.csv` | BindingDB Ki benchmark |
| `idp_benchmark.csv` | IDP benchmark |

## Outputs

- Models: `models/<model_name>/best_model.pt`
- Evaluation summaries: `results/<model_name>/<benchmark>/summary.json`
- Per-protein metrics: `results/<model_name>/<benchmark>/per_protein_results.csv`
- Paper PDF: `paper/main.pdf`
