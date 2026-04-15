# Reproduction Guide

This directory now contains the supported core reproduction path for the current
anchor-transfer repo. Older duplicate scripts that referenced missing files and
legacy `src/idr_gat` code were removed.

The numbered flow reproduces the maintained V1/V2 training path and now uses
the paper Davis cross-dataset evaluation protocol by default. That protocol is
explicit in the numbered flow: canonical drug-duplicate exclusion, chirality-aware
Morgan/Tanimoto retrieval from the DTC training pool only, and per-protein
CI/AUROC/AUPRC/RMSE summaries. Supplemental analyses that go beyond this core
path remain in the `scripts/` directory.

## Numbered Flow

```bash
# Core V1/V2 pipeline
bash reproduce/00_setup.sh
bash reproduce/00_fetch_artifacts.sh
bash reproduce/01_prepare_data.sh
bash reproduce/02_train.sh               # V1-35M, V2-35M, V2-650M
bash reproduce/03_evaluate.sh            # Davis paper protocol

# DrugBanAnchor (bilinear attention + anchor transfer)
bash reproduce/02b_train_drugban.sh      # AnchorDrugBAN on DTC

# ConciseAnchor (BDB cross-dataset)
bash reproduce/05_train_bdb.sh           # Raygun/FP caches + ConciseAnchor-Bilinear
bash reproduce/06_evaluate_bdb_cross_dataset.sh

# Paper generation
bash reproduce/paper_analysis.sh
bash reproduce/04_build_paper.sh
```

## What Each Step Does

1. `00_setup.sh`
   Creates `.venv-repro/`, installs the package and all dependencies
   (`torch-geometric`, `rdkit`, `fair-esm`, `molfeat`, `einops`, `lightning`),
   and creates `data/`, `models/`, and `results/`.
2. `00_fetch_artifacts.sh`
   Reuses exact artifact filenames from local search roots when available,
   downloads the public DTC bulk export when needed, and generates
   `dtc_proteins.csv` plus `benchmark_proteins.csv` explicitly for reviewers.
   It also reuses precomputed `dtc_training_interactions.csv`, ESM `.pt` caches,
   and BDB artifacts (`bindingdb_interactions.csv`, `merged_sequences.json`)
   when exact copies already exist in a search root.
3. `01_prepare_data.sh`
   Filters DTC interactions and extracts ESM-2 embeddings for training and
   benchmark proteins. Validates embedding coverage against DTC interactions
   and warns if proteins are missing.
4. `02_train.sh`
   Trains `V1-35M`, `V2-35M`, and `V2-650M` with strongest-binder anchors
   (100 epochs, patience 20, LR 1e-3, 80/10/10 protein split).
5. `02b_train_drugban.sh`
   Trains AnchorDrugBAN on DTC. Builds molecular graph cache on first run.
6. `03_evaluate.sh`
   Runs the explicit paper Davis protocol for all trained checkpoints with
   anchor pKi quartile breakdown (Q1-Q4). Set `RUN_GENERIC_BENCHMARKS=1`
   if you also want the older generic evaluator on Metz/GLASS/BDB-Ki CSVs.
7. `05_train_bdb.sh`
   Prepares BDB caches (ESM-2 650M → Raygun embeddings + Morgan FPs) and
   trains ConciseAnchor-Bilinear on BindingDB (5 epochs).
8. `06_evaluate_bdb_cross_dataset.sh`
   Evaluates BDB-trained models on Davis and GLASS2 with drug/protein overlap
   exclusion and Tanimoto anchor retrieval.
9. `paper_analysis.sh`
   Runs the manuscript-specific supplemental analyses and copies generated PNGs
   into `paper/figures/` using the filenames included by the LaTeX source.
10. `04_build_paper.sh`
    Compiles the LaTeX paper in `paper/`.

## Models Covered by the Supported Flow

| Model | Training Script | Reproduce Step |
|------|--------|--------|
| `V1-35M` | `scripts/train/train_anchor_transfer.py` | `02_train.sh` |
| `V2-35M` | `scripts/train/train_anchor_transfer_v2.py` | `02_train.sh` |
| `V2-650M` | `scripts/train/train_anchor_transfer_v2.py` | `02_train.sh` |
| `AnchorDrugBAN` | `scripts/train/train_anchor_drugban.py` | `02b_train_drugban.sh` |
| `ConciseAnchor-Bilinear` | `scripts/train/train_concise_anchor_bdb.py` | `05_train_bdb.sh` |

## Required Raw Data

`reproduce/00_fetch_artifacts.sh` can create or reuse the first three files
below. It also reuses `data/processed/dtc_training_interactions.csv` and any
exact precomputed ESM `.pt` files it finds in a search root. `01_prepare_data.sh`
will skip the raw DTC filtering step when `data/processed/dtc_training_interactions.csv`
already exists, unless you set `FORCE_DTC_PREP=1`.

Place or bootstrap these files in `data/raw/`:

| File | Description |
|------|-------------|
| `DTC_data.csv` | DrugTargetCommons bulk export |
| `dtc_proteins.csv` | DTC proteins with sequences |
| `benchmark_proteins.csv` | Benchmark proteins with sequences |
| `benchmark_exclude.txt` | Optional training exclusion list |
| `davis_benchmark.csv` | Davis benchmark |
| `metz_benchmark.csv` | Metz benchmark |
| `glass_benchmark.csv` | GLASS benchmark |
| `bdb_ki_benchmark.csv` | BindingDB Ki benchmark |

`benchmark_exclude.txt` is optional. If it is absent, `01_prepare_data.sh`
continues without exclusion filtering.

By default, `00_fetch_artifacts.sh` searches `$HOME/Desktop/IDP work` and
`$HOME/Downloads` for exact artifact filenames before it downloads or generates
anything. Override this with `ARTIFACT_SEARCH_ROOTS=/path/one:/path/two`.

## Precomputed Embeddings (Recommended)

For exact reproducibility, use the precomputed ESM-2 embeddings from Zenodo
rather than regenerating them. Regenerated embeddings may differ slightly due to
floating-point differences across torch/CUDA versions, which can cause up to
~0.04 CI difference on downstream metrics.

Download from: https://zenodo.org/records/19481471

| File | Size | Description |
|------|------|-------------|
| `esm2_35m_dtc_proteins_full.pt` | 6.6 MB | ESM-2 35M embeddings for 3116 DTC proteins |
| `esm2_35m_benchmark.pt` | 439 KB | ESM-2 35M embeddings for benchmark proteins |
| `esm2_650m_dtc.pt` | ~33 MB | ESM-2 650M embeddings for DTC proteins |
| `esm2_650m_benchmark.pt` | ~5 MB | ESM-2 650M embeddings for benchmark proteins |
| `bindingdb_interactions.csv` | 54 MB | Preprocessed BindingDB Ki interactions |
| `merged_sequences.json` | 11 MB | Protein sequences for DTC + BDB proteins |

Place these in `data/processed/` before running `01_prepare_data.sh`. The script
will detect existing embeddings and skip regeneration. If you regenerate from
scratch, `01_prepare_data.sh` will validate coverage and warn about any missing
proteins.

`00_fetch_artifacts.sh` automatically searches local roots for these files
(including the legacy filename `esm2_35m_dtc_proteins_full.pt`) and copies them
into `data/processed/` when found.

## Runtime Notes

- The numbered scripts use the local virtualenv at `.venv-repro/` created by
  `reproduce/00_setup.sh`.
- `reproduce/03_evaluate.sh` requires `data/raw/dtc_proteins.csv`
  for the Davis self-anchor check used by the paper protocol.
- Set `PYTHON_BIN=/path/to/python3.11` if your default `python3` is older than
  Python 3.11.
- `reproduce/00_setup.sh` automatically installs `torch==2.6.0` from the
  official `cu124` index on hosts where `nvidia-smi` reports CUDA 12.4.
  Override this with `REPRO_TORCH_SPEC` and `REPRO_TORCH_INDEX_URL` if needed.
- `reproduce/00_setup.sh` also installs `rdkit`, which is required for the
  paper Davis evaluation protocol in `03_evaluate.sh`.
- `DEVICE` defaults to `cuda` only when `torch.cuda.is_available()` succeeds;
  otherwise the wrappers default to `cpu`.
- Set `BOOTSTRAP_SAMPLES` or `EVAL_BATCH_SIZE` before `03_evaluate.sh` if you
  want to trade off runtime versus tighter confidence-interval estimates during
  verification. The default numbered path uses 1000 bootstrap samples.

## Outputs

- Models: `models/<model_name>/best_model.pt`
- Davis paper-protocol summaries: `results/<model_name>/davis/summary.json`
- Davis per-protein metrics: `results/<model_name>/davis/per_protein_results.csv`
- Davis pair-level predictions: `results/<model_name>/davis/predictions.csv`
- Paper-analysis outputs: `results/benchmark_filter_ci_panels/` and
  `paper/figures/fig_*.png`
- Paper PDF: `paper/main.pdf`

## Important Scope Note

`paper_analysis.sh` goes beyond the core V1/V2 reproduction path. It assumes
additional DTC baseline checkpoints such as `deepdta_dtc`, `conplex_dtc`, and
`esm_dta_dtc`, plus processed BindingDB/GLASS files when those sections are
available. These baseline checkpoints are available on Zenodo. Missing optional
artifacts are skipped with a clear message rather than causing the whole paper
pipeline to fail.
