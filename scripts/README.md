# Scripts Directory

All scripts in this directory are actively used by the numbered reproduction
pipeline in `reproduce/`. Start with `reproduce/README.md` for the full flow.

## Directory Layout

```text
scripts/
  data/                             # Data preparation (called by reproduce/01, 05)
  train/                            # Model training (called by reproduce/02, 02b, 05, 08)
  eval/                             # Evaluation (called by reproduce/03, 06, 08, paper_analysis)
  plot/                             # Figure generation (called by reproduce/06, paper_analysis)
  compare/                          # Model comparison (called by reproduce/08)
  drugban_paper/                    # DrugBAN paper replication subflow
```

## Data Preparation (`data/`)

| Script | Called By | Purpose |
|--------|-----------|---------|
| `prepare_dtc_data.py` | `01_prepare_data.sh` | Filter DTC interactions to clean Ki/Kd pKi values |
| `extract_esm2_embeddings.py` | `01_prepare_data.sh` | Extract mean-pooled ESM-2 embeddings |
| `prepare_bdb_caches.py` | `05_train_bdb.sh` | Compute ESM-2 -> Raygun + Morgan FP caches for BDB |
| `bootstrap_repro_artifacts.py` | `00_fetch_artifacts.sh` | Local filesystem search fallback for missing artifacts |

## Training (`train/`)

| Script | Called By | Purpose |
|--------|-----------|---------|
| `train_anchor_transfer.py` | `02_train.sh` | Train V1-35M on DTC |
| `train_anchor_transfer_v2.py` | `02_train.sh` | Train V2-35M / V2-650M on DTC |
| `train_anchor_drugban.py` | `02b_train_drugban.sh` | Train AnchorDrugBAN on DTC |
| `train_concise_anchor_bdb.py` | `05_train_bdb.sh` | Train ConciseAnchor-Bilinear on BindingDB |
| `train_concise_bdb.py` | (dependency) | CoNCISE on BDB; produces Raygun/FP caches used by anchor variant |
| `train_eval_concise_dtc.py` | `08_knn_baselines_dtc.sh` | Train + eval ConciseAnchor on DTC cold-protein split |
| `train_moodeng_anchor_eval.py` | `07_eval_moodeng.sh` | Train ConciseAnchor on MooDeng |

## Evaluation (`eval/`)

| Script | Called By | Purpose |
|--------|-----------|---------|
| `evaluate_anchor_transfer_davis_paper.py` | `03_evaluate.sh` | Paper Davis protocol (Tanimoto retrieval, per-protein CI) |
| `evaluate_anchor_transfer.py` | `03_evaluate.sh` | Generic benchmark evaluator (Metz, GLASS, BDB-Ki) |
| `eval_bdb_to_davis.py` | `06_evaluate_bdb_cross_dataset.sh` | BDB -> Davis cross-dataset evaluation |
| `eval_bdb_to_glass.py` | `06_evaluate_bdb_cross_dataset.sh` | BDB -> GLASS2 cross-dataset evaluation |
| `eval_new_models_davis.py` | `03_evaluate.sh` | DrugBAN + AnchorDrugBAN Davis eval |
| `eval_knn_prot_only.py` | `08_knn_baselines_dtc.sh` | Protein-only kNN baselines |
| `eval_robust_davis_v2.py` | `paper_analysis.sh` | Davis realistic retrieval analysis |
| `eval_bdb_family.py` | `paper_analysis.sh` | BindingDB protein family analysis |
| `eval_glass_anchor_bins_baselines.py` | `paper_analysis.sh` | GLASS supplementary panels |

## Plotting (`plot/`)

| Script | Called By | Purpose |
|--------|-----------|---------|
| `generate_benchmark_filter_ci_panels.py` | `paper_analysis.sh` | Paper CI heatmap + quartile panels |
| `plot_bdb_cross_dataset.py` | `06_evaluate_bdb_cross_dataset.sh` | Cross-dataset result figures |

## Comparison (`compare/`)

| Script | Called By | Purpose |
|--------|-----------|---------|
| `compare_knn_vs_concise.py` | `08_knn_baselines_dtc.sh` | kNN vs ConciseAnchor comparison + plots |

## Multi-seed

| Script | Purpose |
|--------|---------|
| `run_multiseed.sh` | Multi-seed wrapper for training scripts |
