# Anchor Transfer DTA

Anchor Transfer DTA predicts a query protein-drug interaction by conditioning on
an **anchor protein** — a protein already known to bind the same drug. Instead of
predicting binding affinity from scratch, the model compares how a drug interacts
with the anchor versus the query, exploiting the anchor's known binding signal as
a reference point.

## Core Idea

For a query protein *Q* and drug *D*, the model selects an anchor protein *A*
that is a known strong binder of *D* (pKi >= 7). It then learns a function
*f(A, Q, D) -> pKi* that predicts the query's binding affinity by comparing the
anchor-drug and query-drug interaction patterns through a shared encoder. This
anchor-conditioned design enables strong cross-dataset generalization: train on
DrugTargetCommons, evaluate on Davis, Metz, GLASS, or BindingDB without
retraining.

## Repository Structure

```text
src/anchor_transfer/              # Core package
  model/
    anchor_transfer.py            # V1 baseline (projection + concat)
    anchor_transfer_v2.py         # V2 main model (triple cross-attention)
    anchor_drugban.py             # DrugBanAnchor (bilinear attention + anchor)
    concise_anchor_bilinear.py    # ConciseAnchor-Bilinear (paper model)
    concise_anchor.py             # ConciseAnchor base (FSQ + cross-attention)
    concise_anchor_v3.py          # ConciseAnchor-V3 (conditional + bilinear)
    concise_anchor_cond.py        # ConciseAnchor-Cond
    drugban.py                    # DrugBAN baseline (no anchor)
    concise_dta.py                # CoNCISE baseline (no anchor)
    esm_dta.py                    # ESM-DTA baseline (DeepDTA + ESM-2)
    drug_encoder.py               # GIN molecular graph encoder
    conplex.py                    # ConPlex baseline
  data/
    dtc_loader.py                 # DTC data loading + splits
    esm_encoder.py                # ESM-2 embedding extraction

scripts/
  train/                          # Training scripts
    train_anchor_transfer.py      # Train V1/V2 on DTC
    train_anchor_drugban.py       # Train DrugBanAnchor
    train_concise_anchor_bdb.py   # Train ConciseAnchor on BindingDB
    train_eval_concise_dtc.py     # Train + eval ConciseAnchor on DTC
    ...
  eval/                           # Evaluation scripts
    evaluate_anchor_transfer_davis_paper.py  # Paper Davis protocol
    evaluate_anchor_transfer.py   # Generic evaluator
    eval_bdb_to_davis.py          # BDB → Davis cross-dataset
    eval_bdb_to_glass.py          # BDB → GLASS cross-dataset
    eval_knn_prot_only.py         # Protein-kNN baselines
    ...
  plot/                           # Plotting and figure generation
  data/                           # Data preparation + embedding extraction
  compare/                        # Model comparison scripts
  archive/                        # Experimental/temporary scripts
  drugban_paper/                  # DrugBAN paper replication subflow

reproduce/                        # Numbered reproduction pipeline
  00_setup.sh                     # Environment setup (venv, dependencies)
  00_fetch_artifacts.sh           # Download from Zenodo (auto)
  01_prepare_data.sh              # Filter DTC + extract ESM-2 embeddings
  02_train.sh                     # Train V1/V2 models
  02b_train_drugban.sh            # Train DrugBanAnchor
  03_evaluate.sh                  # Evaluate (paper Davis protocol)
  04_build_paper.sh               # Compile LaTeX
  05_train_bdb.sh                 # Train ConciseAnchor on BindingDB
  06_evaluate_bdb_cross_dataset.sh
  07_eval_moodeng.sh
  08_knn_baselines_dtc.sh         # kNN baselines vs ConciseAnchor

pyproject.toml
requirements.txt
```

## Supported Models

### Anchor Transfer V1 / V2

The primary models. V1 is a baseline MLP that concatenates projected anchor,
query, and drug embeddings. V2 replaces this with **triple bidirectional
cross-MLPs** (anchor-drug, query-drug, anchor-query) that learn pairwise
interaction patterns before fusing them for prediction.

| Variant | Protein Encoder | Description |
|---------|----------------|-------------|
| `V1-35M` | ESM-2 35M | Baseline projection + concat MLP |
| `V2-35M` | ESM-2 35M | Triple cross-attention (main model) |
| `V2-650M` | ESM-2 650M | Larger-embedding variant of V2 |

### DrugBanAnchor

Extends [DrugBAN](https://doi.org/10.1093/bioinformatics/btac680) with anchor
transfer. Uses a GIN for drug molecular graphs and bilinear attention over
atom-residue pairs. The shared bilinear weight matrix *W* computes binding
contexts for both anchor and query proteins, compared via
`[anchor_ctx, query_ctx, |diff|, product]`.

- Model: `src/anchor_transfer/model/anchor_drugban.py`
- Training: `scripts/train/train_anchor_drugban.py`

### ConciseAnchor

Extends CoNCISE with anchor transfer. Encodes drugs via Morgan fingerprints
into discrete codes, then applies bilinear attention between drug codes and
Raygun protein embeddings. Compares anchor-drug and query-drug binding patterns.

| Variant | Key Change |
|---------|-----------|
| ConciseAnchor | Shared CoNCISE encoder, post-attention comparison |
| ConciseAnchor-V3 | Drug codes conditioned on anchor; bilinear attention |
| ConciseAnchor-Bilinear | Bilinear attention over drug codes (paper model) |

- Models: `src/anchor_transfer/model/concise_anchor*.py`
- Training: `scripts/train/train_concise_anchor_bdb.py`

## Quick Start

```bash
pip install -e .

# Train V2 on DTC
python scripts/train/train_anchor_transfer.py \
    --graph data/processed/esm2_35m_dtc.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/v2_35m \
    --device cuda

# Evaluate on Davis
python scripts/eval/evaluate_anchor_transfer.py \
    --model models/v2_35m/best_model.pt \
    --model-version v2 \
    --esm2 data/processed/esm2_35m_dtc.pt \
    --esm2-benchmark data/processed/esm2_35m_benchmark.pt \
    --benchmark data/raw/davis_benchmark.csv \
    --training data/processed/dtc_training_interactions.csv \
    --output-dir results/v2_35m/davis \
    --device cuda
```

## Full Reproduction

Precomputed artifacts (embeddings, interactions, model checkpoints) are
auto-downloaded from [Zenodo](https://zenodo.org/records/19453090).

```bash
# 1. Setup environment
bash reproduce/00_setup.sh

# 2. Download artifacts from Zenodo
bash reproduce/00_fetch_artifacts.sh

# 3. Prepare data (filter DTC + ESM-2 embeddings)
bash reproduce/01_prepare_data.sh

# 4. Train models
bash reproduce/02_train.sh        # V1/V2
bash reproduce/02b_train_drugban.sh  # DrugBanAnchor

# 5. Evaluate
bash reproduce/03_evaluate.sh     # Paper Davis protocol

# 6. Extended experiments
bash reproduce/05_train_bdb.sh    # ConciseAnchor on BindingDB
bash reproduce/06_evaluate_bdb_cross_dataset.sh
bash reproduce/08_knn_baselines_dtc.sh  # kNN baselines

# 7. Build paper
bash reproduce/04_build_paper.sh
```

See `reproduce/README.md` for detailed documentation of each step.

The numbered scripts install into `.venv-repro/` and auto-detect CUDA.
Set `DEVICE=cpu` to force CPU mode.

## Data Requirements

The reproduction pipeline auto-downloads most files from Zenodo. Three files
are **not** on Zenodo and must be obtained manually:

| File | Description | Source |
|------|-------------|--------|
| `data/raw/DTC_data.csv` | DrugTargetCommons bulk export | [DTC](https://drugtargetcommons.fimm.fi/) |
| `data/raw/dtc_proteins.csv` | DTC protein sequences (`uniprot_id,sequence`) | UniProt |
| `data/raw/benchmark_proteins.csv` | Benchmark protein sequences | UniProt |

Files auto-downloaded from Zenodo (placed in `embeddings_model_files/`):
- ESM-2 embeddings: `esm2_35m_dtc_proteins_full.pt`, `esm2_650m_dtc.pt`, `esm2_{35m,650m}_benchmark.pt`
- Interactions: `dtc_training_interactions.csv`, `bindingdb_interactions.csv`
- Model checkpoints: `v2_35m_best_model.pt`, `v2_650m_best_model.pt`, `anchor_drugban_dtc_best_model.pt`, `concise_anchor_bdb_best_model.pt`
- Sequences: `merged_sequences.json`

## Citation

If you use this work, please cite:

```bibtex
@article{temiz2026anchor,
  title={Anchor Transfer: Cross-Dataset Drug-Target Affinity Prediction via Anchor Protein Conditioning},
  author={Temiz, Basar},
  year={2026}
}
```
