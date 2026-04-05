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
├── src/idrgat/                     # Canonical implementation
│   ├── model/
│   │   ├── anchor_transfer.py      # V1 baseline (projection + concat)
│   │   ├── anchor_transfer_v2.py   # V2 main model (triple cross-attention)
│   │   ├── anchor_drugban.py       # DrugBanAnchor
│   │   ├── concise_anchor.py       # ConciseAnchor
│   │   ├── concise_anchor_v3.py    # ConciseAnchor-V3 (conditional + bilinear)
│   │   ├── concise_anchor_cond.py  # ConciseAnchor-Cond
│   │   ├── concise_anchor_bilinear.py
│   │   ├── drugban.py              # DrugBAN baseline (no anchor)
│   │   ├── concise_dta.py          # CoNCISE baseline (no anchor)
│   │   └── esm_dta.py              # ESM-DTA baseline (DeepDTA + ESM-2)
│   └── data/                       # Data loading utilities
├── src/idr_gat/                    # Compatibility wrappers for legacy imports
├── scripts/
│   ├── prepare_dtc_data.py
│   ├── extract_esm2_embeddings.py
│   ├── train_anchor_transfer_v2.py # Train V2 on DTC
│   ├── train_anchor_drugban.py     # Train DrugBanAnchor on DTC
│   ├── train_concise_anchor_bdb.py # Train ConciseAnchor on BindingDB
│   ├── evaluate_anchor_transfer.py
│   └── ...                         # 40+ eval & analysis scripts
├── reproduce/                      # Numbered reproduction pipeline
├── paper/                          # LaTeX manuscript + figures
└── pyproject.toml
```

## Supported Models

### Anchor Transfer V1 / V2

The primary models of this project. V1 is a baseline MLP that concatenates
projected anchor, query, and drug embeddings. V2 replaces this with **triple
bidirectional cross-MLPs** (anchor-drug, query-drug, anchor-query) that learn
pairwise interaction patterns before fusing them for prediction.

| Variant | Protein Encoder | Description |
|---------|----------------|-------------|
| `V1-35M` | ESM-2 35M | Baseline projection + concat MLP |
| `V2-35M` | ESM-2 35M | Triple cross-attention (main model) |
| `V2-650M` | ESM-2 650M | Larger-embedding variant of V2 |

### DrugBanAnchor

Extends [DrugBAN](https://doi.org/10.1093/bioinformatics/btac680) (Bai et al., 2023)
with anchor transfer. DrugBAN uses a GIN to encode drug molecular graphs and a
1D-CNN for protein sequences, then applies **bilinear attention** over atom-residue
pairs to learn fine-grained binding patterns. DrugBanAnchor adds a shared bilinear
weight matrix *W* that computes atom-level binding contexts for both the anchor and
query proteins. For each drug atom, the model computes:

- anchor context: `softmax(atom @ W @ anchor_residues^T) @ anchor_residues`
- query context: `softmax(atom @ W @ query_residues^T) @ query_residues`

These are compared via `[anchor_ctx, query_ctx, |diff|, product]` and pooled to
predict pKi. The shared *W* ensures both proteins are scored through the same
binding lens.

- Model: `src/idrgat/model/anchor_drugban.py`
- Training: `scripts/train_anchor_drugban.py`
- Non-anchor baseline: `src/idrgat/model/drugban.py`

### ConciseAnchor

Extends [CoNCISE](https://github.com/BIMSBbioinfo/concise) with anchor transfer.
CoNCISE encodes drugs via Morgan fingerprints quantized through FSQ (Finite Scalar
Quantization) into a small set of discrete codes, then applies cross-attention
between drug codes and protein embeddings to model binding interactions.
ConciseAnchor runs this shared encoder for both the anchor-drug and query-drug
pairs, then compares the resulting drug and protein representations:

`[anchor_drug, query_drug, |diff|, product, anchor_prot, query_prot] -> MLP -> pKi`

Several variants explore different conditioning strategies:

| Variant | Key Change |
|---------|-----------|
| ConciseAnchor | Shared CoNCISE encoder, post-attention comparison |
| ConciseAnchor-V3 | Drug codes conditioned on anchor; bilinear attention |
| ConciseAnchor-Cond | Anchor-conditioned drug codes + full cross-attention |
| ConciseAnchor-Bilinear | Bilinear attention over CoNCISE codes |

- Models: `src/idrgat/model/concise_anchor*.py`
- Training: `scripts/train_concise_anchor_bdb.py`
- Non-anchor baseline: `src/idrgat/model/concise_dta.py`

## Quick Start

```bash
pip install -e .

python scripts/train_anchor_transfer_v2.py \
    --graph data/processed/esm2_35m_dtc.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/v2_35m \
    --device cuda

python scripts/evaluate_anchor_transfer.py \
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

```bash
export PYTHON_BIN=/path/to/python3.11  # only if python3 is not 3.11+
bash reproduce/00_setup.sh
bash reproduce/00_fetch_artifacts.sh
bash reproduce/01_prepare_data.sh
bash reproduce/02_train.sh
bash reproduce/03_evaluate.sh
bash reproduce/paper_analysis.sh
bash reproduce/04_build_paper.sh
```

`reproduce/README.md` documents the same flow in more detail.

The numbered scripts install into `.venv-repro/` and use that interpreter for
subsequent steps. If `DEVICE` is unset, they default to `cuda` only when an
NVIDIA GPU is available; otherwise they run on `cpu`. `reproduce/00_setup.sh`
also auto-selects a CUDA-12.4-compatible torch build on hosts where the default
latest wheel would be too new for the installed driver.

## Data Requirements

Put these files under `data/raw/` before running the numbered reproduction scripts:

| File | Description |
|------|-------------|
| `DTC_data.csv` | DrugTargetCommons bulk export with Ki/Kd values |
| `dtc_proteins.csv` | DTC proteins with `uniprot_id,sequence` |
| `benchmark_proteins.csv` | Benchmark proteins with `uniprot_id,sequence` |
| `benchmark_exclude.txt` | Optional UniProt IDs to exclude from training |
| `davis_benchmark.csv` | Davis benchmark |
| `metz_benchmark.csv` | Metz benchmark |
| `glass_benchmark.csv` | GLASS benchmark |
| `bdb_ki_benchmark.csv` | BindingDB Ki benchmark |

`idp_benchmark.csv` is not part of the numbered `reproduce/` flow. Keep it only
if you plan to run separate IDP-focused analyses from `scripts/`.

`reproduce/00_fetch_artifacts.sh` now makes the peer-review path explicit:
it searches common local roots for exact filenames, downloads `DTC_data.csv`
when needed, and generates `dtc_proteins.csv` plus `benchmark_proteins.csv`.
It also reuses an exact precomputed `data/processed/dtc_training_interactions.csv`
when one exists, so `reproduce/01_prepare_data.sh` can skip the raw-DTC
filtering step if the public export format is incomplete.
The derived embedding files such as `data/processed/esm2_35m_dtc.pt` and
`data/processed/esm2_650m_benchmark.pt` are then produced by
`reproduce/01_prepare_data.sh`.

Benchmark CSVs are expected to contain `uniprot_id`, `ligand_smiles`, `pki`,
and optionally `protein_type`. `scripts/train_single_model.py` also accepts
`--dataset-path` and `--sequence-path` overrides for benchmark files that use
common aliases such as `protein_name`, `drug_smiles`, and `pKd`.
