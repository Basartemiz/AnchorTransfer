# Anchor Transfer DTA

Anchor Transfer DTA predicts a query protein-drug interaction by conditioning on
an anchor protein that is already known to bind the same drug.

The repo is intentionally centered on one supported workflow:

1. prepare DrugTargetCommons data
2. extract ESM-2 protein embeddings
3. train anchor-transfer models
4. evaluate them on external benchmarks
5. build the paper PDF

## Repository Structure

```text
├── src/idrgat/                     # Canonical implementation
├── src/idr_gat/                    # Compatibility wrappers for legacy imports
├── scripts/
│   ├── prepare_dtc_data.py
│   ├── extract_esm2_embeddings.py
│   ├── train_anchor_transfer.py
│   ├── train_anchor_transfer_v2.py
│   ├── evaluate_anchor_transfer.py
│   └── ...                         # Supplemental analysis scripts
├── reproduce/                      # Supported numbered reproduction scripts
├── paper/                          # LaTeX manuscript + figures
└── pyproject.toml
```

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
bash reproduce/00_setup.sh
bash reproduce/01_prepare_data.sh
bash reproduce/02_train.sh
bash reproduce/03_evaluate.sh
bash reproduce/04_build_paper.sh
```

`reproduce/README.md` documents the same flow in more detail.

## Supported Models

- `V1-35M`: baseline anchor-transfer MLP
- `V2-35M`: main triple-interaction model with ESM-2 35M embeddings
- `V2-650M`: larger-embedding variant of V2

## Data Requirements

Put these files under `data/raw/` before running the numbered reproduction scripts:

| File | Description |
|------|-------------|
| `DTC_data.csv` | DrugTargetCommons bulk export with Ki/Kd values |
| `dtc_proteins.csv` | DTC proteins with `uniprot_id,sequence` |
| `benchmark_proteins.csv` | Benchmark proteins with `uniprot_id,sequence` |
| `benchmark_exclude.txt` | Optional UniProt IDs to exclude from training |
| `davis_benchmark.csv` | Davis benchmark |
| `glass_benchmark.csv` | GLASS benchmark |
| `bdb_ki_benchmark.csv` | BindingDB Ki benchmark |
| `metz_benchmark.csv` | Metz benchmark |
| `idp_benchmark.csv` | IDP benchmark |

Benchmark CSVs are expected to contain `uniprot_id`, `ligand_smiles`, `pki`,
and optionally `protein_type`. `scripts/train_single_model.py` also accepts
`--dataset-path` and `--sequence-path` overrides for benchmark files that use
common aliases such as `protein_name`, `drug_smiles`, and `pKd`.
