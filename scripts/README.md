# Scripts Directory

This repo now has one supported core workflow and a smaller set of supplemental
analysis scripts. Scripts that depended on deleted legacy `src/idr_gat`
infrastructure were removed.

## Core Reproduction Scripts

These are the scripts used by the numbered flow in `reproduce/`.

| Script | Purpose |
|--------|---------|
| `prepare_dtc_data.py` | Filter DrugTargetCommons interactions and write the processed CSV |
| `extract_esm2_embeddings.py` | Extract mean-pooled ESM-2 embeddings for protein sequences |
| `train_anchor_transfer.py` | Train the V1 anchor-transfer model |
| `train_anchor_transfer_v2.py` | Train the V2 anchor-transfer model |
| `evaluate_anchor_transfer.py` | Evaluate V1/V2 checkpoints on external benchmarks |

## Supplemental Model Scripts

These are still in the repo because they support side analyses or alternative
model comparisons, but they are not required for the main numbered
reproduction flow.

| Script | Purpose |
|--------|---------|
| `train_single_model.py` | Train alternate models such as DeepDTA, ConPlex, ESM-DTA, DrugBAN, Drug-Anchor, and V2 attention on DTC, BindingDB, or Metz-style CSVs |
| `run_multiseed.sh` | Multi-seed wrapper around `train_single_model.py` |
| `eval_multiseed_bootstrap.py` | Bootstrap confidence intervals for the multi-model comparison |
| `eval_v2_attn_benchmarks.py` | Evaluate the attention variant on external benchmarks |

## Supplemental Analysis Scripts

The remaining `eval_*.py`, `generate_*.py`, and embedding helper scripts are
one-off analyses around the anchor-transfer experiments. They are kept because
they only depend on the current anchor-transfer code path, but they are not the
entrypoint for normal reproduction.

If you are trying to reproduce the repo from scratch, start with
`reproduce/README.md`, not with these supplemental scripts.
