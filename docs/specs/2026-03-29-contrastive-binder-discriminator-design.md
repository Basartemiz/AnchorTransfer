# Contrastive Binder/Non-Binder Discriminator

**Date:** 2026-03-29
**Status:** Approved
**Branch:** alphafold

## Problem

Current anchor approach predicts absolute pKi (regression) which is sensitive to calibration — domain sequences shift the prediction scale. A contrastive model that learns to discriminate binders from non-binders avoids calibration entirely. If we can separate binders from non-binders in embedding space, we can screen drug libraries for IDPs via anchor domain similarity.

## Architecture

Existing AffinityGAT with projection head + InfoNCE loss (already implemented in `src/idr_gat/model/affinity_gat.py`):

```
Domain graph node → GATv2 encoder → anchor readout
Drug SMILES       → SMILES CNN encoder
                         ↓
              Cross-attention(protein, drug)
                         ↓
              Projection head (512 → 128, L2-norm)
                         ↓
                   InfoNCE loss
```

Pure contrastive — no MSE head, no curriculum scheduling.

## Training Data

Same `data/processed/domain_training_data.csv` (966K domain-drug pairs, 1,750 proteins):
- **Positives:** pKi ≥ 7 (Ki ≤ 100nM) — strong binders
- **Hard negatives:** pKi ≤ 5 (Ki ≥ 10μM) — weak/non-binders
- **Middle zone (5 < pKi < 7):** excluded from contrastive loss
- **In-batch negatives:** other drugs in the batch paired with different domains

## Batching

ProteinBatchSampler (already implemented in `src/idr_gat/data/protein_batch_sampler.py`):
- Group by protein/domain
- Sample k_pos positives and k_neg hard negatives per domain
- InfoNCE: positives vs (hard negatives + all other drugs in batch)

## Inference

1. For query IDP: generate conformations → Foldseek → find best anchor domain
2. Extract anchor domain graph features from the global graph
3. For each candidate drug: project (domain, drug) → 128-dim embedding
4. Score by cosine similarity
5. Rank candidates by score

## Evaluation

Same 203-protein benchmark, stratified by disorder quartile and TM quartile:

**Retrieval metrics:**
- AUROC: binder vs non-binder classification (threshold at pKi=7)
- AUPRC: precision-recall for binder detection

**Ranking metrics:**
- MRR: mean reciprocal rank of first true binder
- Hit@k (k=1,5,10): fraction of proteins where a true binder appears in top-k
- NDCG: normalized discounted cumulative gain

## Holdout

203 benchmark proteins excluded from training (same as domain-native DeepDTA).

## File Structure

```
scripts/train_contrastive_binder.py       — Training script (InfoNCE only)
scripts/evaluate_contrastive_binder.py    — Eval with retrieval + ranking metrics
```

Reuses existing:
- `src/idr_gat/model/affinity_gat.py` — AffinityGAT + projection head + infonce_loss
- `src/idr_gat/data/protein_batch_sampler.py` — ProteinBatchSampler
- `src/idr_gat/data/binding_domain.py` — Domain identification
- `data/processed/domain_training_data.csv` — Training data
