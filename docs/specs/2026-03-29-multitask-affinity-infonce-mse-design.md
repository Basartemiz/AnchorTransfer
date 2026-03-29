# Multi-Task AffinityGAT: InfoNCE + MSE with Curriculum

**Date:** 2026-03-29
**Status:** Approved
**Branch:** alphafold

## Problem

The current AffinityGAT trains with pure MSE loss on pKi regression. This ignores a large source of information: drugs that don't bind (or bind weakly) to a protein. BindingDB contains both strong binders and weak/non-binders, but MSE alone treats a prediction error on a non-binder the same as an error on a strong binder. We want the model to first learn to distinguish binders from non-binders (ranking), then refine exact affinity values (regression).

## Approach

Add a contrastive InfoNCE head alongside the existing MSE regression head. Train with curriculum scheduling that shifts from ranking-first to regression-first over training.

## Architecture

Same GATv2 graph encoder as current AffinityGAT. Two heads branch after cross-attention:

```
AlphaFold Graph (nodes=domains, edges=TM-score)
       │
  GATv2 (4 layers, 8 heads, 512-dim)
       │
  Anchor node readout (per conformation)
       │
  SMILES CNN encoder (parallel 3-kernel)
       │
  Cross-Attention (protein ↔ drug, 4 heads)
       │
  shared_repr (512-dim)
       │
  ┌────┴────┐
  │         │
Projection  Regression
Head        Head
(512→128)   (512→256→1)
L2-norm
  │         │
InfoNCE     MSE
Loss        Loss
```

### Projection Head (new)
- Linear(512, 128) + L2-normalize
- Outputs unit-norm embeddings for contrastive learning
- Shared embedding space: protein-drug pairs that bind should be close

### Regression Head (existing)
- Linear(512, 256) + ReLU + Dropout(0.3) + Linear(256, 1)
- Predicts pKi value directly
- Unchanged from current AffinityGAT

## Data Pipeline

### Source
BindingDB interactions filtered to proteins in the AlphaFold graph (via anchor lookup).

### pKi Filtering

| Category | pKi Range | Ki Range | Role |
|----------|-----------|----------|------|
| Positive | pKi ≥ 7.0 | Ki ≤ 100 nM | InfoNCE positive + MSE target |
| Middle | 5.0 < pKi < 7.0 | 100 nM < Ki < 10 μM | MSE target only (excluded from InfoNCE) |
| Hard negative | pKi ≤ 5.0 | Ki ≥ 10 μM | InfoNCE hard negative + MSE target |
| Easy negative | No measured interaction | — | InfoNCE easy negative (other drugs in batch) |

### Batch Construction

Protein-centric batches:
1. Sample a protein-anchor from the training set
2. Sample `k_pos` drugs with pKi ≥ 7.0 (default: 8)
3. Sample `k_hard_neg` drugs with pKi ≤ 5.0 (default: 8)
4. Sample `k_mid` drugs with 5.0 < pKi < 7.0 (default: 4, MSE only)
5. Total batch per protein: `k_pos + k_hard_neg + k_mid` = 20 drug-protein pairs

For InfoNCE within this batch:
- Anchor: one positive drug embedding
- Positives: other positive drug embeddings for same protein
- Negatives: hard negatives + easy negatives (drugs paired with other proteins in the mini-batch)

For MSE:
- All 20 pairs contribute (positives, middle, hard negatives all have measured pKi)

### Multi-Protein Mini-Batch

Each mini-batch contains `B` protein-anchors (default B=4), each with 20 drugs = 80 pairs total.
- InfoNCE uses cross-protein negatives: for protein A's positives, protein B/C/D's drugs are easy negatives
- This gives ~60 negatives per positive without extra sampling

## Loss Schedule (Curriculum)

Linear interpolation across three phases:

```
Phase 1 (epoch 0–50):    α=0.2, β=0.8   → ranking-focused
Phase 2 (epoch 50–100):  α=0.5, β=0.5   → balanced
Phase 3 (epoch 100–200): α=0.8, β=0.2   → regression-focused

total_loss = α * MSE_loss + β * InfoNCE_loss
```

The `α` and `β` values are linearly interpolated by epoch:
```python
progress = epoch / max_epochs  # 0 → 1
alpha = 0.2 + 0.6 * progress  # 0.2 → 0.8
beta = 0.8 - 0.6 * progress   # 0.8 → 0.2
```

### InfoNCE Details

Temperature: 0.07 (learnable, initialized to 0.07)

```python
# For each positive pair (protein_i, drug_pos):
sim_pos = cos_sim(proj_protein_i, proj_drug_pos) / temperature
sim_neg = cos_sim(proj_protein_i, proj_drug_neg_j) / temperature  # all negatives
loss = -log(exp(sim_pos) / (exp(sim_pos) + Σ exp(sim_neg_j)))
```

Hard negatives (pKi ≤ 5) are explicitly included in the denominator alongside easy negatives from other proteins in the batch.

## Inference Modes

### Retrieval Mode (new capability)
1. Pre-compute projection embeddings for all anchor nodes in graph
2. For a query IDP: find anchors → get projection embeddings
3. Cosine similarity against a drug library → rank candidates
4. Fast: no regression head needed, just dot products

### Regression Mode (existing)
1. Find anchors for protein
2. Full forward pass through cross-attention + regression head
3. TM-weighted average across anchors → predicted pKi
4. Accurate but slower

### Combined Pipeline
1. Retrieval mode to screen 100K drugs → top 1000
2. Regression mode on top 1000 → precise pKi ranking

## Training Configuration

```yaml
# Model (unchanged from AffinityGAT)
hidden_dim: 512
gat_layers: 4
gat_heads: 8
dropout: 0.3

# New: projection head
proj_dim: 128

# Optimizer
lr: 1e-4
weight_decay: 1e-4
scheduler: CosineAnnealingLR
max_epochs: 200

# Batch
proteins_per_batch: 4
pos_per_protein: 8
hard_neg_per_protein: 8
mid_per_protein: 4

# InfoNCE
temperature: 0.07  # learnable
temperature_min: 0.01
temperature_max: 0.5

# Curriculum
curriculum_mse_start: 0.2
curriculum_mse_end: 0.8

# pKi thresholds
positive_pki_threshold: 7.0
negative_pki_threshold: 5.0
pki_bounds: [3.0, 12.0]

# Early stopping
patience: 20
monitor: val_mse
```

## Evaluation

### Metrics
- **Regression:** CI (concordance index), MSE, Pearson r — same as current
- **Retrieval:** Hit@k, MRR, AUROC on binder vs non-binder classification
- **Per-protein:** breakdown by IDP vs ordered, by anchor TM quartile

### Validation Strategy
- Same protein split as current benchmark (203 proteins)
- Retrieval metrics on held-out drugs per protein
- Compare against:
  1. Current AffinityGAT (MSE only)
  2. DeepDTA-1192 baseline
  3. Anchor DTA (sequence-only DeepDTA with anchors)

## File Structure

```
alphafold/
  src/idr_gat/model/affinity_gat.py    # Add projection head + InfoNCE
  scripts/train_affinity_gat.py         # Update training loop
  scripts/evaluate_anchor_dta.py        # Add retrieval evaluation
  tests/test_affinity.py                # Test new loss computation
```

Changes are contained to existing files — no new model files needed. The projection head is a small addition to AffinityGAT, and the training loop changes are in the loss computation and batch construction.

## Key Design Decisions

1. **Single shared encoder, two heads** — the contrastive head regularizes the shared representation, making it more discriminative. The regression head fine-tunes for precise values.

2. **Curriculum over fixed weighting** — early InfoNCE-heavy training builds a good embedding space for ranking. Later MSE-heavy training refines predictions. This avoids the failure mode where MSE collapses the embedding space early on.

3. **Protein-centric batches** — ensures each protein has both positives and hard negatives in every batch. Cross-protein drugs provide abundant easy negatives for free.

4. **Middle zone excluded from InfoNCE** — drugs with moderate affinity (100nM–10μM) are ambiguous for ranking but still provide useful regression signal. Including them in InfoNCE would add noise.

5. **Learnable temperature** — lets the model control how peaked the similarity distribution is, adapting to the data.
