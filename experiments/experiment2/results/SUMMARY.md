# Experiment 2 — DrugBAN vs AnchorDrugBAN

Faithful PyG port of upstream DrugBAN + an anchor-transfer variant
`AnchorDrugBAN` that runs the shared BANLayer twice (once for `(drug, anchor)`
and once for `(drug, query)`) and compares the pooled interaction vectors
via `[f_a, f_q, |f_a − f_q|, f_a ⊙ f_q] → MLPDecoder`.

Pipeline differences vs experiment 1:
- Drug encoder: **PyG GCN** on 9-dim atom features (replaces ESM-free SMILES
  token CNN from exp1). Upstream DrugBAN uses DGL; we swap for PyG to keep
  the project on a single graph stack.
- Protein encoder: **DrugBAN's `ProteinCNN`** on CHARPROTSET residue tokens
  (replaces ESM-2 650M embeddings from exp1). 26-vocab, max length 1200.
- Interaction head: **`BANLayer` (bilinear attention)** + `MLPDecoder`, both
  byte-for-byte from upstream DrugBAN's `ban.py` / `models.py`. The only
  change is `BatchNorm1d → LayerNorm` in MLPDecoder — BN's running
  mean/var drift badly under MSE on continuous pKi, LayerNorm is stable.
- Loss: **MSE** on continuous pKi (upstream used BCE on binary labels).

All runs: 10 epochs, AdamW lr=1e-4 wd=1e-4, batch 64, MMseqs2 OOD split
(test ≤30% identity, val ≤50% identity to train, seed=42), no
`min_target_pki` filter. AUROC: binder = pKi > 7, non-binder = pKi ≤ 5,
ambiguous dropped. `n` = proteins with both classes.

## DTC-trained (dtc_training_interactions.csv)

Sample counts: train=179,834 / val=10,192 / test=16,435 interactions across
2263 / 282 / 282 proteins. Graph cache built for all 164,831 SMILES.

| Split        | Model         | macro_CI | macro_RMSE | macro_AUROC (n)  |
|--------------|---------------|----------|------------|------------------|
| Test (std)   | DrugBAN       | 0.5285   | 1.7826     | 0.6311 (62)      |
| Test (std)   | AnchorDrugBAN | **0.5304** | **1.4686** | **0.7224 (62)** |
| Test (oracle)| DrugBAN       | **0.5348** | 1.6263   | **0.6425 (66)**  |
| Test (oracle)| AnchorDrugBAN | 0.4985   | **1.4548** | 0.6253 (66)      |
| Davis-OOD    | DrugBAN       | 0.5215   | 1.6412     | 0.5667 (99)      |
| Davis-OOD    | AnchorDrugBAN | **0.5418** | **1.3801** | **0.6674 (99)** |

On standard test and Davis-OOD, AnchorDrugBAN wins cleanly on RMSE and
AUROC. CI is closer (tied on test, +0.02 on Davis).

**Oracle anchors don't help AnchorDrugBAN here** — CI drops from 0.53 → 0.50
and AUROC from 0.72 → 0.63 when we substitute the strongest test-set
binder per drug. This is the opposite of what `AnchorTransferDTAv2`'s
residual-over-anchor-pKi architecture showed on the same test in
experiment 1. AnchorDrugBAN has no mechanism to up-weight the anchor
branch when the anchor truly binds the drug — the BCN attends to both
proteins symmetrically regardless of anchor quality.

## Standard-test quartile (AnchorDrugBAN wins every bin)

```
bin            n      DB RMSE  DB CI   | Anc RMSE  Anc CI
≤7.60        4460    1.5394  0.5254   | 1.0154   0.5797
(7.60, 8.20] 3830    1.5570  0.5321   | 1.0711   0.5910
(8.20, 8.82] 4058    1.5715  0.5261   | 1.0746   0.6171
>8.82        4087    1.6533  0.5442   | 1.1286   0.6366
```
RMSE gap ~0.5 across every quartile, CI climbs with anchor strength
(0.58 → 0.64) — the "anchor-transfer works better for strong binders"
signal is present.

## Davis-OOD quartile (widest gap at top quartile)

```
bin            n      DB RMSE  DB CI   | Anc RMSE  Anc CI
≤8.15        1680    1.5885  0.5006   | 1.5319   0.5038
(8.15, 8.70] 1575    1.6896  0.5595   | 1.3724   0.5530
(8.70, 9.29] 1575    1.7363  0.4658   | 1.3693   0.5525
>9.29        1470    1.5975  0.5413   | 1.2602   0.5881
```

## Cross-experiment comparison (DTC-trained, Davis-OOD)

| Model                      | CI     | RMSE   | AUROC  |
|----------------------------|--------|--------|--------|
| ESM-DTA (exp1)             | 0.4725 | 1.4262 | 0.4329 |
| AnchorTransfer v2 (exp1)   | 0.5257 | 1.3692 | 0.6759 |
| DrugBAN (exp2)             | 0.5215 | 1.6412 | 0.5667 |
| **AnchorDrugBAN (exp2)**   | **0.5418** | 1.3801 | 0.6674 |

AnchorDrugBAN's Davis CI (0.542) beats every other model. AUROC is tied
with AnchorTransfer v2 (0.67). Both anchor-aware models (v2 / AnchorDrugBAN)
cluster around AUROC 0.67 on Davis — the anchor signal dominates whatever
base architecture you pick. The plain baselines (ESM-DTA / DrugBAN) split
by encoder quality: DrugBAN's CNN-on-residues transfers OOD much better
than ESM-DTA's mean-pooled ESM (0.57 vs 0.43 AUROC).

## Takeaways

- **DrugBAN architecture works out of the box with PyG.** We lost upstream's
  CanonicalAtomFeaturizer (74-dim) + virtual-node-bit padding and replaced
  with the project's 9-dim atom featurizer — architecture still gives a
  competitive DTA baseline.
- **MLPDecoder needs LayerNorm for MSE regression.** Upstream BN head was
  tuned for BCE classification dynamics; drop-in reuse for MSE regression
  caused val RMSE to explode to 100+ within a few epochs (see training log
  from the killed attempt). LayerNorm fixes it with zero architectural cost.
- **AnchorDrugBAN ≈ AnchorTransfer v2 on Davis-OOD AUROC**, both around
  0.67. Different mechanisms reaching similar Davis transfer numbers:
  v2 uses `anchor_pki + delta` residual, AnchorDrugBAN uses bilinear
  attention over anchor+query with shared weights.
- **AnchorDrugBAN doesn't benefit from oracle anchors** (CI 0.53 → 0.50).
  If you want to exploit anchor-quality information, you need an explicit
  mechanism (like v2's residual) — BCN symmetry alone isn't enough.

## Files

- `dtc_train_test_oracle.log` — DTC 10-epoch DrugBAN + AnchorDrugBAN training log + test + oracle + quartiles
- `dtc_davis_eval.log` — DTC-trained Davis-OOD eval with AUROC + quartiles
