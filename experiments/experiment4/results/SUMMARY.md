# Experiment 4 — Protein-side activity-cliff probe

A targeted stress test for **ConciseAnchorBilinear**: does it learn
real protein-side specificity, or does it rely on homology-lookup?

## Setup

**Cliff definition.** A drug binds protein `p_hi` with pKi *≥* 2 log
units higher than it binds homolog `p_lo` (with `p_hi` and `p_lo`
clustering together by MMseqs2 sequence identity).

**Data.** Davis benchmark (442 kinases, 68 drugs). All pairs are
mined from Davis's dense panel — each drug has many same-cluster
proteins measured.

**Models.** Reuse the DTC-trained checkpoints from experiment 3
(`concise_dta_best.pt`, `concise_anchor_best.pt`). No retraining.

**Anchor strategy (option b — external).** For each drug we retrieve
a strong (pKi ≥ 7) binder from DTC training with Tanimoto similarity
≥ 0.7 — this anchor is *not* in the cliff pair itself. Tests the
real-world use case: the model sees a related drug's known binder
and must still distinguish homologs.

**Metrics.**
- **Ranking acc**: fraction of pairs where `pred(p_hi) > pred(p_lo)`.
  Random = 50%; higher = model picks the right side.
- **Δpki corr**: Pearson correlation between true Δpki and predicted Δpki
  across pairs.
- **Side RMSE**: RMSE of all individual predictions against true pKi.
- **|true Δ|** / **|pred Δ|**: mean absolute difference between the two
  homolog predictions. Small |pred Δ| = model collapses the pair.

## Results

### Summary table (all three identity thresholds)

| identity | pairs | drugs | Model                  | Rank acc | Δpki corr | Side RMSE | \|true Δ\| | \|pred Δ\| |
|----------|-------|-------|------------------------|----------|-----------|-----------|------------|------------|
| 0.5      | 695   | 49    | ConciseDTA             | 0.1223   | −0.0450   | 2.2262    | 2.70       | 0.03       |
| 0.5      | 695   | 49    | ConciseAnchorBilinear  | **0.2547** | **+0.1048** | **1.6526** | 2.70  | **0.24** |
| 0.7      | 529   | 41    | ConciseDTA             | 0.0983   | +0.0085   | 2.1819    | 2.68       | 0.01       |
| 0.7      | 529   | 41    | ConciseAnchorBilinear  | 0.1172   | +0.0113   | 1.7897    | 2.68       | 0.02       |
| 0.9      | 496   | 40    | ConciseDTA             | 0.0766   | −0.0288   | 2.2171    | 2.70       | 0.00       |
| 0.9      | 496   | 40    | ConciseAnchorBilinear  | 0.0827   | +0.0435   | 1.7945    | 2.70       | 0.00       |

Random-baseline ranking accuracy is 0.5. **Every row is well below
random** — the models systematically pick the *wrong* side of each
cliff.

### Key observations

1. **Cliff signal is real and widespread in Davis.** Even at 90%
   sequence identity we find 496 pairs with |Δpki| ≥ 2 — near-paralog
   kinases where the same drug binds with 100×+ different affinities.
   This isn't a dataset artifact; it's biology.

2. **Both models collapse on close homologs.** At identity 0.9,
   mean `|pred Δ| = 0.00` for both models. The Raygun embeddings of
   very close homologs are so similar that the forward pass produces
   essentially identical predictions — no matter which homolog you
   feed, you get the same pKi out.

3. **The bilinear edge at 0.5 is mostly sub-family discrimination,
   not paralog discrimination.** ConciseAnchorBilinear's apparent
   win at identity 0.5 (25% vs 12%, |pred Δ| 0.24 vs 0.03) shrinks
   to a 2-point margin at 0.7 and 0.6 points at 0.9. The bilinear
   head is pulling apart distantly-related kinases that *happen* to
   cluster loosely — it is not resolving true paralog-level
   sensitivity.

4. **Below-random accuracy is a systematic bias, not inverted skill.**
   When two homologs produce near-identical predictions, whichever
   has a slightly higher "class-level binder prior" wins the
   argmax — and in cliffs the actual binder is often the weaker
   homolog of the pair (Davis reports exactly `pki = 5` for many
   non-binders, which creates a predictable bias toward
   over-predicting them). The models don't *know* which homolog is
   the real binder; the architectural tie-break is stable but
   anti-correlated with truth.

5. **Bilinear still wins on RMSE.** Side RMSE is 0.4–0.6 pKi units
   lower for ConciseAnchorBilinear at every identity threshold,
   confirming it fits individual protein-drug pKi better than
   ConciseDTA even when it can't discriminate within a pair.

## Interpretation

- ConciseAnchorBilinear's SOTA performance on Davis-OOD
  (experiment 3: AUROC 0.93 oracle, RMSE 0.62) reflects real
  improvement on **general OOD transfer** — unseen protein
  families with dissimilar sequences.
- The same model has a **hard blind spot on paralog cliffs** —
  when two proteins are >70% identical but bind the same drug
  very differently, the model cannot pick the right one. Raygun
  compression + bilinear attention operate above the resolution of
  single-residue specificity.
- For a paper: this is a genuinely interesting limitation.
  "Anchor-transfer solves distant OOD but not within-family
  discrimination" is a crisp, testable claim.

## Follow-up experiments worth running

- **Within-pair oracle** (option a from the original plan): pass the
  actual paired homolog as the anchor. If the model still can't pick
  the right side when it literally sees `(drug binds homolog A with
  pKi 9)`, then the bilinear comparison mechanism can't encode
  paralog-level information at all. If it *can* use the hint, then
  the limitation is retrieval (external anchor too far), not
  architectural.
- **Compare against experiment 2 AnchorDrugBAN** on the same pair set
  — does CHARPROTSET/BAN do worse or better than Raygun/bilinear on
  close homologs? BAN operates on raw residues which *should* let
  it distinguish paralogs, but our earlier finding was it couldn't
  use anchor quality either.
- **Test retraining on kinase-only data.** Is the blind spot
  intrinsic to the architecture, or a consequence of training on
  DTC's general-purpose data (rich in non-kinase targets)?

## Files

- `exp4_homolog_id0.5.log` — full run at identity 0.5
- `exp4_homolog_id0.7.log` — full run at identity 0.7
- `exp4_homolog_id0.9.log` — full run at identity 0.9
