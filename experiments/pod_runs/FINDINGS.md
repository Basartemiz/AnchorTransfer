# GPU re-run of all four experiments — 2026-08-10

All four experiments re-run on a RunPod box (RTX 2000 Ada 16 GB, 48 vCPU,
251 GB RAM; **not** an A100). Logs in `logs/`, parsed metrics in
`results/seed_summary.json` and `results/all_results.json`.

Two environments, because `concise-dti` requires Python ≥3.12 while the pod
image ships 3.11:
- `python3.11` + torch 2.4.1+cu124 → experiments 1, 2
- `/root/venv312` + torch 2.6.0+cu124 + `concise-dti` → experiments 3, 4

**Experiment 3 ran end to end for the first time** — it had been impossible
locally (Python version + a 2.7 GB Raygun download). The Raygun cache is 691 MB
covering 2826/2827 proteins.

---

## 1. The OOD split was never reproducible — fixed

`ood_split` produced a **different train/val/test split on every process**,
including different held-out proteins and a different test-set size:

```
before fix, two default runs:   test=285/b6ec51cfbc26
                                test=289/fef72071c71e
with PYTHONHASHSEED=0:          test=282/7ade7862ee7a   (identical both runs)
after fix, four runs:           test=283/6412f7a2b391   (identical all four)
```

**Cause.** `main.py` builds `set(interactions_df["uniprot_id"].unique())` and
iterates it into the sequence dict. Python randomizes string hashing per
process, so the FASTA handed to MMseqs was ordered differently every run,
changing the clustering and the `reps` order consumed by `rng.shuffle`. MMseqs
itself is deterministic (0/2827 members differed on repeat) — `--threads 1` was
doing its job. `seed=42` only ever controlled the shuffle.

**Fix** (`split.py`, all four copies are byte-identical):
`sorted(sequences.items())` when writing the FASTA, and `sorted(...)` on both
`test_clusters.keys()` and `val_clusters.keys()` before `rng.shuffle`.

**Consequence.** Every previously reported number — published tables and the
earlier local reproduction — is conditioned on an unrecorded per-process hash
seed. Any multi-seed run done before this fix conflates training-init variance
with split variance. Fixed split for all results below: `2262/282/283`.

---

## 2. Experiment 1 — 5 seeds × 2 sources

| Source | Split | Metric | ESM-DTA | AnchorMLP | Δ | t (df=4) |
|---|---|---|---|---|---|---|
| DTC | Test | AUROC | 0.7987±0.0144 | 0.8052±0.0267 | +0.007 | 0.54 |
| DTC | Test | RMSE | 1.4407±0.0205 | **1.3968±0.0310** | −0.044 | −5.67 |
| DTC | Oracle | AUROC | 0.7040±0.0107 | **0.7604±0.0252** | +0.056 | 8.17 |
| DTC | Oracle | RMSE | 1.3719±0.0275 | **1.2806±0.0234** | −0.091 | −4.70 |
| DTC | Davis-OOD | AUROC | 0.6104±0.1826 | 0.7926±0.0438 | +0.182 | 2.40 |
| DTC | Davis-OOD | RMSE | 1.3003±0.1399 | 1.1062±0.0877 | −0.194 | −2.41 |
| BDB | Test | AUROC | 0.7907±0.0166 | 0.7945±0.0226 | +0.004 | 0.31 |
| BDB | Test | RMSE | 1.4355±0.0443 | 1.3940±0.0660 | −0.042 | −1.03 |
| BDB | Oracle | CI | **0.5340±0.0056** | 0.5215±0.0063 | −0.013 | −3.28 |
| BDB | Oracle | RMSE | 1.5838±0.0357 | **1.4966±0.0276** | −0.087 | −4.88 |
| BDB | Davis-OOD | AUROC | 0.6421±0.0872 | **0.7689±0.0461** | +0.127 | 3.87 |
| BDB | Davis-OOD | CI | 0.5361±0.0189 | **0.5661±0.0144** | +0.030 | 7.54 |
| BDB | Davis-OOD | RMSE | 1.9161±0.2481 | **1.2546±0.0686** | −0.662 | −7.36 |

**The effect is specific to distribution shift.** On Davis-OOD the anchor wins
on both sources, decisively on BDB (RMSE −35%, t = −7.36). In-distribution
there is no ranking difference at all (Test AUROC t = 0.54 and 0.31), and on
BDB the Test RMSE difference is not significant either. This is a narrower but
far more defensible claim than "wins in every case".

**The anchor model is much more stable across seeds.** Baseline → anchor std:
Davis AUROC ±0.183 → ±0.044 (DTC) and ±0.087 → ±0.046 (BDB); Davis RMSE
±0.248 → ±0.069 (BDB). This is why the large Davis AUROC gaps still only reach
t ≈ 2.4–3.9: the baseline is wildly variable, not the anchor model. The
published ESM-DTA Davis AUROC of 0.433 sits about 1σ below this distribution's
mean — a low draw, consistent with it being a single-seed artifact.

## 3. Experiment 3 — DTC and BDB (single seed)

| Source | Split | CoNCISE (CI/RMSE/AUROC) | ConciseAnchor |
|---|---|---|---|
| DTC | Test | 0.5536 / 1.4202 / 0.8012 | **0.6114 / 1.3521 / 0.8504** |
| DTC | Oracle | 0.5538 / 1.3381 / 0.6949 | **0.5814 / 1.1467 / 0.7819** |
| DTC | Davis-OOD | 0.6233 / 0.7584 / 0.7964 | **0.6248 / 0.6878 / 0.8199** |
| DTC | Davis-oracle | 0.6132 / 0.7799 / 0.8090 | **0.6181 / 0.6935 / 0.8785** |
| BDB | Test | 0.5620 / 1.4291 / 0.8227 | **0.6049 / 1.3106 / 0.8244** |
| BDB | Oracle | 0.5384 / 1.4111 / 0.6727 | 0.5354 / **1.3383 / 0.6914** |
| BDB | Davis-OOD | 0.6348 / 0.7539 / 0.9084 | **0.6420 / 0.6980 / 0.9418** |
| BDB | Davis-oracle | 0.6195 / 0.8860 / 0.8637 | **0.6266 / 0.7934 / 0.8957** |

RMSE favours the anchor in all 8 rows.

**The "anchor pathway is read" argument replicates on DTC but not on BDB.**
Using CoNCISE as the control for sample-set composition, the anchor-attributable
Davis-oracle shift is **+0.046 on DTC** (anchor +0.059, control +0.013) but
**−0.001 on BDB** (anchor −0.046, control −0.045). It should not be presented as
a general property.

> **Superseded (2026-08-11).** The +0.046 is a single-seed figure. Over three
> seeds the anchor-attributable shift is **+0.014 on DTC** and **−0.002 on BDB**
> — averaging removes most of it. The oracle condition is a weak probe and the
> mechanism claim now rests on the random-anchor control instead. See §6.

**Checkpoint selection is working correctly** here — `CI − RMSE` picked the true
per-epoch optimum for both models (epoch 2 and 3). The early stop is genuine
overfitting, not the selection stall seen in the earlier local run. The valid
criticism that remains: AUROC never enters selection anywhere.

## 4. Experiment 4 — the cliff set is mostly the same protein twice

**This is the largest finding.** Davis stores point mutants under the wild-type
sequence — `KIT` and `KIT(D816V)` have byte-identical sequence fields and
different measured pKi. So the "homolog cliff" pairs are largely the same input
twice:

| identity | pairs | identical sequence |
|---|---|---|
| ≥0.5 | 758 | 530 (69.9%) |
| ≥0.7 | 564 | 530 (94.0%) |
| ≥0.9 | 530 | 530 (**100%**) |

Davis has 442 protein ids but only **379 distinct sequences**; 81 ids share a
sequence with another id.

The reported metric counts ties as failures, and the models tie on exactly
485/695 pairs — *the same count for both models*, because the cause is the
input, not the model. Dropping identical-sequence pairs takes ties to **exactly
zero**, confirming it:

| identity | pairs | CoNCISE acc | ConciseAnchor acc | CoNCISE corr | Anchor corr |
|---|---|---|---|---|---|
| ≥0.5 | 199 | 0.4372 | **0.8141** | +0.0095 | **+0.2537** |
| ≥0.7 | 33 | 0.3333 | **0.7879** | +0.2366 | **+0.5472** |
| ≥0.9 | 0 | — | — | — | — |

**The paper currently writes this experiment up as a failure** ("both models
rank below the 0.5 random baseline… a clear blind spot for differences within a
family"). Correctly constructed it is one of the strongest results in the paper:
ConciseAnchor ranks genuine activity cliffs correctly **81%** of the time with a
real Δ-correlation, while CoNCISE sits at chance. The `identity≥0.9` row should
be dropped — after filtering there is nothing left in it to measure.

## 5. Experiment 2 — inconclusive over 5 seeds

DTC, seeds 42–46, ~4 h per run. **No metric on any split separates the two
models: every |t| < 1.4.** The point estimates also disagree with each other —
AUROC leans anchor by 0.047–0.060 while RMSE favours plain on all four splits —
and single seeds reverse the ordering (Davis-OOD: one seed +0.163 AUROC for the
anchor, another −0.060).

Two protocol properties explain the instability, neither about anchoring:
- **Asymmetric checkpoint selection.** `CI − RMSE` stops improving for DrugBAN
  around epoch 2 while AnchorDrugBAN keeps improving to epoch 10, so the arms
  are not trained to comparable points.
- **Matching the epoch does not repair it.** At a common final epoch both models
  fall *below chance* on Davis (AUROC 0.333 and 0.452) — 10 epochs is past the
  useful range for this architecture.

Reported as inconclusive, not as evidence either way.

AnchorDrugBAN is slow for a structural reason: two models trained in one loop,
and the anchor model runs the shared attention twice, producing many small GPU
kernels launched from one Python thread. Adding `num_workers=12` did **not**
help — the workers sit idle at 0.2% CPU while the main process stays pegged, so
data loading was never the bottleneck.

**Caveat that belongs in the paper.** This is not upstream DrugBAN. The port
swaps DGL/dgllife for PyG (`drugban.py` docstring), uses a hand-rolled **9-dim**
atom featurizer where upstream uses dgllife's 74-dim `CanonicalAtomFeaturizer`
(`drug_encoder.py:12`), and trains MSE regression on pKi where upstream is a BCE
binary classifier. The comparison is internally controlled — both arms share the
featurizer — but the absolute level is not DrugBAN's, and the null result is
therefore not evidence about DrugBAN.

---

# Follow-up run — 2026-08-11

Logs in `logs/splits/` and `logs/retrieval/`, parsed metrics in
`results/split_summary.json`.

## 6. Retrieval-only floor — the control the baselines were missing

The anchor models get a retrieved training protein **and its measured pKi** at
test time; the anchor-blind baselines get neither. Beating them therefore does
not show that *comparing against* the anchor helps. The floor consumes exactly
that information and nothing else:

```
pred(query_protein, drug) = pKi(anchor(drug), drug)
```

Untrained, deterministic given the split, and non-trivial under per-protein
metrics (it ranks drugs within a protein by their analogue's affinity).

**Davis-OOD, mean over three partitions:**

| Exp | Source | Model | CI | RMSE | AUROC |
|---|---|---|---|---|---|
| 1 | DTC | plain | 0.504 | 1.329 | 0.509 |
| 1 | DTC | **retrieval** | 0.550 | 3.446 | 0.676 |
| 1 | DTC | AnchorMLP | 0.557 | 1.018 | 0.740 |
| 1 | BDB | plain | 0.510 | 1.740 | 0.587 |
| 1 | BDB | **retrieval** | 0.528 | 3.450 | 0.644 |
| 1 | BDB | AnchorMLP | 0.541 | 1.275 | 0.689 |
| 3 | DTC | **retrieval** | 0.564 | 3.505 | 0.707 |
| 3 | DTC | plain | 0.640 | 0.797 | 0.864 |
| 3 | DTC | ConciseAnchor | 0.647 | 0.730 | 0.912 |
| 3 | BDB | **retrieval** | 0.556 | 3.521 | 0.721 |
| 3 | BDB | plain | 0.633 | 0.762 | 0.881 |
| 3 | BDB | ConciseAnchor | 0.640 | 0.697 | 0.924 |

**The two architectures separate.** Retrieval's share of the anchor model's gain:

| | AUROC | CI |
|---|---|---|
| exp1 DTC | **72%** | **87%** |
| exp1 BDB | **55%** | **60%** |
| exp3 DTC | 0% (floor is *below* plain) | 0% |
| exp3 BDB | 0% | 0% |

For AnchorMLP most of the ranking gain is the retrieved information, not the
model. For ConciseAnchor the plain model clears the floor by ~0.16 AUROC, so the
gain sits entirely above it. Rule of thumb: **adding an anchor to a baseline
weaker than the retrieval floor mostly adds retrieval.**

RMSE is the exception in both — the floor is ~3.45 everywhere, because anchors
are strong binders by construction while most Davis pairs sit at the pKi=5
detection limit. Calibration is the part no retrieval step performs.

## 7. Three protein partitions

Split seeds 42/1/2 at fixed training seed 42, so the partition is the only
moving part. Partitions overlap little (27 and 16 of ~283 test proteins shared
between consecutive draws).

- **Experiment 1 holds completely.** Anchor wins all three Davis-OOD metrics in
  all six source-partition runs, no sign reversal; BDB RMSE t = −51. Still inert
  in-distribution (|t| ≤ 2.2).
- **Experiment 3 narrows.** AUROC (+0.047, +0.043) and RMSE (−0.067, −0.066)
  hold, but **the CI gains do not** (t = 1.05 and 1.37), and individual
  partitions reverse sign on three in-distribution cells. "Wins in all 24 cells"
  is true of the seed-42 partition only, not of the method.

These t values measure **partition variance at fixed initialisation** and are
not comparable with the seed-based tests. Reported separately, never pooled.

## 8. A refactor bug the smoke test caught

Extracting `build_davis_dataset()` out of `eval_davis.main()` left `davis_df`
referenced in main's oracle block but defined only inside the new function.
`py_compile` passed — it is a runtime `NameError`, not a syntax error. A CPU
smoke run against archived checkpoints caught it ~7 minutes before the live
sweep would have hit it and lost experiment 3's split-2 Davis row. An AST scan
for undefined names in function bodies now covers the same class of mistake.

---

## Status of the earlier recommendations

| # | Recommendation | Status |
|---|---|---|
| 1 | Re-centre Table 1 on distribution shift, report mean ± std | done |
| 2 | Rebuild experiment 4: filter identical-sequence pairs, drop ≥0.9, report 81% | done |
| 3 | Qualify the oracle claim | done — went further, the claim was dropped (§3 note) |
| 4 | State the split fix | done |
| 5 | Note exp1/exp3 evaluate different Davis subsets | done |

## Open

1. **The DrugBAN port's featurizer** (§5 caveat). Either lift it to the 74-dim
   canonical featurizer and re-run experiment 2, or state the deviation and
   downgrade the claim to "we could not measure an effect with *our port*".
   Currently the paper does the latter; the former would make it evidence.
2. **Retrieval floor for experiment 2.** Not computed — exp2 is inconclusive
   either way, so it would not change the conclusion.
3. **Multi-split for experiment 2.** Same reasoning.
