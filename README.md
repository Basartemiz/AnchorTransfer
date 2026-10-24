# Anchor Transfer Learning for Cross-Dataset Drug-Target Affinity Prediction

Code for the paper. **This README and the `experiments/` directory correspond to
version 5 of the preprint** ([Zenodo record 21902290](https://zenodo.org/records/21902290),
DOI [10.5281/zenodo.21902290](https://doi.org/10.5281/zenodo.21902290)). The
concept DOI [10.5281/zenodo.19405967](https://doi.org/10.5281/zenodo.19405967)
always resolves to the latest version.

Earlier versions of the paper and of this README described different models
(V1-35M, V2-35M, V2-650M) and a different pipeline (`reproduce/`). Those are not
part of v5; see [Earlier versions](#earlier-versions).

## Core idea

Instead of scoring a query protein *q* and a drug *d* on their own, the model is
also given an **anchor** *a*: a training protein already known to bind a drug
similar to *d*. It predicts *f(a, q, d) → pKi* by judging the query against that
known binder. At test time the anchor is retrieved by Tanimoto similarity to
the training drugs, so no test labels are used.

## What is in v5

Each anchor model is compared against its own anchor-blind counterpart. All are
trained on a protein-level out-of-distribution split of DTC (and BindingDB where
noted) and transferred to the Davis kinase set.

| Experiment | Anchor model | Plain baseline | Training data | Paper table | Code |
|---|---|---|---|---|---|
| 1 | AnchorMLP | ESM-DTA | DTC, BDB | 3 | `experiments/experiment1/` |
| 2 | AnchorDrugBAN | DrugBAN | DTC | 4 | `experiments/experiment2/` |
| 3 | ConciseAnchor | CoNCISE | DTC, BDB | 5 | `experiments/experiment3/` |
| 1, 3 on three splits, with the retrieval-only floor | — | — | DTC, BDB | 6 | `eval_retrieval.py` in 1 and 3 |
| 4 (activity cliffs) | ConciseAnchor | CoNCISE | DTC | 7 | `experiments/experiment4/` |

| Paper name | Model class | Protein / drug input |
|---|---|---|
| AnchorMLP | `AnchorTransferDTAv2` (`src/anchor_transfer/model/anchor_transfer_v2.py`) | mean-pooled ESM-2 650M / SMILES CNN |
| ESM-DTA | `EsmDTAModel` (`esm_dta.py`) | same, without the anchor |
| AnchorDrugBAN | `AnchorDrugBAN` (`anchor_drugban.py`) | residue CNN / molecular graph |
| DrugBAN | `DrugBAN` (`drugban.py`) | same, without the anchor |
| ConciseAnchor | `ConciseAnchorBilinear` (`concise_anchor_bilinear.py`) | Raygun-compressed ESM-2 / Morgan-fingerprint codes |
| CoNCISE | `ConciseDTA` (`concise_dta.py`) | same, without the anchor |

**Mapping from earlier versions:** AnchorMLP uses the architecture previously
called V2-650M, called non-residually and retrained from scratch in Experiment 1.
V1-35M and V2-35M do not appear in v5.

## Repository layout

```text
experiments/                  # v5: everything behind the paper's tables
  experiment{1,2,3,4}/        # split.py, dataset.py, train.py, main.py, eval_davis.py, ...
  seed_run.py                 # seeded-run wrapper (SEED, SPLIT_SEED, EXP, DS, MOD)
  summarize_seeds.py          # mean ± std and paired t over seeds or splits
  pod_runs/                   # logs and parsed metrics of the runs reported in v5
src/anchor_transfer/          # model architectures and data utilities (pip install -e .)
REPRODUCE.md                  # step-by-step commands, table by table
revised_paper/                # LaTeX source of v5, with the retrieval correction below
prod/v5.pdf                   # the v5 PDF (identical to the Zenodo file)
reproduce/, scripts/          # pipeline of earlier versions (see below)
```

## Reproducing the v5 results

Full instructions, including the exact command for each table, are in
[`REPRODUCE.md`](REPRODUCE.md). In short:

**1. Environments.** Python 3.11 for Experiments 1–2 (`pip install -r requirements.txt`),
Python 3.12 with `concise-dti` for Experiments 3–4, and the MMseqs2 binary on
`PATH`.

**2. Input data.** Four files, taken from the artifact record
[Zenodo 19481471](https://zenodo.org/records/19481471):

```bash
Z=https://zenodo.org/api/records/19481471/files
mkdir -p data/processed data/raw
for f in dtc_training_interactions.csv bindingdb_interactions.csv merged_sequences.json; do
  curl -fL "$Z/$f/content" -o "data/processed/$f"
done
curl -fL "$Z/davis_benchmark.csv/content" -o data/raw/davis_benchmark.csv
```

`reproduce/01_prepare_data.sh` documents how these were built from the raw DTC
export. Protein embeddings (ESM-2 650M, Raygun) are computed on first use and
cached under each experiment's `caches/` directory.

**3. Run.** For example, one seed of Experiment 1:

```bash
SEED=42 EXP=experiment1 DS=dtc MOD=main       PYTHONPATH=src:. python experiments/seed_run.py
SEED=42 EXP=experiment1 DS=dtc MOD=eval_davis PYTHONPATH=src:. python experiments/seed_run.py
```

`MOD=main` trains both models and reports the Test and Oracle rows;
`MOD=eval_davis` loads the resulting checkpoints and reports Davis-OOD.

## Splits

Proteins are split with MMseqs2 clustering: test proteins share at most 30%
sequence identity with training proteins, validation proteins at most 50%.
Splits are not stored as files; `split.py` regenerates them deterministically
from the interaction file and the split seed. Sequences are sorted before
clustering, so the same seed gives a byte-identical split on any machine.

| `SPLIT_SEED` | Train / val / test proteins | Used for |
|---|---|---|
| 42 (default) | 2,262 / 282 / 283 | all per-seed tables (3, 4, 5, 7) |
| 1 | 2,263 / 282 / 282 | Table 6 |
| 2 | 2,256 / 282 / 289 | Table 6 |

`REPRODUCE.md` §3 has a one-line check that prints the seed-42 test-set digest
(`2262 282 283 6412f7a2b391`).

## Davis evaluation protocol

As implemented in `experiments/experiment*/eval_davis.py`:

- **Proteins.** Only Davis proteins with at most 30% identity to every training
  protein are scored (MMseqs2, 80% coverage).
- **Drugs.** Davis drugs are **not** removed from the training data. The split
  is by protein, and 57 of the 68 Davis drugs occur in DTC by canonical SMILES
  (paper Table 1), so Davis-OOD tests dataset shift on unseen proteins rather
  than compound novelty.
- **Anchors.** For each Davis drug, the most similar drug among the
  training-split interactions is found by Tanimoto similarity (Morgan radius 2,
  1024 bits). The anchor is that drug's strongest training binder. Drugs are
  dropped when the best similarity is below 0.7 or the anchor's pKi is below 7.
- **Retrieval floor.** `eval_retrieval.py` predicts the retrieved anchor's own
  pKi for every query, on the same samples and with the same metric code.

> **Correction to v5.** Section 3.4 and the captions of Tables 1 and 2 in the
> v5 PDF state that canonical duplicates of the evaluation drugs are removed
> from the retrieval pool and that the fingerprints are chirality-aware and
> 2048-bit. That describes the earlier pipeline
> (`scripts/eval/evaluate_anchor_transfer_davis_paper.py`), not the code behind
> the v5 tables. Duplicates are not removed, so a Davis drug that is present in
> the training data retrieves itself: on the DTC training split (seed 42), 56 of
> the 60 anchored Davis drugs match a training drug at Tanimoto 1.0 (51 the same
> molecule, 5 a stereoisomer). The text is corrected in `revised_paper/`.

## Checkpoints and results

Checkpoints for the v5 runs are not published; the commands above regenerate
them. The run logs behind the v5 tables and the parsed metrics are in
`experiments/pod_runs/` (`FINDINGS.md`, `results/seed_summary.json`,
`results/split_summary.json`, `results/all_results.json`). Per-sample
prediction files are not saved.

The checkpoints on Zenodo 19481471 (`v1_35m_*`, `v2_35m_*`, `v2_650m_*`, ...)
come from an earlier version of the paper and do not reproduce the v5 numbers.

## Earlier versions

`reproduce/` (steps 00–09) and most of `scripts/` are the pipeline of earlier
paper versions: the V1/V2 models, BindingDB→GLASS2, MooDeng, LCIdb and kNN
comparisons. They are kept for reference only. Several of these scripts import
model variants that have since been removed from `src/anchor_transfer/model/`
(e.g. V1 `anchor_transfer.py`, `conplex.py`), so they no longer run on the
current code.

## Citation

```bibtex
@misc{temiz2026anchor,
  title     = {Anchor Transfer Learning for Cross-Dataset Drug-Target Affinity Prediction},
  author    = {Temiz, Ba{\c{s}}ar},
  year      = {2026},
  publisher = {Zenodo},
  note      = {Preprint, version 5},
  doi       = {10.5281/zenodo.21902290}
}
```
