# Reproducing the paper

Every table in the paper is produced by the commands below. Nothing here depends
on a machine-specific path or on a cached artifact that is not rebuilt on demand.

## 1. Environments

Two are needed, because `concise-dti` requires Python $\geq$ 3.12 while the rest
of the stack is validated on 3.11.

```bash
# Experiments 1 and 2
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # torch, rdkit, fair-esm, torch-geometric, ...

# Experiments 3 and 4 (CoNCISE backbone + Raygun)
python3.12 -m venv .venv312
.venv312/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
.venv312/bin/pip install "concise-dti>=1.0" rdkit fair-esm pandas scikit-learn tqdm scipy "numpy<2"
```

MMseqs2 is a system binary, not a pip package:

```bash
wget https://github.com/soedinglab/MMseqs2/releases/download/15-6f452/mmseqs-linux-avx2.tar.gz
tar xzf mmseqs-linux-avx2.tar.gz && sudo cp mmseqs/bin/mmseqs /usr/local/bin/
```

Set `PYTHONPATH=src:.` for all commands below, and run them from the repo root.

## 2. Data

`data/processed/` must contain `dtc_training_interactions.csv`,
`bindingdb_interactions.csv`, and `merged_sequences.json`; `data/raw/` must
contain `davis_benchmark.csv`. See `reproduce/01_prepare_data.sh` for how these
are built from source.

Protein embeddings are computed on first use and cached under each experiment's
`caches/` directory (ESM-2 650M for experiment 1; Raygun-compressed ESM-2, about
691 MB, for experiments 3 and 4). The first run is the slow one.

## 3. Determinism

The OOD split is pinned by sorting sequences before clustering
(`experiments/*/split.py`). Without that sort, protein identifiers reach MMseqs2
in the iteration order of a Python `set`, which is randomised per process: the
seed pins the shuffle but not the clustering, and separate runs get different
held-out proteins and different test-set sizes. With the sort, repeated runs
produce byte-identical splits — verify with:

```bash
PYTHONPATH=src:. python - <<'PY'
import hashlib, json, pandas as pd
from experiments.experiment1.split import ood_split
df = pd.read_csv("data/processed/dtc_training_interactions.csv")
seqs_all = json.load(open("data/processed/merged_sequences.json"))
seqs = {u: seqs_all[u] for u in set(df.uniprot_id.unique()) if u in seqs_all}
tr, va, te = ood_split(seqs, 0.1, 0.1, 0.3, 0.5, seed=42)
print(len(tr), len(va), len(te), hashlib.md5("".join(sorted(te)).encode()).hexdigest()[:12])
PY
```

Expected, on any machine and in any process: `2262 282 283 6412f7a2b391`.

Training itself is seeded only through `experiments/seed_run.py`; the
entrypoints do not take a `--seed` flag.

Two independent knobs control repetition, and they answer different questions:

| Variable | Default | Varying it measures |
|---|---|---|
| `SEED` (via `seed_run.py`) | — | initialisation variance, split held fixed |
| `SPLIT_SEED` | `42` | dependence on which proteins were held out, seed held fixed |

`SPLIT_SEED=42` reproduces the partition every per-seed table uses. Splits 1 and
2 give `2263 282 282` and `2256 282 289`; consecutive draws share only 27 and 16
of roughly 283 test proteins. Never pool the two kinds of spread — the paper
reports them separately.

When running a non-default split, redirect the checkpoints so the seed-42 run is
not overwritten: `EXP1_CHECKPOINTS` and `EXP3_CHECKPOINTS` override the
destination directory (`EXP2_CHECKPOINTS` does the same for experiment 2).

## 4. Table by table

`main` emits the **Test** and **Oracle** rows; `eval_davis` emits **Davis-OOD**
(and **Davis-oracle** for experiment 3). `eval_davis` loads the `*_best.pt`
written by `main`, so it must run before the next seed overwrites
`checkpoints/`.

### Paper Table 3 — Experiment 1 (5 seeds, DTC and BDB)

```bash
mkdir -p logs/seeds
for ds in dtc bdb; do
  for seed in 42 43 44 45 46; do
    SEED=$seed EXP=experiment1 DS=$ds MOD=main \
      PYTHONPATH=src:. python experiments/seed_run.py > logs/seeds/experiment1_${ds}_s${seed}.log 2>&1
    SEED=$seed EXP=experiment1 DS=$ds MOD=eval_davis \
      PYTHONPATH=src:. python experiments/seed_run.py > logs/seeds/experiment1_${ds}_s${seed}_davis.log 2>&1
  done
done
PYTHONPATH=src:. python experiments/summarize_seeds.py logs/seeds results/seed_summary.json
```

`summarize_seeds.py` prints mean ± std and the paired $t$ over seeds, which is
what the table reports.

### Paper Table 4 — Experiment 2 (DTC)

Same loop with `EXP=experiment2`, `ds=dtc`. Budget roughly 4 h per seed: two
models train in one loop and the anchor model runs its shared attention twice,
so the step is dominated by many small GPU kernels launched from one Python
thread. `num_workers` does not help — data loading is not the bottleneck.

### Paper Table 5 — Experiment 3 (DTC and BDB)

```bash
for ds in dtc bdb; do
  PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment3.main --dataset $ds
  PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment3.eval_davis --dataset $ds
done
```

Run DTC first and archive `experiments/experiment3/checkpoints/` before the BDB
run overwrites it — Table 7 needs the DTC checkpoints.

### Paper Table 6 — three protein partitions, with the retrieval floor

Two parts. First, repeat experiments 1 and 3 on the extra partitions, holding
the training seed at 42 so the partition is the only thing that moves:

```bash
mkdir -p logs/splits
for split in 1 2; do
  for ds in dtc bdb; do
    EXP1_CHECKPOINTS=experiments/experiment1/ckpt_split${split}_${ds} \
    SPLIT_SEED=$split SEED=42 EXP=experiment1 DS=$ds MOD=main \
      PYTHONPATH=src:. python experiments/seed_run.py > logs/splits/experiment1_${ds}_split${split}.log 2>&1
    EXP1_CHECKPOINTS=experiments/experiment1/ckpt_split${split}_${ds} \
    SPLIT_SEED=$split SEED=42 EXP=experiment1 DS=$ds MOD=eval_davis \
      PYTHONPATH=src:. python experiments/seed_run.py > logs/splits/experiment1_${ds}_split${split}_davis.log 2>&1
  done
done
```

Same loop with `EXP=experiment3`, `EXP3_CHECKPOINTS`, and the 3.12 interpreter.
The seed-42 partition is already covered by the per-seed runs above: copy
`logs/seeds/experiment{1,3}_${ds}_s42*.log` into `logs/splits/` renamed to
`_split42`, then

```bash
PYTHONPATH=src:. python experiments/summarize_seeds.py logs/splits results/split_summary.json
```

`summarize_seeds.py` reads both the `_s{n}` and `_split{n}` naming, so the same
paired $t$-test aggregates either axis. Keep the two directories separate — a
directory holding both namings would count the shared seed-42 run twice.

Second, the retrieval-only floor. It has no training and no seed, so one run per
dataset and partition determines it:

```bash
for ds in dtc bdb; do
  for split in 42 1 2; do
    SPLIT_SEED=$split PYTHONPATH=src:. python -m experiments.experiment1.eval_retrieval --dataset $ds
    SPLIT_SEED=$split PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment3.eval_retrieval --dataset $ds
  done
done
```

Both scripts call the same `build_davis_dataset()` the trained models are scored
on and push the predictions through the same `aggregate_records()`, so the floor
is measured on identical samples with identical metric code rather than on a set
rebuilt from the same steps. No GPU is needed once the embedding caches exist;
`CUDA_VISIBLE_DEVICES=""` lets these run beside a training job.

The two experiments give different floors (Table 6 reports both) because they
evaluate different Davis subsets — ESM-2 drops sequences over 1,022 residues
where Raygun truncates them, so the same control scores 2,992 samples in
experiment 1 and more in experiment 3.

### Paper Table 7 — Experiment 4 (activity cliffs)

```bash
EXP3_CHECKPOINTS=results/ckpt_experiment3_dtc \
PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment4.eval_homolog_cliff --identity 0.5
```

Repeat for `--identity 0.7`. `EXP3_CHECKPOINTS` is optional; it defaults to
`experiments/experiment3/checkpoints` and exists so the evaluation can point at
an archived DTC run while that directory is being rewritten by another job.

The script prints two blocks. The first covers all pairs and is reported only
for transparency: Davis stores point mutants under the wild-type sequence
(`KIT` and `KIT(D816V)` have identical sequence fields), so most "cliff" pairs
are the same input twice, the models return identical predictions, and the
ranking metric scores those ties as failures. The second block, **distinct-
sequence pairs only**, is what the paper reports; filtering removes every tie.
The `identity 0.9` bin is omitted because 100% of its pairs are same-sequence.

## 5. GPU notes

Run one experiment at a time on a 16 GB card. Experiment 1's BDB embedding pass
alone holds about 13 GB, and running experiment 4 alongside it triggers a CUDA
OOM.
