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

## 4. Table by table

`main` emits the **Test** and **Oracle** rows; `eval_davis` emits **Davis-OOD**
(and **Davis-oracle** for experiment 3). `eval_davis` loads the `*_best.pt`
written by `main`, so it must run before the next seed overwrites
`checkpoints/`.

### Table 1 — Experiment 1 (5 seeds, DTC and BDB)

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

### Table 2 — Experiment 2 (DTC)

Same loop with `EXP=experiment2`, `ds=dtc`. Budget roughly 4 h per seed: two
models train in one loop and the anchor model runs its shared attention twice,
so the step is dominated by many small GPU kernels launched from one Python
thread. `num_workers` does not help — data loading is not the bottleneck.

### Table 3 — Experiment 3 (DTC and BDB)

```bash
for ds in dtc bdb; do
  PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment3.main --dataset $ds
  PYTHONPATH=src:. .venv312/bin/python -m experiments.experiment3.eval_davis --dataset $ds
done
```

Run DTC first and archive `experiments/experiment3/checkpoints/` before the BDB
run overwrites it — Table 4 needs the DTC checkpoints.

### Table 4 — Experiment 4 (activity cliffs)

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
