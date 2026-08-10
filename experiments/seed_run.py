"""Run an experiment entrypoint under an explicit training seed.

The experiment entrypoints take no --seed flag and never call torch.manual_seed,
so repeated runs differ only by uncontrolled RNG. This wrapper seeds every RNG
before importing and calling main(), which is where the models are constructed.

The OOD split is unaffected: ood_split() builds its own random.Random(42)
internally and split.py sorts sequences before clustering, so the split stays
fixed while the training initialisation varies. That is the decomposition
needed to report seed variance meaningfully.

Usage:
    SEED=42 EXP=experiment1 DS=dtc MOD=main       python experiments/seed_run.py
    SEED=42 EXP=experiment1 DS=dtc MOD=eval_davis python experiments/seed_run.py

MOD=main produces the Test and Oracle rows; MOD=eval_davis produces Davis-OOD
(and, for experiment 3, Davis-oracle). eval_davis loads the *_best.pt written by
main, so it must run before the next seed overwrites the checkpoint directory.
"""
from __future__ import annotations

import importlib
import os
import random
import sys

import numpy as np
import torch

seed = int(os.environ["SEED"])
exp = os.environ["EXP"]
ds = os.environ["DS"]
mod = os.environ["MOD"]

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

print(f"[seed_run] seed={seed} exp={exp} dataset={ds} entry={mod}", flush=True)

m = importlib.import_module(f"experiments.{exp}.{mod}")
sys.argv = [mod, "--dataset", ds]
m.main()
