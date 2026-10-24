"""Retrieval-only baseline on Davis-OOD: predict the retrieved anchor's pKi.

The anchor model sees information the plain model never gets — a training
protein retrieved for the query drug, together with that protein's measured
pKi. Beating the plain model therefore does not by itself show that comparing
against the anchor helps; the retrieval alone might account for the gap.

This baseline uses exactly that retrieved information and nothing else:

    pred(query_protein, drug) = pKi(anchor(drug), drug)

The query protein is ignored entirely, so the predictor cannot compare anything.
It is still non-trivial under the per-protein metrics, which rank drugs within a
protein: drugs whose nearest training analogue binds strongly are ranked above
drugs whose analogue binds weakly. Whatever the anchor model scores above this
line is what conditioning on the anchor buys beyond retrieving it.

There is no training and no seed: given a split, the prediction is fixed. One
run per (dataset, split) is the whole result.

Run from repo root:
    python -m experiments.experiment1.eval_retrieval                 # DTC anchors
    SPLIT_SEED=1 python -m experiments.experiment1.eval_retrieval --dataset bdb
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from experiments.experiment1.dataset import collate_fn
from experiments.experiment1.eval import aggregate_records
from experiments.experiment1.eval_davis import build_davis_dataset
from experiments.experiment1.main import INTERACTIONS_PATHS, SPLIT_SEED
from experiments.experiment1.train import _macro_auroc, _macro_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(INTERACTIONS_PATHS), default="dtc",
                        help="Training-source anchors + OOD split to replay (dtc or bdb).")
    args = parser.parse_args()

    davis_ds = build_davis_dataset(args.dataset)
    loader = DataLoader(davis_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    records = []
    for batch in loader:
        for i in range(len(batch["protein_id"])):
            records.append({
                "protein_id": batch["protein_id"][i],
                "drug_id": batch["drug_id"][i],
                "pred": float(batch["anchor_pki"][i]),
                "true": float(batch["pki"][i]),
            })

    metrics = aggregate_records(records)
    ci, rmse, score = _macro_metrics(metrics)
    auroc, n_auroc = _macro_auroc(metrics)

    print("=" * 72)
    print(f"DAVIS-OOD — RETRIEVAL ONLY (split_seed={SPLIT_SEED}, {len(records)} samples)")
    print(f"  Retrieval-kNN:     macro_CI={ci:.4f}  macro_RMSE={rmse:.4f}  "
          f"macro_AUROC={auroc:.4f} (n={n_auroc})  score={score:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
