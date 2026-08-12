"""Retrieval-only baseline on Davis-OOD for experiment 3.

Same control as experiments/experiment1/eval_retrieval.py, recomputed inside
this experiment's pipeline: ConciseAnchor evaluates a different sample set
(Raygun needs sequences of at least 50 residues, ESM-2 drops sequences over
1022), so experiment 1's numbers are not the right comparison for this table.

    pred(query_protein, drug) = pKi(anchor(drug), drug)

No training and no seed: given a split the prediction is fixed.

Run from repo root:
    python -m experiments.experiment3.eval_retrieval                 # DTC anchors
    SPLIT_SEED=1 python -m experiments.experiment3.eval_retrieval --dataset bdb
"""
from __future__ import annotations

import argparse

from torch.utils.data import DataLoader

from experiments.experiment3.dataset import collate_fn
from experiments.experiment3.eval import aggregate_records
from experiments.experiment3.eval_davis import build_davis_dataset
from experiments.experiment3.main import INTERACTIONS_PATHS, SPLIT_SEED
from experiments.experiment3.train import _macro_auroc, _macro_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(INTERACTIONS_PATHS), default="dtc",
                        help="Training-source anchors + OOD split to replay (dtc or bdb).")
    args = parser.parse_args()

    davis_ds, _, _, _ = build_davis_dataset(args.dataset)
    loader = DataLoader(davis_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

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
