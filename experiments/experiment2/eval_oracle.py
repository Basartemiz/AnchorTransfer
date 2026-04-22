"""Oracle-anchor eval on the DTC held-out test split (upper-bound baseline)."""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
from anchor_transfer.model.esm_dta import EsmDTAModel

from experiments.experiment2.dataset import (
    AnchorTransferDataset,
    collate_fn,
    compute_drug_anchors_oracle,
    get_esm2_embeddings,
)
from experiments.experiment2.eval import evaluate_model, quartile_metrics
from experiments.experiment2.main import DTC_INTERACTIONS_PATH, MERGED_SEQUENCES_PATH
from experiments.experiment2.split import ood_split
from experiments.experiment2.train import CHECKPOINT_DIR, _macro_auroc, _macro_metrics


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    interactions_df = pd.read_csv(DTC_INTERACTIONS_PATH)
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)

    dtc_uniprots = set(interactions_df["uniprot_id"].unique())
    dtc_sequences = {uid: all_sequences[uid] for uid in dtc_uniprots if uid in all_sequences}

    # Replay the training split (seed=42).
    train_uids, val_uids, test_uids = ood_split(
        dtc_sequences,
        test_frac=0.1, val_frac=0.1,
        test_identity=0.3, val_identity=0.5,
        seed=42,
    )
    print(f"Split proteins: train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}")

    esm2_embeddings = get_esm2_embeddings(dtc_sequences)

    test_interactions_df = interactions_df[interactions_df["uniprot_id"].isin(test_uids)]
    oracle_anchor, oracle_second, oracle_anchor_pki, oracle_second_pki = compute_drug_anchors_oracle(
        test_interactions_df
    )
    oracle_test_ds = AnchorTransferDataset(
        test_interactions_df, esm2_embeddings, test_uids,
        oracle_anchor, oracle_second,
        drug_to_anchor_pki=oracle_anchor_pki,
        drug_to_second_pki=oracle_second_pki,
    )
    oracle_test_loader = DataLoader(oracle_test_ds, batch_size=64, shuffle=False,
                                    collate_fn=collate_fn)
    print(f"Oracle test samples: {len(oracle_test_ds)} (drugs with oracle anchor: {len(oracle_anchor)})")

    esm_dta_model = EsmDTAModel(esm2_dim=1280, prot_proj_dim=128).to(device)
    anchor_dta_model = AnchorTransferDTAv2(esm2_dim=1280, proj_dim=128, dropout=0.3).to(device)
    esm_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "esm_dta_best.pt", map_location=device, weights_only=True)
    )
    anchor_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "anchor_dta_best.pt", map_location=device, weights_only=True)
    )

    esm_oracle, esm_records = evaluate_model(esm_dta_model, oracle_test_loader, device, return_records=True)
    anchor_oracle, anchor_records = evaluate_model(anchor_dta_model, oracle_test_loader, device, return_records=True)
    esm_ci, esm_rmse, esm_score = _macro_metrics(esm_oracle)
    anchor_ci, anchor_rmse, anchor_score = _macro_metrics(anchor_oracle)
    esm_auroc, esm_n_auroc = _macro_auroc(esm_oracle)
    anchor_auroc, anchor_n_auroc = _macro_auroc(anchor_oracle)

    print("=" * 72)
    print("TEST SET — ORACLE ANCHORS (strongest test binder per drug)")
    print(f"  ESM-DTA:           macro_CI={esm_ci:.4f}  macro_RMSE={esm_rmse:.4f}  macro_AUROC={esm_auroc:.4f} (n={esm_n_auroc})  score={esm_score:.4f}")
    print(f"  AnchorTransfer v2: macro_CI={anchor_ci:.4f}  macro_RMSE={anchor_rmse:.4f}  macro_AUROC={anchor_auroc:.4f} (n={anchor_n_auroc})  score={anchor_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown:")
    esm_q = quartile_metrics(esm_records, by="anchor_pki")
    anchor_q = quartile_metrics(anchor_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'ESM RMSE':>9s} {'ESM CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qe, qa in zip(esm_q, anchor_q):
        print(f"{qe['bin']:>14s} | {qe['n']:>6d} | "
              f"{qe['rmse']:>9.4f} {qe['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
