"""Entry point: ESM-DTA vs AnchorTransfer v2 with MMseqs OOD split.

Run from repo root:
    python -m experiments.experiment1.main                 # DTC (default)
    python -m experiments.experiment1.main --dataset bdb   # BindingDB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
from anchor_transfer.model.esm_dta import EsmDTAModel

from experiments.experiment1.dataset import (
    AnchorTransferDataset,
    collate_fn,
    compute_drug_anchors,
    compute_drug_anchors_oracle,
    get_esm2_embeddings,
)
from experiments.experiment1.eval import evaluate_model, quartile_metrics
from experiments.experiment1.split import ood_split
from experiments.experiment1.train import CHECKPOINT_DIR, _macro_auroc, _macro_metrics, train_models

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DTC_INTERACTIONS_PATH = DATA_DIR / "processed" / "dtc_training_interactions.csv"
BDB_INTERACTIONS_PATH = DATA_DIR / "processed" / "bindingdb_interactions.csv"
MERGED_SEQUENCES_PATH = DATA_DIR / "processed" / "merged_sequences.json"

INTERACTIONS_PATHS = {"dtc": DTC_INTERACTIONS_PATH, "bdb": BDB_INTERACTIONS_PATH}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(INTERACTIONS_PATHS), default="dtc",
                        help="Training interactions source: dtc (default) or bdb (BindingDB).")
    args = parser.parse_args()

    esm2_dim = 1280
    prot_proj_dim = 128
    head_dropout = 0.3
    batch_size = 64
    epochs = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interactions_path = INTERACTIONS_PATHS[args.dataset]
    print(f"Using device: {device}  |  dataset: {args.dataset} ({interactions_path.name})")

    esm_dta_model = EsmDTAModel(esm2_dim=esm2_dim, prot_proj_dim=prot_proj_dim).to(device)
    anchor_dta_model = AnchorTransferDTAv2(
        esm2_dim=esm2_dim,
        proj_dim=prot_proj_dim,
        dropout=head_dropout,
    ).to(device)

    interactions_df = pd.read_csv(interactions_path)
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)

    dtc_uniprots = set(interactions_df["uniprot_id"].unique())
    dtc_sequences = {uid: all_sequences[uid] for uid in dtc_uniprots if uid in all_sequences}

    # Two-tier OOD split: test ≤30% identity, val ≤50% identity to train.
    train_uids, val_uids, test_uids = ood_split(
        dtc_sequences,
        test_frac=0.1, val_frac=0.1,
        test_identity=0.3, val_identity=0.5,
        seed=42,
    )
    print(f"Split proteins: train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}")

    esm2_embeddings = get_esm2_embeddings(dtc_sequences)

    # Anchors come from TRAIN proteins only to avoid leaking held-out identities.
    drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki = compute_drug_anchors(
        interactions_df, train_uids
    )

    train_ds = AnchorTransferDataset(interactions_df, esm2_embeddings, train_uids,
                                     drug_to_anchor, drug_to_second,
                                     drug_to_anchor_pki=drug_to_anchor_pki,
                                     drug_to_second_pki=drug_to_second_pki)
    val_ds = AnchorTransferDataset(interactions_df, esm2_embeddings, val_uids,
                                   drug_to_anchor, drug_to_second,
                                   drug_to_anchor_pki=drug_to_anchor_pki,
                                   drug_to_second_pki=drug_to_second_pki)
    test_ds = AnchorTransferDataset(interactions_df, esm2_embeddings, test_uids,
                                    drug_to_anchor, drug_to_second,
                                    drug_to_anchor_pki=drug_to_anchor_pki,
                                    drug_to_second_pki=drug_to_second_pki)
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn)

    train_models(train_loader, val_loader, epochs=epochs,
                 esm_dta_model=esm_dta_model,
                 anchor_dta_model=anchor_dta_model)

    device = next(esm_dta_model.parameters()).device
    esm_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "esm_dta_best.pt", map_location=device)
    )
    anchor_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "anchor_dta_best.pt", map_location=device)
    )

    esm_test, esm_records = evaluate_model(esm_dta_model, test_loader, device, return_records=True)
    anchor_test, anchor_records = evaluate_model(anchor_dta_model, test_loader, device, return_records=True)
    esm_ci, esm_rmse, esm_score = _macro_metrics(esm_test)
    anchor_ci, anchor_rmse, anchor_score = _macro_metrics(anchor_test)
    esm_auroc, esm_n_auroc = _macro_auroc(esm_test)
    anchor_auroc, anchor_n_auroc = _macro_auroc(anchor_test)

    print("=" * 72)
    print("TEST SET (best checkpoints)")
    print(f"  ESM-DTA:           macro_CI={esm_ci:.4f}  macro_RMSE={esm_rmse:.4f}  macro_AUROC={esm_auroc:.4f} (n={esm_n_auroc})  score={esm_score:.4f}")
    print(f"  AnchorTransfer v2: macro_CI={anchor_ci:.4f}  macro_RMSE={anchor_rmse:.4f}  macro_AUROC={anchor_auroc:.4f} (n={anchor_n_auroc})  score={anchor_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown (data-driven quartiles of anchor pKi):")
    esm_q = quartile_metrics(esm_records, by="anchor_pki")
    anchor_q = quartile_metrics(anchor_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'ESM RMSE':>9s} {'ESM CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qe, qa in zip(esm_q, anchor_q):
        print(f"{qe['bin']:>14s} | {qe['n']:>6d} | "
              f"{qe['rmse']:>9.4f} {qe['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)

    # Oracle upper bound: strongest test binder per drug as the anchor.
    # Anchor pKi tracks target pKi closely — shows the ceiling when anchors are "correct".
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
    oracle_test_loader = DataLoader(oracle_test_ds, batch_size=batch_size, shuffle=False,
                                    collate_fn=collate_fn)
    print(f"Oracle test samples: {len(oracle_test_ds)} (drugs with oracle anchor: {len(oracle_anchor)})")

    esm_oracle, esm_oracle_records = evaluate_model(esm_dta_model, oracle_test_loader, device, return_records=True)
    anchor_oracle, anchor_oracle_records = evaluate_model(anchor_dta_model, oracle_test_loader, device, return_records=True)
    esm_o_ci, esm_o_rmse, esm_o_score = _macro_metrics(esm_oracle)
    anchor_o_ci, anchor_o_rmse, anchor_o_score = _macro_metrics(anchor_oracle)
    esm_o_auroc, esm_o_n = _macro_auroc(esm_oracle)
    anchor_o_auroc, anchor_o_n = _macro_auroc(anchor_oracle)

    print("=" * 72)
    print("TEST SET — ORACLE ANCHORS (strongest test binder per drug, upper bound)")
    print(f"  ESM-DTA:           macro_CI={esm_o_ci:.4f}  macro_RMSE={esm_o_rmse:.4f}  macro_AUROC={esm_o_auroc:.4f} (n={esm_o_n})  score={esm_o_score:.4f}")
    print(f"  AnchorTransfer v2: macro_CI={anchor_o_ci:.4f}  macro_RMSE={anchor_o_rmse:.4f}  macro_AUROC={anchor_o_auroc:.4f} (n={anchor_o_n})  score={anchor_o_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown (oracle):")
    esm_oq = quartile_metrics(esm_oracle_records, by="anchor_pki")
    anchor_oq = quartile_metrics(anchor_oracle_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'ESM RMSE':>9s} {'ESM CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qe, qa in zip(esm_oq, anchor_oq):
        print(f"{qe['bin']:>14s} | {qe['n']:>6d} | "
              f"{qe['rmse']:>9.4f} {qe['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)


if __name__ == "__main__":

    
    main()
