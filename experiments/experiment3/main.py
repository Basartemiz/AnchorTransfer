"""Entry point: ConciseDTA vs ConciseAnchorBilinear with MMseqs OOD split.

Run from repo root:
    python -m experiments.experiment3.main                 # DTC (default)
    python -m experiments.experiment3.main --dataset bdb   # BindingDB
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear
from anchor_transfer.model.concise_dta import ConciseDTA

from experiments.experiment3.dataset import (
    RAYGUN_DIM,
    AnchorTransferDataset,
    build_morgan_cache,
    build_raygun_cache,
    collate_fn,
    compute_drug_anchors,
    compute_drug_anchors_oracle,
)
from experiments.experiment3.eval import evaluate_model, quartile_metrics
from experiments.experiment3.split import ood_split
from experiments.experiment3.train import CHECKPOINT_DIR, _macro_auroc, _macro_metrics, train_models

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DTC_INTERACTIONS_PATH = DATA_DIR / "processed" / "dtc_training_interactions.csv"
BDB_INTERACTIONS_PATH = DATA_DIR / "processed" / "bindingdb_interactions.csv"
MERGED_SEQUENCES_PATH = DATA_DIR / "processed" / "merged_sequences.json"

INTERACTIONS_PATHS = {"dtc": DTC_INTERACTIONS_PATH, "bdb": BDB_INTERACTIONS_PATH}

# CoNCISE backbone dims: Raygun outputs (50, 1280). The paper uses FSQ drug
# quantization with 3 codes → drug_dim/proj_dim; defaults mirror the
# ConciseAnchor class constructor.
PROJ_DIM = 256
N_HEADS = 32
DROPOUT = 0.2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(INTERACTIONS_PATHS), default="dtc")
    args = parser.parse_args()

    batch_size = 256
    epochs = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    interactions_path = INTERACTIONS_PATHS[args.dataset]
    print(f"Using device: {device}  |  dataset: {args.dataset} ({interactions_path.name})")

    dta_model = ConciseDTA(esm_dim=RAYGUN_DIM, proj_dim=PROJ_DIM, nheads=N_HEADS, dropout=DROPOUT).to(device)
    anchor_model = ConciseAnchorBilinear(residue_dim=RAYGUN_DIM, proj_dim=PROJ_DIM, dropout=DROPOUT).to(device)

    interactions_df = pd.read_csv(interactions_path)
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)

    uniprots_in_df = set(interactions_df["uniprot_id"].unique())
    sequences = {uid: all_sequences[uid] for uid in uniprots_in_df if uid in all_sequences}

    train_uids, val_uids, test_uids = ood_split(
        sequences,
        test_frac=0.1, val_frac=0.1,
        test_identity=0.3, val_identity=0.5,
        seed=42,
    )
    print(f"Split proteins: train={len(train_uids)} val={len(val_uids)} test={len(test_uids)}")

    protein_embeddings = build_raygun_cache(sequences, device=device)
    print(f"Raygun cache: {len(protein_embeddings)} / {len(sequences)} proteins embedded")

    drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki = compute_drug_anchors(
        interactions_df, train_uids
    )

    all_smi = set(interactions_df["ligand_smiles"].astype(str))
    morgan_cache = build_morgan_cache(list(all_smi))
    print(f"Morgan cache: {len(morgan_cache)} / {len(all_smi)} SMILES successfully fingerprinted")

    train_ds = AnchorTransferDataset(
        interactions_df, protein_embeddings, train_uids,
        drug_to_anchor, drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
        morgan_cache=morgan_cache,
    )
    val_ds = AnchorTransferDataset(
        interactions_df, protein_embeddings, val_uids,
        drug_to_anchor, drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
        morgan_cache=morgan_cache,
    )
    test_ds = AnchorTransferDataset(
        interactions_df, protein_embeddings, test_uids,
        drug_to_anchor, drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
        morgan_cache=morgan_cache,
    )
    print(f"Samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    train_models(train_loader, val_loader, epochs=epochs,
                 dta_model=dta_model,
                 anchor_model=anchor_model)

    dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "concise_dta_best.pt", map_location=device)
    )
    anchor_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "concise_anchor_best.pt", map_location=device)
    )

    dta_test, dta_records = evaluate_model(dta_model, test_loader, device, return_records=True)
    anchor_test, anchor_records = evaluate_model(anchor_model, test_loader, device, return_records=True)
    d_ci, d_rmse, d_score = _macro_metrics(dta_test)
    a_ci, a_rmse, a_score = _macro_metrics(anchor_test)
    d_auroc, d_n = _macro_auroc(dta_test)
    a_auroc, a_n = _macro_auroc(anchor_test)

    print("=" * 72)
    print("TEST SET (best checkpoints)")
    print(f"  ConciseDTA:    macro_CI={d_ci:.4f}  macro_RMSE={d_rmse:.4f}  macro_AUROC={d_auroc:.4f} (n={d_n})  score={d_score:.4f}")
    print(f"  ConciseAnchorBilinear: macro_CI={a_ci:.4f}  macro_RMSE={a_rmse:.4f}  macro_AUROC={a_auroc:.4f} (n={a_n})  score={a_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown:")
    d_q = quartile_metrics(dta_records, by="anchor_pki")
    a_q = quartile_metrics(anchor_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'DTA RMSE':>9s} {'DTA CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qd, qa in zip(d_q, a_q):
        print(f"{qd['bin']:>14s} | {qd['n']:>6d} | "
              f"{qd['rmse']:>9.4f} {qd['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)

    # Oracle upper bound.
    test_interactions_df = interactions_df[interactions_df["uniprot_id"].isin(test_uids)]
    oracle_anchor, oracle_second, oracle_anchor_pki, oracle_second_pki = compute_drug_anchors_oracle(
        test_interactions_df
    )
    oracle_test_ds = AnchorTransferDataset(
        test_interactions_df, protein_embeddings, test_uids,
        oracle_anchor, oracle_second,
        drug_to_anchor_pki=oracle_anchor_pki,
        drug_to_second_pki=oracle_second_pki,
        morgan_cache=morgan_cache,
    )
    oracle_test_loader = DataLoader(oracle_test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"Oracle test samples: {len(oracle_test_ds)} (drugs with oracle anchor: {len(oracle_anchor)})")

    dta_oracle, dta_oracle_records = evaluate_model(dta_model, oracle_test_loader, device, return_records=True)
    anchor_oracle, anchor_oracle_records = evaluate_model(anchor_model, oracle_test_loader, device, return_records=True)
    d_o_ci, d_o_rmse, d_o_score = _macro_metrics(dta_oracle)
    a_o_ci, a_o_rmse, a_o_score = _macro_metrics(anchor_oracle)
    d_o_auroc, d_o_n = _macro_auroc(dta_oracle)
    a_o_auroc, a_o_n = _macro_auroc(anchor_oracle)

    print("=" * 72)
    print("TEST SET — ORACLE ANCHORS (strongest test binder per drug, upper bound)")
    print(f"  ConciseDTA:    macro_CI={d_o_ci:.4f}  macro_RMSE={d_o_rmse:.4f}  macro_AUROC={d_o_auroc:.4f} (n={d_o_n})  score={d_o_score:.4f}")
    print(f"  ConciseAnchorBilinear: macro_CI={a_o_ci:.4f}  macro_RMSE={a_o_rmse:.4f}  macro_AUROC={a_o_auroc:.4f} (n={a_o_n})  score={a_o_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown (oracle):")
    d_oq = quartile_metrics(dta_oracle_records, by="anchor_pki")
    a_oq = quartile_metrics(anchor_oracle_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'DTA RMSE':>9s} {'DTA CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qd, qa in zip(d_oq, a_oq):
        print(f"{qd['bin']:>14s} | {qd['n']:>6d} | "
              f"{qd['rmse']:>9.4f} {qd['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
