# eval on davis dataset using trained models

from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
from anchor_transfer.model.esm_dta import EsmDTAModel

from experiments.experiment1.dataset import (
    AnchorTransferDataset,
    collate_fn,
    compute_drug_anchors_tanimoto,
    get_esm2_embeddings,
)
from experiments.experiment1.eval import evaluate_model as eval_model
from experiments.experiment1.eval import quartile_metrics
from experiments.experiment1.main import (
    DATA_DIR,
    INTERACTIONS_PATHS,
    MERGED_SEQUENCES_PATH,
)
from experiments.experiment1.split import mmseqs_cluster, ood_split
from experiments.experiment1.train import CHECKPOINT_DIR, _macro_auroc, _macro_metrics

DAVIS_PATH = DATA_DIR / "raw" / "davis_benchmark.csv"

__all__ = ["eval_model", "get_non_homologs", "main"]


def get_non_homologs(
    training_proteins: set[str],
    eval_proteins: set[str],
    sequences: dict[str, str],
    identity: float = 0.3,
    coverage: float = 0.8,
) -> set[str]:
    """Return eval protein IDs not homologous to any training protein.

    Homology = sharing an MMseqs cluster at `identity` fraction identity over
    `coverage` fraction of the longer sequence. Proteins whose cluster contains
    any training protein are filtered out.

    `sequences` must contain entries for every id in `training_proteins ∪ eval_proteins`
    that should be considered; missing ids are dropped.
    """
    relevant = training_proteins | eval_proteins
    all_seqs = {uid: sequences[uid] for uid in relevant if uid in sequences}

    member_to_rep = mmseqs_cluster(all_seqs, identity=identity, coverage=coverage)

    training_clusters = {
        member_to_rep[uid] for uid in training_proteins if uid in member_to_rep
    }

    return {
        uid for uid in eval_proteins
        if uid in member_to_rep and member_to_rep[uid] not in training_clusters
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(INTERACTIONS_PATHS), default="dtc",
                        help="Training-source anchors + OOD split to replay (dtc or bdb).")
    args = parser.parse_args()
    interactions_path = INTERACTIONS_PATHS[args.dataset]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  anchors from: {args.dataset} ({interactions_path.name})")

    # Training interactions — replay the OOD split and build the anchor map.
    interactions_df = pd.read_csv(interactions_path)
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)

    dtc_uniprots = set(interactions_df["uniprot_id"].unique())
    dtc_sequences = {uid: all_sequences[uid] for uid in dtc_uniprots if uid in all_sequences}

    # Replay the same split used for training (seed=42) to recover train_uids.
    train_uids, _, _ = ood_split(
        dtc_sequences,
        test_frac=0.1, val_frac=0.1,
        test_identity=0.3, val_identity=0.5,
        seed=42,
    )

    # Davis benchmark — rename columns to match DTC schema so AnchorTransferDataset works.
    davis_df = pd.read_csv(DAVIS_PATH).rename(
        columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles"}
    )
    davis_uniprots = dict(
        zip(davis_df["uniprot_id"].astype(str), davis_df["protein_sequence"].astype(str))
    )

    # OOD filter: keep only Davis proteins ≤30% identity to any DTC train protein.
    combined_seqs = {uid: dtc_sequences[uid] for uid in train_uids if uid in dtc_sequences}
    combined_seqs.update(davis_uniprots)
    ood_davis = get_non_homologs(
        training_proteins=train_uids,
        eval_proteins=set(davis_uniprots.keys()),
        sequences=combined_seqs,
        identity=0.3,
    )
    print(f"Davis proteins: {len(davis_uniprots)} total → {len(ood_davis)} OOD vs DTC train (≤30% identity)")

    # Tanimoto-retrieved anchors (Davis drugs don't overlap with DTC training drugs).
    # Use the same filters as eval_test_tanimoto: ≥0.5 Tanimoto similarity, anchor pKi ≥ 7.
    davis_drugs = set(davis_df["ligand_smiles"].astype(str))
    drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki = compute_drug_anchors_tanimoto(
        interactions_df,
        train_uniprots=train_uids,
        eval_drugs=davis_drugs,
        tanimoto_threshold=0.7,
        pki_threshold=7.0,
    )
    print(f"Davis drugs: {len(davis_drugs)} total → {len(drug_to_anchor)} with Tanimoto anchors (≥0.7 similarity)")

    # ESM-2 embeddings for both the OOD Davis queries and the DTC anchor proteins.
    anchor_uids = set(drug_to_anchor.values()) | set(drug_to_second.values())
    query_seqs = {uid: davis_uniprots[uid] for uid in ood_davis}
    anchor_seqs = {uid: dtc_sequences[uid] for uid in anchor_uids if uid in dtc_sequences}
    esm2_embeddings = get_esm2_embeddings({**query_seqs, **anchor_seqs})

    # Dataset drops rows whose drug has no DTC-train anchor, keeping only drugs
    # with a valid anchor available — both models see identical samples.
    davis_ds = AnchorTransferDataset(
        interactions_df=davis_df,
        esm2_embeddings=esm2_embeddings,
        split_uniprots=ood_davis,
        drug_to_anchor=drug_to_anchor,
        drug_to_second=drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
    )
    davis_loader = DataLoader(davis_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Davis-OOD evaluable samples: {len(davis_ds)}")

    # Load best checkpoints from training.
    esm_dta_model = EsmDTAModel(esm2_dim=1280, prot_proj_dim=128).to(device)
    anchor_dta_model = AnchorTransferDTAv2(esm2_dim=1280, proj_dim=128, dropout=0.3).to(device)
    esm_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "esm_dta_best.pt", map_location=device, weights_only=True)
    )
    anchor_dta_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "anchor_dta_best.pt", map_location=device, weights_only=True)
    )

    esm_metrics, esm_records = eval_model(esm_dta_model, davis_loader, device, return_records=True)
    anchor_metrics, anchor_records = eval_model(anchor_dta_model, davis_loader, device, return_records=True)

    esm_ci, esm_rmse, esm_score = _macro_metrics(esm_metrics)
    anchor_ci, anchor_rmse, anchor_score = _macro_metrics(anchor_metrics)
    esm_auroc, esm_n_auroc = _macro_auroc(esm_metrics)
    anchor_auroc, anchor_n_auroc = _macro_auroc(anchor_metrics)

    print("=" * 72)
    print("DAVIS-OOD (≤30% identity to DTC train, best checkpoints)")
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
