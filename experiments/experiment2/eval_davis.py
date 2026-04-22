"""Davis-OOD eval for DrugBAN vs AnchorDrugBAN (PyG graphs + CHARPROTSET tokens)."""
from __future__ import annotations

import argparse
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.anchor_drugban import AnchorDrugBAN
from anchor_transfer.model.drugban import DrugBAN

from experiments.experiment2.dataset import (
    AnchorTransferDataset,
    build_graph_cache,
    build_protein_token_cache,
    collate_fn,
    compute_drug_anchors_oracle,
    compute_drug_anchors_tanimoto,
)
from experiments.experiment2.eval import evaluate_model, quartile_metrics
from experiments.experiment2.main import (
    DATA_DIR,
    DRUGBAN_CONFIG,
    INTERACTIONS_PATHS,
    MERGED_SEQUENCES_PATH,
)
from experiments.experiment2.split import mmseqs_cluster, ood_split
from experiments.experiment2.train import CHECKPOINT_DIR, _macro_auroc, _macro_metrics

DAVIS_PATH = DATA_DIR / "raw" / "davis_benchmark.csv"


def get_non_homologs(
    training_proteins: set[str],
    eval_proteins: set[str],
    sequences: dict[str, str],
    identity: float = 0.3,
    coverage: float = 0.8,
) -> set[str]:
    """Eval proteins not homologous to any training protein at given identity/coverage."""
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

    interactions_df = pd.read_csv(interactions_path)
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)

    src_uniprots = set(interactions_df["uniprot_id"].unique())
    src_sequences = {uid: all_sequences[uid] for uid in src_uniprots if uid in all_sequences}

    train_uids, _, _ = ood_split(
        src_sequences,
        test_frac=0.1, val_frac=0.1,
        test_identity=0.3, val_identity=0.5,
        seed=42,
    )

    davis_df = pd.read_csv(DAVIS_PATH).rename(
        columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles"}
    )
    davis_uniprots = dict(
        zip(davis_df["uniprot_id"].astype(str), davis_df["protein_sequence"].astype(str))
    )

    combined_seqs = {uid: src_sequences[uid] for uid in train_uids if uid in src_sequences}
    combined_seqs.update(davis_uniprots)
    ood_davis = get_non_homologs(
        training_proteins=train_uids,
        eval_proteins=set(davis_uniprots.keys()),
        sequences=combined_seqs,
        identity=0.3,
    )
    print(f"Davis proteins: {len(davis_uniprots)} total → {len(ood_davis)} OOD vs {args.dataset} train (≤30% identity)")

    davis_drugs = set(davis_df["ligand_smiles"].astype(str))
    drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki = compute_drug_anchors_tanimoto(
        interactions_df,
        train_uniprots=train_uids,
        eval_drugs=davis_drugs,
        tanimoto_threshold=0.7,
        pki_threshold=7.0,
    )
    print(f"Davis drugs: {len(davis_drugs)} total → {len(drug_to_anchor)} with Tanimoto anchors (≥0.7 similarity)")

    # Protein tokens for both OOD Davis queries and the source-dataset anchor proteins.
    anchor_uids = set(drug_to_anchor.values()) | set(drug_to_second.values())
    query_seqs = {uid: davis_uniprots[uid] for uid in ood_davis}
    anchor_seqs = {uid: src_sequences[uid] for uid in anchor_uids if uid in src_sequences}
    protein_tokens = build_protein_token_cache({**query_seqs, **anchor_seqs})

    graph_cache = build_graph_cache(list(davis_drugs))
    print(f"Graph cache: {len(graph_cache)} / {len(davis_drugs)} Davis SMILES featurized")

    davis_ds = AnchorTransferDataset(
        interactions_df=davis_df,
        protein_tokens=protein_tokens,
        split_uniprots=ood_davis,
        drug_to_anchor=drug_to_anchor,
        drug_to_second=drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
        graph_cache=graph_cache,
    )
    davis_loader = DataLoader(davis_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Davis-OOD evaluable samples: {len(davis_ds)}")

    drugban_model = DrugBAN(**DRUGBAN_CONFIG).to(device)
    anchor_model = AnchorDrugBAN(**DRUGBAN_CONFIG).to(device)
    drugban_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "drugban_best.pt", map_location=device, weights_only=True)
    )
    anchor_model.load_state_dict(
        torch.load(CHECKPOINT_DIR / "anchor_drugban_best.pt", map_location=device, weights_only=True)
    )

    drugban_metrics, drugban_records = evaluate_model(drugban_model, davis_loader, device, return_records=True)
    anchor_metrics, anchor_records = evaluate_model(anchor_model, davis_loader, device, return_records=True)

    d_ci, d_rmse, d_score = _macro_metrics(drugban_metrics)
    a_ci, a_rmse, a_score = _macro_metrics(anchor_metrics)
    d_auroc, d_n = _macro_auroc(drugban_metrics)
    a_auroc, a_n = _macro_auroc(anchor_metrics)

    print("=" * 72)
    print(f"DAVIS-OOD (≤30% identity to {args.dataset} train, best checkpoints)")
    print(f"  DrugBAN:       macro_CI={d_ci:.4f}  macro_RMSE={d_rmse:.4f}  macro_AUROC={d_auroc:.4f} (n={d_n})  score={d_score:.4f}")
    print(f"  AnchorDrugBAN: macro_CI={a_ci:.4f}  macro_RMSE={a_rmse:.4f}  macro_AUROC={a_auroc:.4f} (n={a_n})  score={a_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown:")
    d_q = quartile_metrics(drugban_records, by="anchor_pki")
    a_q = quartile_metrics(anchor_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'DB RMSE':>9s} {'DB CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qd, qa in zip(d_q, a_q):
        print(f"{qd['bin']:>14s} | {qd['n']:>6d} | "
              f"{qd['rmse']:>9.4f} {qd['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)

    # Oracle upper bound: strongest Davis binder per drug as the anchor.
    # Anchors are Davis proteins themselves — tokens built for all Davis uniprots.
    oracle_anchor, oracle_second, oracle_anchor_pki, oracle_second_pki = compute_drug_anchors_oracle(
        davis_df
    )
    oracle_anchor_uids = set(oracle_anchor.values()) | set(oracle_second.values())
    oracle_anchor_seqs = {uid: davis_uniprots[uid] for uid in oracle_anchor_uids if uid in davis_uniprots}
    oracle_protein_tokens = {**protein_tokens, **build_protein_token_cache(oracle_anchor_seqs)}

    oracle_ds = AnchorTransferDataset(
        interactions_df=davis_df,
        protein_tokens=oracle_protein_tokens,
        split_uniprots=ood_davis,
        drug_to_anchor=oracle_anchor,
        drug_to_second=oracle_second,
        drug_to_anchor_pki=oracle_anchor_pki,
        drug_to_second_pki=oracle_second_pki,
        graph_cache=graph_cache,
    )
    oracle_loader = DataLoader(oracle_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)
    print(f"Davis-OOD oracle samples: {len(oracle_ds)} (drugs with oracle anchor: {len(oracle_anchor)})")

    drugban_o_metrics, drugban_o_records = evaluate_model(drugban_model, oracle_loader, device, return_records=True)
    anchor_o_metrics, anchor_o_records = evaluate_model(anchor_model, oracle_loader, device, return_records=True)

    d_o_ci, d_o_rmse, d_o_score = _macro_metrics(drugban_o_metrics)
    a_o_ci, a_o_rmse, a_o_score = _macro_metrics(anchor_o_metrics)
    d_o_auroc, d_o_n = _macro_auroc(drugban_o_metrics)
    a_o_auroc, a_o_n = _macro_auroc(anchor_o_metrics)

    print("=" * 72)
    print(f"DAVIS-OOD — ORACLE ANCHORS (strongest Davis binder per drug, upper bound)")
    print(f"  DrugBAN:       macro_CI={d_o_ci:.4f}  macro_RMSE={d_o_rmse:.4f}  macro_AUROC={d_o_auroc:.4f} (n={d_o_n})  score={d_o_score:.4f}")
    print(f"  AnchorDrugBAN: macro_CI={a_o_ci:.4f}  macro_RMSE={a_o_rmse:.4f}  macro_AUROC={a_o_auroc:.4f} (n={a_o_n})  score={a_o_score:.4f}")
    print("-" * 72)
    print("Anchor-pKi quartile breakdown (oracle):")
    d_oq = quartile_metrics(drugban_o_records, by="anchor_pki")
    a_oq = quartile_metrics(anchor_o_records, by="anchor_pki")
    print(f"{'bin':>14s} | {'n':>6s} | {'DB RMSE':>9s} {'DB CI':>7s} | {'Anc RMSE':>9s} {'Anc CI':>7s}")
    for qd, qa in zip(d_oq, a_oq):
        print(f"{qd['bin']:>14s} | {qd['n']:>6d} | "
              f"{qd['rmse']:>9.4f} {qd['ci']:>7.4f} | "
              f"{qa['rmse']:>9.4f} {qa['ci']:>7.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
