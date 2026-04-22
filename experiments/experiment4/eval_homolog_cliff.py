"""Protein-side activity-cliff eval: same drug, homologous proteins, divergent pKi.

Tests whether ConciseAnchorBilinear has learned protein-side specificity
(not just "homologs bind alike") by evaluating on pairs (drug, p_a, p_b)
where p_a and p_b cluster together at ≥identity sequence identity but
have |pki_a − pki_b| ≥ min_delta in the Davis benchmark.

Anchor strategy (b): external anchor = Tanimoto-retrieved strong binder
from DTC training — simulates real-world use where the anchor is
distant from either member of the cliff pair.

Uses DTC checkpoints from experiment3 (no retraining).
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear
from anchor_transfer.model.concise_dta import ConciseDTA

from experiments.experiment4.dataset import (
    RAYGUN_DIM,
    AnchorTransferDataset,
    build_morgan_cache,
    build_raygun_cache,
    collate_fn,
    compute_drug_anchors_tanimoto,
)
from experiments.experiment4.eval import evaluate_model
from experiments.experiment4.main import (
    DATA_DIR,
    DROPOUT,
    INTERACTIONS_PATHS,
    MERGED_SEQUENCES_PATH,
    N_HEADS,
    PROJ_DIM,
)
from experiments.experiment4.split import mmseqs_cluster, ood_split

DAVIS_PATH = DATA_DIR / "raw" / "davis_benchmark.csv"
EXP3_CHECKPOINTS = Path(__file__).parent.parent / "experiment3" / "checkpoints"


def find_homolog_cliffs(
    davis_df: pd.DataFrame,
    sequences: dict[str, str],
    identity: float,
    min_delta: float,
) -> list[tuple[str, str, str, float, float]]:
    """Return (drug, high_pki_uid, low_pki_uid, high_pki, low_pki) tuples."""
    member_to_rep = mmseqs_cluster(sequences, identity=identity, coverage=0.8)
    pairs: list[tuple[str, str, str, float, float]] = []
    for drug, group in davis_df.groupby("ligand_smiles"):
        rows = [
            (r.uniprot_id, float(r.pki))
            for r in group.itertuples()
            if r.uniprot_id in member_to_rep
        ]
        by_rep: dict[str, list[tuple[str, float]]] = {}
        for uid, pki in rows:
            by_rep.setdefault(member_to_rep[uid], []).append((uid, pki))
        for members in by_rep.values():
            if len(members) < 2:
                continue
            for (ua, pa), (ub, pb) in itertools.combinations(members, 2):
                if abs(pa - pb) < min_delta:
                    continue
                if pa >= pb:
                    pairs.append((drug, ua, ub, pa, pb))
                else:
                    pairs.append((drug, ub, ua, pb, pa))
    return pairs


def pair_metrics(preds: dict[tuple[str, str], float],
                 pairs: list[tuple[str, str, str, float, float]],
                 label: str) -> None:
    correct = 0
    total = 0
    true_deltas: list[float] = []
    pred_deltas: list[float] = []
    squared_errors: list[float] = []
    for drug, u_hi, u_lo, pki_hi, pki_lo in pairs:
        k_hi, k_lo = (drug, u_hi), (drug, u_lo)
        if k_hi not in preds or k_lo not in preds:
            continue
        p_hi, p_lo = preds[k_hi], preds[k_lo]
        if p_hi > p_lo:
            correct += 1
        total += 1
        true_deltas.append(pki_hi - pki_lo)
        pred_deltas.append(p_hi - p_lo)
        squared_errors.append((p_hi - pki_hi) ** 2)
        squared_errors.append((p_lo - pki_lo) ** 2)
    if not total:
        print(f"{label}:  no evaluable pairs")
        return
    acc = correct / total
    corr = float(np.corrcoef(true_deltas, pred_deltas)[0, 1]) if total > 1 else float("nan")
    rmse = float(np.sqrt(np.mean(squared_errors)))
    mean_true_delta = float(np.mean(np.abs(true_deltas)))
    mean_pred_delta = float(np.mean(np.abs(pred_deltas)))
    print(f"{label}:  acc={acc:.4f} (n={total})  "
          f"Δpki corr={corr:+.4f}  side RMSE={rmse:.4f}  "
          f"|true Δ|={mean_true_delta:.2f}  |pred Δ|={mean_pred_delta:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--identity", type=float, default=0.5,
                        help="MMseqs2 identity threshold for homology cluster (default 0.5).")
    parser.add_argument("--min-delta", type=float, default=2.0,
                        help="Minimum |Δpki| between paired homologs (default 2.0).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}  |  identity≥{args.identity}, |Δpki|≥{args.min_delta}")

    dtc_df = pd.read_csv(INTERACTIONS_PATHS["dtc"])
    with open(MERGED_SEQUENCES_PATH, "r") as f:
        all_sequences = json.load(f)
    dtc_uniprots = set(dtc_df["uniprot_id"].unique())
    dtc_sequences = {uid: all_sequences[uid] for uid in dtc_uniprots if uid in all_sequences}
    train_uids, _, _ = ood_split(
        dtc_sequences,
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

    pairs = find_homolog_cliffs(davis_df, davis_uniprots,
                                identity=args.identity, min_delta=args.min_delta)
    print(f"Homolog-cliff pairs found: {len(pairs)}")
    if not pairs:
        return

    pair_drugs = {p[0] for p in pairs}
    drug_to_anchor, drug_to_second, drug_to_anchor_pki, drug_to_second_pki = compute_drug_anchors_tanimoto(
        dtc_df,
        train_uniprots=train_uids,
        eval_drugs=pair_drugs,
        tanimoto_threshold=0.7,
        pki_threshold=7.0,
    )
    print(f"Drugs with Tanimoto anchors (≥0.7): {len(drug_to_anchor)} / {len(pair_drugs)}")

    valid_pairs = [p for p in pairs if p[0] in drug_to_anchor]
    print(f"Pairs with external anchor available: {len(valid_pairs)}")
    if not valid_pairs:
        return

    # Embeddings for all proteins we touch: pair queries + anchor proteins.
    pair_uids = set()
    for _, uh, ul, _, _ in valid_pairs:
        pair_uids.update([uh, ul])
    anchor_uids = set(drug_to_anchor.values()) | set(drug_to_second.values())

    seqs_needed: dict[str, str] = {}
    for uid in pair_uids:
        if uid in davis_uniprots:
            seqs_needed[uid] = davis_uniprots[uid]
    for uid in anchor_uids:
        if uid in dtc_sequences:
            seqs_needed[uid] = dtc_sequences[uid]
    protein_embeddings = build_raygun_cache(seqs_needed, device=device)
    print(f"Raygun embeddings ready: {len(protein_embeddings)} proteins")

    morgan_cache = build_morgan_cache([p[0] for p in valid_pairs])
    print(f"Morgan cache: {len(morgan_cache)} drugs fingerprinted")

    # One row per side of each pair; dedupe by (uniprot, smiles).
    rows: list[dict] = []
    for drug, uh, ul, pki_h, pki_l in valid_pairs:
        rows.append({"uniprot_id": uh, "ligand_smiles": drug, "pki": pki_h})
        rows.append({"uniprot_id": ul, "ligand_smiles": drug, "pki": pki_l})
    eval_df = pd.DataFrame(rows).drop_duplicates(["uniprot_id", "ligand_smiles"])

    eval_ds = AnchorTransferDataset(
        interactions_df=eval_df,
        protein_embeddings=protein_embeddings,
        split_uniprots=set(eval_df["uniprot_id"]),
        drug_to_anchor=drug_to_anchor,
        drug_to_second=drug_to_second,
        drug_to_anchor_pki=drug_to_anchor_pki,
        drug_to_second_pki=drug_to_second_pki,
        morgan_cache=morgan_cache,
    )
    print(f"Eval samples (pair sides): {len(eval_ds)}")
    loader = DataLoader(eval_ds, batch_size=256, shuffle=False, collate_fn=collate_fn)

    dta_model = ConciseDTA(
        esm_dim=RAYGUN_DIM, proj_dim=PROJ_DIM, nheads=N_HEADS, dropout=DROPOUT
    ).to(device)
    anchor_model = ConciseAnchorBilinear(
        residue_dim=RAYGUN_DIM, proj_dim=PROJ_DIM, dropout=DROPOUT
    ).to(device)
    dta_model.load_state_dict(
        torch.load(EXP3_CHECKPOINTS / "concise_dta_best.pt", map_location=device, weights_only=True)
    )
    anchor_model.load_state_dict(
        torch.load(EXP3_CHECKPOINTS / "concise_anchor_best.pt", map_location=device, weights_only=True)
    )

    _, dta_records = evaluate_model(dta_model, loader, device, return_records=True)
    _, anchor_records = evaluate_model(anchor_model, loader, device, return_records=True)

    dta_pred = {(r["drug_id"], r["protein_id"]): r["pred"] for r in dta_records}
    anc_pred = {(r["drug_id"], r["protein_id"]): r["pred"] for r in anchor_records}

    print("=" * 72)
    print(f"HOMOLOG-CLIFF PAIRS  (identity≥{args.identity}, |Δpki|≥{args.min_delta})")
    print(f"  metric = pair-ranking accuracy (did model predict higher side correctly)")
    print("-" * 72)
    pair_metrics(dta_pred, valid_pairs, "ConciseDTA           ")
    pair_metrics(anc_pred, valid_pairs, "ConciseAnchorBilinear")
    print("=" * 72)


if __name__ == "__main__":
    main()
