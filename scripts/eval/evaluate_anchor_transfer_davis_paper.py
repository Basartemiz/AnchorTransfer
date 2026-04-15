#!/usr/bin/env python3
"""Paper-protocol Davis evaluation for anchor-transfer models.

This script reproduces the Davis cross-dataset protocol described in the paper:
1. Recreate the DTC 80/10/10 protein split with seed 42 and keep only the
   training-protein subset as the retrieval pool.
2. For each DTC drug, keep the strongest binder with pKi >= 7 as the anchor.
3. Exclude any DTC anchor drug whose canonical SMILES matches a Davis drug.
4. Retrieve anchors for Davis drugs by nearest chirality-aware Morgan Tanimoto.
5. Skip self-anchors by sequence-equivalent DTC UniProt IDs.
6. Report macro per-protein CI, AUROC, AUPRC, and RMSE with bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from anchor_transfer.model.anchor_transfer import AnchorTransferDTA, encode_smiles

try:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
except ImportError as exc:  # pragma: no cover - environment-specific
    raise SystemExit(
        "RDKit is required for the paper Davis protocol. Install it in the reproduction venv."
    ) from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
RDLogger.DisableLog("rdApp.*")


def sanitize_esm2_embeddings(embeddings: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    clean = {}
    dropped = 0
    for protein_id, tensor in embeddings.items():
        if torch.isfinite(tensor).all():
            clean[protein_id] = tensor.float()
        else:
            dropped += 1
    if dropped:
        logger.warning("Dropped %d proteins with non-finite ESM-2 embeddings", dropped)
    return clean


def normalize_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    if "protein_name" in frame.columns and "uniprot_id" not in frame.columns:
        rename_map["protein_name"] = "uniprot_id"
    if "drug_smiles" in frame.columns and "ligand_smiles" not in frame.columns:
        rename_map["drug_smiles"] = "ligand_smiles"
    if "pKd" in frame.columns and "pki" not in frame.columns:
        rename_map["pKd"] = "pki"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    required = {"uniprot_id", "ligand_smiles", "pki"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Benchmark CSV is missing required columns after normalization: {', '.join(missing)}"
        )
    return frame


def load_model(model_path: Path, esm2_dim: int, model_version: str, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if model_version == "v2":
        from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2

        saved_args = checkpoint.get("args", {})
        model = AnchorTransferDTAv2(
            esm2_dim=esm2_dim,
            proj_dim=saved_args.get("proj_dim", 256),
        ).to(device)
    else:
        model = AnchorTransferDTA(esm2_dim=esm2_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded %s model from epoch %d", model_version, checkpoint.get("epoch", -1))
    return model


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    n = len(y_true)
    if n < 2:
        return float("nan")
    max_pairs = 100_000
    if n * (n - 1) // 2 > max_pairs:
        idx_i = np.random.randint(0, n, max_pairs)
        idx_j = np.random.randint(0, n, max_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    else:
        idx_i, idx_j = np.triu_indices(n, k=1)
    diff_true = y_true[idx_i] - y_true[idx_j]
    diff_pred = y_pred[idx_i] - y_pred[idx_j]
    valid = diff_true != 0
    if not np.any(valid):
        return float("nan")
    concordant = np.sum((diff_true[valid] * diff_pred[valid]) > 0)
    return float(concordant / np.sum(valid))


def canonicalize_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)


def build_dtc_training_pool(
    train_df: pd.DataFrame,
    valid_proteins: set[str],
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    dtc_valid = train_df[train_df["uniprot_id"].isin(valid_proteins)].copy()
    proteins = sorted(dtc_valid["uniprot_id"].unique())
    random.Random(seed).shuffle(proteins)

    n_test = max(1, int(len(proteins) * test_fraction))
    n_val = max(1, int(len(proteins) * val_fraction))
    train_proteins = set(proteins[n_test + n_val :])
    dtc_train = dtc_valid[dtc_valid["uniprot_id"].isin(train_proteins)].copy()

    logger.info(
        "DTC retrieval pool: %d interactions, %d proteins (seed=%d, split=%.0f/%.0f/%.0f)",
        len(dtc_train),
        dtc_train["uniprot_id"].nunique(),
        seed,
        (1.0 - val_fraction - test_fraction) * 100,
        val_fraction * 100,
        test_fraction * 100,
    )
    return dtc_train


def build_anchor_pool(
    dtc_train: pd.DataFrame,
    esm2_dict: dict[str, torch.Tensor],
    excluded_canonical_smiles: set[str],
    anchor_threshold: float,
):
    anchor_pool = {}
    for smiles, group in dtc_train.groupby("ligand_smiles"):
        strongest = group.sort_values("pki", ascending=False).iloc[0]
        anchor_uid = strongest["uniprot_id"]
        anchor_pki = float(strongest["pki"])
        if anchor_pki < anchor_threshold or anchor_uid not in esm2_dict:
            continue
        canonical = canonicalize_smiles(smiles)
        if canonical and canonical in excluded_canonical_smiles:
            continue
        fp = smiles_to_fp(smiles)
        if fp is None:
            continue
        anchor_pool[smiles] = {
            "anchor_uid": anchor_uid,
            "anchor_pki": anchor_pki,
            "canonical_smiles": canonical,
            "fingerprint": fp,
        }
    logger.info(
        "DTC anchor pool after canonical exclusion: %d drugs (anchor pKi >= %.1f)",
        len(anchor_pool),
        anchor_threshold,
    )
    return anchor_pool


def nearest_anchor_drugs(
    benchmark_drugs: list[str],
    anchor_pool: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    anchor_smiles = list(anchor_pool.keys())
    anchor_fps = [anchor_pool[smiles]["fingerprint"] for smiles in anchor_smiles]
    nearest = {}
    similarities = []

    for smiles in benchmark_drugs:
        query_fp = smiles_to_fp(smiles)
        if query_fp is None or not anchor_fps:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, anchor_fps)
        best_idx = int(np.argmax(sims))
        nearest[smiles] = {
            "anchor_drug_smiles": anchor_smiles[best_idx],
            "tanimoto": float(sims[best_idx]),
        }
        similarities.append(float(sims[best_idx]))

    if similarities:
        logger.info(
            "Nearest-anchor Tanimoto stats: min=%.3f median=%.3f mean=%.3f max=%.3f",
            min(similarities),
            float(np.median(similarities)),
            float(np.mean(similarities)),
            max(similarities),
        )
    return nearest


def bootstrap_mean_ci(values: list[float], seed: int, samples: int) -> tuple[float, float, float]:
    valid = np.array([value for value in values if not np.isnan(value)], dtype=np.float64)
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for idx in range(samples):
        draws = rng.choice(valid, size=len(valid), replace=True)
        means[idx] = np.mean(draws)
    return (
        float(np.mean(valid)),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    )


def summarize_tanimoto_bins(predictions_df: pd.DataFrame, pos_threshold: float, neg_threshold: float) -> list[dict[str, float | int | str]]:
    bins = [
        (0.0, 0.6, "[0.0,0.6)"),
        (0.6, 0.8, "[0.6,0.8)"),
        (0.8, 0.95, "[0.8,0.95)"),
        (0.95, 1.001, "[0.95,1.0]"),
    ]
    rows = []
    for low, high, label in bins:
        subset = predictions_df[(predictions_df["tanimoto"] >= low) & (predictions_df["tanimoto"] < high)].copy()
        if subset.empty:
            continue
        protein_metrics = []
        for uid, group in subset.groupby("uniprot_id"):
            true = group["true_pki"].to_numpy(dtype=np.float32)
            pred = group["pred_pki"].to_numpy(dtype=np.float32)
            prob = group["binding_prob"].to_numpy(dtype=np.float32)
            ci = concordance_index(true, pred)
            rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
            pearson_r = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else float("nan")
            pos_mask = true >= pos_threshold
            neg_mask = true <= neg_threshold
            cls_mask = pos_mask | neg_mask
            auroc = float("nan")
            auprc = float("nan")
            if cls_mask.sum() >= 2:
                cls_labels = pos_mask[cls_mask].astype(int)
                cls_prob = prob[cls_mask]
                if len(np.unique(cls_labels)) == 2:
                    auroc = float(roc_auc_score(cls_labels, cls_prob))
                    auprc = float(average_precision_score(cls_labels, cls_prob))
            protein_metrics.append((ci, rmse, auroc, auprc, pearson_r))

        ci_vals = [m[0] for m in protein_metrics if not np.isnan(m[0])]
        rmse_vals = [m[1] for m in protein_metrics if not np.isnan(m[1])]
        auroc_vals = [m[2] for m in protein_metrics if not np.isnan(m[2])]
        auprc_vals = [m[3] for m in protein_metrics if not np.isnan(m[3])]
        pearson_vals = [m[4] for m in protein_metrics if not np.isnan(m[4])]

        rows.append({
            "bin": label,
            "n_pairs": int(len(subset)),
            "n_drugs": int(subset["ligand_smiles"].nunique()),
            "n_proteins": int(subset["uniprot_id"].nunique()),
            "ci": float(np.mean(ci_vals)) if ci_vals else float("nan"),
            "rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "pearson_r": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
            "auroc": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
            "auprc": float(np.mean(auprc_vals)) if auprc_vals else float("nan"),
        })
    return rows


def summarize_anchor_quartiles(predictions_df: pd.DataFrame, pos_threshold: float, neg_threshold: float) -> list[dict]:
    """Break down per-protein metrics by anchor pKi quartile (Q1-Q4)."""
    quartile_edges = predictions_df["anchor_pki"].quantile([0, 0.25, 0.5, 0.75, 1.0]).values
    labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
    rows = []
    for i in range(4):
        low, high = quartile_edges[i], quartile_edges[i + 1]
        if i < 3:
            subset = predictions_df[(predictions_df["anchor_pki"] >= low) & (predictions_df["anchor_pki"] < high)]
        else:
            subset = predictions_df[predictions_df["anchor_pki"] >= low]
        if subset.empty:
            continue
        protein_metrics = []
        for uid, group in subset.groupby("uniprot_id"):
            true = group["true_pki"].to_numpy(dtype=np.float32)
            pred = group["pred_pki"].to_numpy(dtype=np.float32)
            prob = group["binding_prob"].to_numpy(dtype=np.float32)
            ci = concordance_index(true, pred)
            rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
            pearson_r = float(np.corrcoef(true, pred)[0, 1]) if len(true) > 1 else float("nan")
            pos_mask = true >= pos_threshold
            neg_mask = true <= neg_threshold
            cls_mask = pos_mask | neg_mask
            auroc = float("nan")
            auprc = float("nan")
            if cls_mask.sum() >= 2:
                cls_labels = pos_mask[cls_mask].astype(int)
                cls_prob = prob[cls_mask]
                if len(np.unique(cls_labels)) == 2:
                    auroc = float(roc_auc_score(cls_labels, cls_prob))
                    auprc = float(average_precision_score(cls_labels, cls_prob))
            protein_metrics.append((ci, rmse, auroc, auprc, pearson_r))

        ci_vals = [m[0] for m in protein_metrics if not np.isnan(m[0])]
        rmse_vals = [m[1] for m in protein_metrics if not np.isnan(m[1])]
        auroc_vals = [m[2] for m in protein_metrics if not np.isnan(m[2])]
        auprc_vals = [m[3] for m in protein_metrics if not np.isnan(m[3])]
        pearson_vals = [m[4] for m in protein_metrics if not np.isnan(m[4])]

        rows.append({
            "quartile": labels[i],
            "anchor_pki_range": f"[{low:.2f}, {high:.2f}]",
            "n_pairs": int(len(subset)),
            "n_drugs": int(subset["ligand_smiles"].nunique()),
            "n_proteins": int(subset["uniprot_id"].nunique()),
            "ci": float(np.mean(ci_vals)) if ci_vals else float("nan"),
            "rmse": float(np.mean(rmse_vals)) if rmse_vals else float("nan"),
            "pearson_r": float(np.mean(pearson_vals)) if pearson_vals else float("nan"),
            "auroc": float(np.mean(auroc_vals)) if auroc_vals else float("nan"),
            "auprc": float(np.mean(auprc_vals)) if auprc_vals else float("nan"),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Evaluate anchor-transfer Davis paper protocol")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--esm2-train", required=True, help="Training ESM-2 embeddings .pt")
    parser.add_argument("--esm2-benchmark", required=True, help="Benchmark ESM-2 embeddings .pt")
    parser.add_argument("--training", required=True, help="DTC interactions CSV")
    parser.add_argument("--benchmark", required=True, help="Davis benchmark CSV")
    parser.add_argument("--dtc-proteins", required=True, help="DTC proteins CSV with uniprot_id,sequence")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--positive-threshold", type=float, default=7.0,
                        help="pKi >= this is positive for AUROC/AUPRC")
    parser.add_argument("--negative-threshold", type=float, default=5.0,
                        help="pKi <= this is negative for AUROC/AUPRC (ambiguous range excluded)")
    parser.add_argument("--anchor-threshold", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--homolog-exclude", default=None,
                        help="File with protein IDs to exclude (one per line, e.g. homologs)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading ESM-2 embeddings...")
    esm2_dict = sanitize_esm2_embeddings(
        torch.load(args.esm2_train, map_location="cpu", weights_only=False)
    )
    benchmark_esm2 = sanitize_esm2_embeddings(
        torch.load(args.esm2_benchmark, map_location="cpu", weights_only=False)
    )
    esm2_dict.update(benchmark_esm2)
    esm2_dim = next(iter(esm2_dict.values())).shape[0]
    logger.info("Combined ESM-2 embeddings: %d proteins (dim=%d)", len(esm2_dict), esm2_dim)

    model = load_model(Path(args.model), esm2_dim, args.model_version, device)

    training_df = pd.read_csv(args.training)
    dtc_train = build_dtc_training_pool(
        training_df,
        valid_proteins=set(esm2_dict.keys()),
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )

    benchmark_raw = pd.read_csv(args.benchmark)
    benchmark_df = normalize_benchmark_frame(benchmark_raw.copy())
    if "protein_sequence" not in benchmark_raw.columns:
        raise ValueError("Davis benchmark CSV must include a protein_sequence column for self-anchor checks")

    benchmark_sequences = dict(
        zip(
            benchmark_raw["protein_name"] if "protein_name" in benchmark_raw.columns else benchmark_df["uniprot_id"],
            benchmark_raw["protein_sequence"],
        )
    )
    benchmark_df = benchmark_df[benchmark_df["uniprot_id"].isin(esm2_dict)].copy()

    if args.homolog_exclude:
        exclude_ids = set(open(args.homolog_exclude).read().strip().split("\n"))
        before = len(benchmark_df)
        benchmark_df = benchmark_df[~benchmark_df["uniprot_id"].isin(exclude_ids)].copy()
        logger.info("Homolog exclusion: removed %d proteins (%d pairs), %d novel proteins remain",
                     len(exclude_ids & set(benchmark_df["uniprot_id"].unique()) | exclude_ids),
                     before - len(benchmark_df),
                     benchmark_df["uniprot_id"].nunique())

    logger.info(
        "Benchmark after filtering: %d pairs, %d proteins, %d drugs",
        len(benchmark_df),
        benchmark_df["uniprot_id"].nunique(),
        benchmark_df["ligand_smiles"].nunique(),
    )

    benchmark_canonical_smiles = {
        canonical
        for canonical in (canonicalize_smiles(smiles) for smiles in benchmark_df["ligand_smiles"].unique())
        if canonical
    }
    anchor_pool = build_anchor_pool(
        dtc_train=dtc_train,
        esm2_dict=esm2_dict,
        excluded_canonical_smiles=benchmark_canonical_smiles,
        anchor_threshold=args.anchor_threshold,
    )
    nearest = nearest_anchor_drugs(sorted(benchmark_df["ligand_smiles"].unique()), anchor_pool)

    seq_to_dtc_uids = defaultdict(set)
    dtc_proteins = pd.read_csv(args.dtc_proteins)
    if not {"uniprot_id", "sequence"}.issubset(dtc_proteins.columns):
        raise ValueError("DTC proteins CSV must include uniprot_id and sequence columns")
    for row in dtc_proteins[["uniprot_id", "sequence"]].dropna().drop_duplicates().itertuples(index=False):
        seq_to_dtc_uids[row.sequence].add(row.uniprot_id)

    eval_records = []
    for row in benchmark_df.itertuples(index=False):
        nearest_match = nearest.get(row.ligand_smiles)
        if nearest_match is None:
            continue

        benchmark_sequence = benchmark_sequences.get(row.uniprot_id)
        equivalent_dtc_uids = seq_to_dtc_uids.get(benchmark_sequence, set()) if benchmark_sequence else set()
        anchor_meta = anchor_pool[nearest_match["anchor_drug_smiles"]]
        anchor_uid = anchor_meta["anchor_uid"]
        if anchor_uid in equivalent_dtc_uids:
            continue

        eval_records.append({
            "uniprot_id": row.uniprot_id,
            "ligand_smiles": row.ligand_smiles,
            "true_pki": float(row.pki),
            "anchor_uid": anchor_uid,
            "anchor_pki": float(anchor_meta["anchor_pki"]),
            "anchor_drug_smiles": nearest_match["anchor_drug_smiles"],
            "tanimoto": float(nearest_match["tanimoto"]),
        })

    if not eval_records:
        raise RuntimeError("No Davis interactions remain after paper-protocol filtering")

    eval_df = pd.DataFrame(eval_records)
    logger.info(
        "Paper Davis eval set: %d interactions (%.1f%% coverage), %d proteins, %d drugs",
        len(eval_df),
        100.0 * len(eval_df) / len(benchmark_df),
        eval_df["uniprot_id"].nunique(),
        eval_df["ligand_smiles"].nunique(),
    )

    pred_values = np.empty(len(eval_df), dtype=np.float32)
    prob_values = np.empty(len(eval_df), dtype=np.float32)

    logger.info("Running batched inference on %d pairs (batch_size=%d)", len(eval_df), args.batch_size)
    with torch.inference_mode():
        for start in range(0, len(eval_df), args.batch_size):
            end = min(start + args.batch_size, len(eval_df))
            batch = eval_df.iloc[start:end]

            anchor_emb = torch.stack([esm2_dict[uid] for uid in batch["anchor_uid"]]).to(device, non_blocking=True)
            query_emb = torch.stack([esm2_dict[uid] for uid in batch["uniprot_id"]]).to(device, non_blocking=True)
            drug_tokens = torch.tensor(
                [encode_smiles(smiles) for smiles in batch["ligand_smiles"]],
                dtype=torch.long,
                device=device,
            )

            out = model(anchor_emb, query_emb, drug_tokens)
            pred_values[start:end] = out["pki_pred"].detach().float().cpu().numpy().reshape(-1)
            prob_values[start:end] = out["binding_prob"].detach().float().cpu().numpy().reshape(-1)

            if start == 0 or end == len(eval_df) or ((start // args.batch_size) + 1) % 10 == 0:
                logger.info("Inference progress: %d/%d pairs", end, len(eval_df))

    eval_df["pred_pki"] = pred_values
    eval_df["binding_prob"] = prob_values
    eval_df.to_csv(output_dir / "predictions.csv", index=False)

    per_protein_rows = []
    ci_values = []
    rmse_values = []
    auroc_values = []
    auprc_values = []
    pearson_values = []

    for idx, (uid, group) in enumerate(eval_df.groupby("uniprot_id"), start=1):
        true = group["true_pki"].to_numpy(dtype=np.float32)
        pred = group["pred_pki"].to_numpy(dtype=np.float32)
        prob = group["binding_prob"].to_numpy(dtype=np.float32)

        ci = concordance_index(true, pred)
        rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
        pearson_r = float(np.corrcoef(true, pred)[0, 1]) if len(group) > 1 else float("nan")

        # Paper protocol: >=7 positive, <=5 negative, exclude ambiguous 5-7 range
        pos_mask = true >= args.positive_threshold
        neg_mask = true <= args.negative_threshold
        cls_mask = pos_mask | neg_mask
        auroc = float("nan")
        auprc = float("nan")
        if cls_mask.sum() >= 2:
            cls_labels = pos_mask[cls_mask].astype(int)
            cls_prob = prob[cls_mask]
            if len(np.unique(cls_labels)) == 2:
                auroc = float(roc_auc_score(cls_labels, cls_prob))
                auprc = float(average_precision_score(cls_labels, cls_prob))

        ci_values.append(ci)
        rmse_values.append(rmse)
        pearson_values.append(pearson_r)
        auroc_values.append(auroc)
        auprc_values.append(auprc)

        per_protein_rows.append({
            "uniprot_id": uid,
            "n_drugs": int(len(group)),
            "ci": ci,
            "rmse": rmse,
            "pearson_r": pearson_r,
            "auroc": auroc,
            "auprc": auprc,
        })

        if idx % 20 == 0:
            logger.info(
                "[%d] %s (%d drugs): CI=%.3f RMSE=%.3f AUROC=%.3f",
                idx,
                uid,
                len(group),
                ci,
                rmse,
                auroc,
            )

    per_protein_df = pd.DataFrame(per_protein_rows)
    per_protein_df.to_csv(output_dir / "per_protein_results.csv", index=False)

    ci_mean, ci_lo, ci_hi = bootstrap_mean_ci(ci_values, args.seed, args.bootstrap_samples)
    auroc_mean, auroc_lo, auroc_hi = bootstrap_mean_ci(auroc_values, args.seed + 1, args.bootstrap_samples)
    auprc_mean, auprc_lo, auprc_hi = bootstrap_mean_ci(auprc_values, args.seed + 2, args.bootstrap_samples)

    summary = {
        "protocol": {
            "name": "davis_paper_cross_dataset",
            "seed": args.seed,
            "dtc_train_fraction": 1.0 - args.val_fraction - args.test_fraction,
            "dtc_val_fraction": args.val_fraction,
            "dtc_test_fraction": args.test_fraction,
            "anchor_threshold_pki": args.anchor_threshold,
            "positive_threshold_pki": args.positive_threshold,
            "negative_threshold_pki": args.negative_threshold,
            "canonical_duplicate_exclusion": True,
            "tanimoto_fingerprint": "Morgan radius=2 bits=2048 useChirality=True",
        },
        "coverage": {
            "n_input_pairs": int(len(benchmark_df)),
            "n_evaluated_pairs": int(len(eval_df)),
            "coverage_fraction": float(len(eval_df) / len(benchmark_df)),
            "n_proteins": int(eval_df["uniprot_id"].nunique()),
            "n_drugs": int(eval_df["ligand_smiles"].nunique()),
        },
        "tanimoto": {
            "min": float(eval_df["tanimoto"].min()),
            "median": float(eval_df["tanimoto"].median()),
            "mean": float(eval_df["tanimoto"].mean()),
            "max": float(eval_df["tanimoto"].max()),
        },
        "all": {
            "ci": ci_mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "rmse": float(np.nanmean(rmse_values)),
            "pearson_r": float(np.nanmean(pearson_values)),
            "auroc": auroc_mean,
            "auroc_lo": auroc_lo,
            "auroc_hi": auroc_hi,
            "auprc": auprc_mean,
            "auprc_lo": auprc_lo,
            "auprc_hi": auprc_hi,
            "n": int(len(per_protein_df)),
        },
        "tanimoto_bins": summarize_tanimoto_bins(eval_df, args.positive_threshold, args.negative_threshold),
        "anchor_quartiles": summarize_anchor_quartiles(eval_df, args.positive_threshold, args.negative_threshold),
    }

    with open(output_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("=" * 60)
    logger.info(
        "DAVIS PAPER RESULTS (n=%d proteins, %d interactions): CI=%.3f AUROC=%.3f AUPRC=%.3f RMSE=%.3f",
        summary["all"]["n"],
        summary["coverage"]["n_evaluated_pairs"],
        summary["all"]["ci"],
        summary["all"]["auroc"],
        summary["all"]["auprc"],
        summary["all"]["rmse"],
    )
    for q in summary["anchor_quartiles"]:
        logger.info(
            "  %s %s: CI=%.3f AUROC=%.3f AUPRC=%.3f RMSE=%.3f (n_pairs=%d)",
            q["quartile"], q["anchor_pki_range"],
            q["ci"], q["auroc"], q["auprc"], q["rmse"], q["n_pairs"],
        )
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
