#!/usr/bin/env python3
"""Evaluate the Anchor Transfer DTA model on external benchmarks.

For each benchmark (query_protein, drug) pair:
1. Find anchor candidates from training set via drug->binders lookup
2. If no drug-based anchor, fall back to ESM-2 cosine similarity
3. Feed (anchor_esm2, query_esm2, drug_smiles) -> model -> pKi + binding prob
4. Compute CI, RMSE, Pearson r, AUROC per protein and overall

Supports both v1 (AnchorTransferDTA) and v2 (AnchorTransferDTAv2) models.

Usage:
  python scripts/evaluate_anchor_transfer.py \
    --model models/anchor_transfer_v2/best_model.pt \
    --model-version v2 \
    --esm2 data/processed/esm2_35m_dtc_proteins.pt \
    --benchmark data/raw/davis_benchmark.csv \
    --training data/processed/dtc_training_interactions.csv \
    --output-dir results/v2_35m_davis \
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from idr_gat.model.anchor_transfer import AnchorTransferDTA, encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def concordance_index(y_true, y_pred):
    """Fast vectorized concordance index."""
    n = len(y_true)
    if n < 2:
        return 0.5
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    max_pairs = 100_000
    if n * (n - 1) // 2 > max_pairs:
        idx_i = np.random.randint(0, n, max_pairs)
        idx_j = np.random.randint(0, n, max_pairs)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
    else:
        idx = np.triu_indices(n, k=1)
        idx_i, idx_j = idx[0], idx[1]
    diff_true = y_true[idx_i] - y_true[idx_j]
    diff_pred = y_pred[idx_i] - y_pred[idx_j]
    tied = diff_true == 0
    concordant = ((diff_true * diff_pred) > 0).sum()
    total = (~tied).sum()
    return float(concordant / total) if total > 0 else 0.5


def find_anchor_by_drug(drug_smiles, drug_to_binders, esm2_dict):
    """Find training proteins that bind this drug and have ESM-2 embeddings."""
    binders = drug_to_binders.get(drug_smiles, [])
    return [b for b in binders if b in esm2_dict]


def find_anchor_by_similarity(query_id, training_proteins, esm2_dict):
    """Find most similar training protein by ESM-2 cosine similarity."""
    if query_id not in esm2_dict:
        return None
    query_emb = esm2_dict[query_id]
    best_sim, best_anchor = -1, None
    for pid in training_proteins:
        if pid not in esm2_dict or pid == query_id:
            continue
        sim = F.cosine_similarity(query_emb.unsqueeze(0), esm2_dict[pid].unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_anchor = pid
    return best_anchor


def load_model(model_path, esm2_dim, model_version, device):
    """Load model checkpoint (v1 or v2)."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if model_version == "v2":
        from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
        saved_args = checkpoint.get("args", {})
        model = AnchorTransferDTAv2(
            esm2_dim=esm2_dim,
            proj_dim=saved_args.get("proj_dim", 256),
        ).to(device)
    else:
        model = AnchorTransferDTA(esm2_dim=esm2_dim).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded %s model from epoch %d", model_version,
                checkpoint.get("epoch", -1))
    return model


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
    if "target_uniprot_id" in frame.columns and "uniprot_id" not in frame.columns:
        rename_map["target_uniprot_id"] = "uniprot_id"
    if "Target_ID" in frame.columns and "uniprot_id" not in frame.columns:
        rename_map["Target_ID"] = "uniprot_id"
    if rename_map:
        frame = frame.rename(columns=rename_map)

    required = {"uniprot_id", "ligand_smiles", "pki"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            f"Benchmark CSV is missing required columns after normalization: {', '.join(missing)}"
        )
    return frame


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anchor Transfer DTA")
    parser.add_argument("--model", required=True, help="Path to best_model.pt")
    parser.add_argument("--model-version", choices=["v1", "v2"], default="v1",
                        help="Model architecture version")
    parser.add_argument("--esm2", required=True, help="ESM-2 embeddings .pt (training)")
    parser.add_argument("--esm2-benchmark", default=None,
                        help="ESM-2 embeddings for benchmark proteins")
    parser.add_argument("--benchmark", required=True, help="Benchmark CSV")
    parser.add_argument("--training", required=True, help="Training interactions CSV")
    parser.add_argument("--output-dir", default="results/anchor_transfer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--binder-threshold", type=float, default=7.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ESM-2 embeddings
    logger.info("Loading ESM-2 embeddings...")
    esm2_dict = sanitize_esm2_embeddings(
        torch.load(args.esm2, map_location="cpu", weights_only=False)
    )
    if args.esm2_benchmark:
        bench_emb = sanitize_esm2_embeddings(
            torch.load(args.esm2_benchmark, map_location="cpu", weights_only=False)
        )
        esm2_dict.update(bench_emb)
        logger.info("Added %d benchmark ESM-2 embeddings", len(bench_emb))
    esm2_dim = next(iter(esm2_dict.values())).shape[0]
    logger.info("ESM-2: %d proteins (dim=%d)", len(esm2_dict), esm2_dim)

    # Load model
    model = load_model(args.model, esm2_dim, args.model_version, device)

    # Build drug -> binders lookup from training data
    logger.info("Building drug->binders lookup...")
    train_df = pd.read_csv(args.training)
    drug_to_binders = defaultdict(list)
    for uid, smi in zip(train_df["uniprot_id"], train_df["ligand_smiles"]):
        if uid in esm2_dict and uid not in drug_to_binders[smi]:
            drug_to_binders[smi].append(uid)
    training_proteins = set(train_df["uniprot_id"]) & set(esm2_dict.keys())
    training_protein_ids = sorted(training_proteins)
    training_emb_matrix = None
    if training_protein_ids:
        training_emb_matrix = F.normalize(
            torch.stack([esm2_dict[pid] for pid in training_protein_ids]).to(device),
            dim=1,
        )
    logger.info("Training: %d drugs, %d proteins with ESM-2",
                len(drug_to_binders), len(training_proteins))

    # Load benchmark
    bench_df = normalize_benchmark_frame(pd.read_csv(args.benchmark))
    logger.info("Benchmark: %d pairs, %d proteins",
                len(bench_df), bench_df["uniprot_id"].nunique())

    # Resolve anchors first so inference can run in large GPU batches.
    eval_records = []
    anchor_stats = {"drug_match": 0, "similarity_fallback": 0, "no_anchor": 0}

    for uid, group in bench_df.groupby("uniprot_id"):
        ptype = group["protein_type"].iloc[0] if "protein_type" in group.columns else "unknown"

        if uid not in esm2_dict:
            continue

        smiles_list = group["ligand_smiles"].tolist()
        true_pkis_all = group["pki"].values
        similarity_fallback = None

        valid_indices = []
        for i, smi in enumerate(smiles_list):
            anchors = find_anchor_by_drug(smi, drug_to_binders, esm2_dict)
            if anchors:
                valid_indices.append((i, smi, anchors[0], "drug"))
            else:
                if similarity_fallback is None and training_emb_matrix is not None:
                    query_vec = F.normalize(esm2_dict[uid].unsqueeze(0).to(device), dim=1)
                    sims = torch.matmul(training_emb_matrix, query_vec.squeeze(0))
                    if uid in training_proteins:
                        sims[training_protein_ids.index(uid)] = -1
                    similarity_fallback = training_protein_ids[int(torch.argmax(sims).item())]
                if similarity_fallback:
                    valid_indices.append((i, smi, similarity_fallback, "similarity"))

        if not valid_indices:
            anchor_stats["no_anchor"] += 1
            continue

        for i, smi, anchor_id, method in valid_indices:
            anchor_stats["drug_match" if method == "drug" else "similarity_fallback"] += 1
            eval_records.append({
                "uniprot_id": uid,
                "protein_type": ptype,
                "ligand_smiles": smi,
                "anchor_id": anchor_id,
                "true_pki": float(true_pkis_all[i]),
            })

    if not eval_records:
        raise RuntimeError("No evaluable benchmark pairs found after anchor selection")

    logger.info(
        "Running batched inference on %d anchored pairs (batch_size=%d)",
        len(eval_records), args.batch_size
    )

    preds = np.empty(len(eval_records), dtype=np.float32)
    probs = np.empty(len(eval_records), dtype=np.float32)

    with torch.inference_mode():
        for start in range(0, len(eval_records), args.batch_size):
            end = min(start + args.batch_size, len(eval_records))
            batch = eval_records[start:end]

            anchor_emb = torch.stack([
                esm2_dict[record["anchor_id"]] for record in batch
            ]).to(device, non_blocking=True)
            query_emb = torch.stack([
                esm2_dict[record["uniprot_id"]] for record in batch
            ]).to(device, non_blocking=True)
            smi_enc = torch.tensor(
                [encode_smiles(record["ligand_smiles"]) for record in batch],
                dtype=torch.long,
                device=device,
            )

            out = model(anchor_emb, query_emb, smi_enc)
            preds[start:end] = out["pki_pred"].detach().float().cpu().numpy().reshape(-1)
            probs[start:end] = out["binding_prob"].detach().float().cpu().numpy().reshape(-1)

            if start == 0 or end == len(eval_records) or ((start // args.batch_size) + 1) % 10 == 0:
                logger.info("Inference progress: %d/%d pairs", end, len(eval_records))

    by_protein = defaultdict(list)
    for record, pred, prob in zip(eval_records, preds, probs):
        by_protein[record["uniprot_id"]].append({
            "true_pki": record["true_pki"],
            "pred": float(pred),
            "prob": float(prob),
            "protein_type": record["protein_type"],
        })

    protein_results = []
    for idx, (uid, rows) in enumerate(by_protein.items(), start=1):
        ptype = rows[0]["protein_type"]
        true_pkis = np.array([row["true_pki"] for row in rows], dtype=np.float32)
        preds = np.array([row["pred"] for row in rows], dtype=np.float32)
        probs = np.array([row["prob"] for row in rows], dtype=np.float32)

        ci = concordance_index(true_pkis, preds)
        rmse = float(np.sqrt(np.mean((true_pkis - preds) ** 2)))
        pearson_r = float(np.corrcoef(true_pkis, preds)[0, 1]) if len(preds) > 1 else 0.0

        true_labels = (true_pkis >= args.binder_threshold).astype(int)
        if len(set(true_labels)) == 2:
            auroc = roc_auc_score(true_labels, probs)
            auprc = average_precision_score(true_labels, probs)
        else:
            auroc = auprc = float("nan")

        protein_results.append({
            "uniprot_id": uid, "protein_type": ptype, "n_drugs": len(rows),
            "ci": ci, "rmse": rmse, "pearson_r": pearson_r,
            "auroc": auroc, "auprc": auprc,
        })

        if idx % 20 == 0:
            logger.info("[%d] %s (%s, %d drugs): CI=%.3f RMSE=%.3f AUROC=%.3f",
                        idx, uid, ptype, len(rows), ci, rmse, auroc)

    # Aggregate
    res_df = pd.DataFrame(protein_results)
    res_df.to_csv(output_dir / "per_protein_results.csv", index=False)

    logger.info("=" * 60)
    logger.info("RESULTS (%d proteins)", len(res_df))
    logger.info("Anchor stats: %s", anchor_stats)

    summary = {}
    for ptype in sorted(res_df["protein_type"].unique()) + ["all"]:
        sub = res_df if ptype == "all" else res_df[res_df["protein_type"] == ptype]
        if len(sub) == 0:
            continue
        metrics = {
            col: float(sub[col].mean())
            for col in ["ci", "rmse", "pearson_r", "auroc", "auprc"]
            if col in sub.columns
        }
        metrics["n"] = len(sub)
        summary[ptype] = metrics
        logger.info("  %s (n=%d): CI=%.3f RMSE=%.3f r=%.3f AUROC=%.3f",
                    ptype.upper(), len(sub),
                    metrics.get("ci", 0), metrics.get("rmse", 0),
                    metrics.get("pearson_r", 0), metrics.get("auroc", 0))

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
