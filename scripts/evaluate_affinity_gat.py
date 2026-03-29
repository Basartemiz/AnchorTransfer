#!/usr/bin/env python3
"""Evaluate AffinityGAT with anchor-based conformation aggregation.

Usage:
  PYTHONPATH=src:. python scripts/evaluate_affinity_gat.py \
    --model models/affinity_gat_alphafold/best_model.pt \
    --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
    --benchmark data/raw/benchmark_affinity.csv \
    --output-dir results/affinity_gat_alphafold
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from idr_gat.model.affinity_gat import AffinityGAT, encode_smiles
from scripts.evaluate_anchor_dta import (
    select_diverse_conformations,
    extract_trajectory_conformations,
    find_anchors_for_protein,
    load_idrome_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _ci(y_true, y_pred):
    n = len(y_true)
    if n < 2:
        return float("nan")
    concordant = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] != y_true[j]:
                total += 1
                if (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0:
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    return concordant / total if total > 0 else float("nan")


@torch.no_grad()
def predict_gat(
    model: AffinityGAT,
    graph,
    anchor_idx: int,
    smiles_list: list[str],
    device: str,
    batch_size: int = 512,
) -> np.ndarray:
    """Predict pKi for one anchor node against multiple drugs."""
    model.eval()
    all_preds = []
    for start in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[start:start + batch_size]
        anchors = torch.full((len(batch_smiles),), anchor_idx, dtype=torch.long, device=device)
        smiles_enc = torch.tensor([encode_smiles(s) for s in batch_smiles], dtype=torch.long, device=device)
        preds = model(graph, anchors, smiles_enc)
        all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-ranges", required=True)
    parser.add_argument("--benchmark", default="data/raw/benchmark_affinity.csv")
    parser.add_argument("--domain-metadata", required=True)
    parser.add_argument("--domain-dir", required=True)
    parser.add_argument("--target-db", default=None)
    parser.add_argument("--idrome-index", default="data/processed/idrome_conformation_index.json")
    parser.add_argument("--n-conformations", type=int, default=10)
    parser.add_argument("--anchor-tm-threshold", type=float, default=0.4)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-proteins", type=int, default=0)
    parser.add_argument("--output-dir", default="results/affinity_gat_alphafold")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    logger.info("Loading graph...")
    graph = torch.load(args.graph, map_location=device, weights_only=False).to(device)
    node_ranges = torch.load(args.node_ranges, map_location="cpu", weights_only=False)

    logger.info("Loading model from %s...", args.model)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)
    model_args = checkpoint.get("args", {})
    esm2_dim = graph.x_esm2.shape[1] if hasattr(graph, "x_esm2") else 480
    model = AffinityGAT(
        esm2_input_dim=esm2_dim,
        hidden_dim=model_args.get("hidden_dim", 512),
        gat_layers=model_args.get("gat_layers", 4),
        dropout=model_args.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    logger.info("Running full graph GATv2 forward pass...")
    with torch.no_grad():
        node_embs = model.encode_graph(graph)
    logger.info("Node embeddings: %s", node_embs.shape)

    with open(args.domain_metadata) as f:
        domain_metadata = json.load(f)
    idrome_index = load_idrome_index(Path(args.idrome_index))
    benchmark_df = pd.read_csv(args.benchmark)

    protein_groups = list(benchmark_df.groupby("uniprot_id"))
    if args.max_proteins > 0:
        protein_groups = protein_groups[:args.max_proteins]

    all_rows = []
    protein_summaries = []

    for idx, (uniprot_id, group) in enumerate(protein_groups):
        protein_type = group["protein_type"].iloc[0]
        smiles_list = group["ligand_smiles"].tolist()
        true_pki = group["pki"].values

        logger.info("[%d/%d] %s (%s) — %d drugs",
                    idx + 1, len(protein_groups), uniprot_id, protein_type, len(smiles_list))

        pred = np.full(len(smiles_list), np.nan)
        n_anchors = 0
        mean_tm = 0.0

        if uniprot_id in node_ranges:
            start, _ = node_ranges[uniprot_id]
            pred = predict_gat(model, graph, start, smiles_list, device)
            n_anchors = 1
            mean_tm = 1.0
        else:
            conf_dirs = idrome_index.get(uniprot_id, [])
            if conf_dirs:
                all_frames = extract_trajectory_conformations(
                    conf_dirs, n_frames=100, scratch_dir=Path(tempfile.mkdtemp()),
                )
                selected = select_diverse_conformations(all_frames, k=args.n_conformations)

                anchors = find_anchors_for_protein(
                    selected, Path(args.domain_dir), domain_metadata,
                    target_db_path=args.target_db,
                    foldseek_bin=args.foldseek_bin,
                    tm_threshold=args.anchor_tm_threshold,
                    threads=args.threads,
                )

                if anchors:
                    weighted_sum = np.zeros(len(smiles_list), dtype=np.float64)
                    tm_sum = 0.0
                    for anc in anchors:
                        anc_uid = anc["anchor_uniprot"]
                        if anc_uid in node_ranges:
                            start, _ = node_ranges[anc_uid]
                            p = predict_gat(model, graph, start, smiles_list, device)
                            weighted_sum += anc["anchor_tm"] * p
                            tm_sum += anc["anchor_tm"]
                    if tm_sum > 0:
                        pred = (weighted_sum / tm_sum).astype(np.float32)
                        n_anchors = len(anchors)
                        mean_tm = np.mean([a["anchor_tm"] for a in anchors])

        valid = ~np.isnan(pred) & ~np.isnan(true_pki)
        ci = _ci(true_pki[valid], pred[valid]) if valid.sum() >= 2 else float("nan")
        mse = float(np.mean((true_pki[valid] - pred[valid]) ** 2)) if valid.sum() > 0 else float("nan")

        logger.info("  >>> %s (%s): CI=%.3f  MSE=%.3f  anchors=%d  TM=%.3f",
                    uniprot_id, protein_type, ci, mse, n_anchors, mean_tm)

        for i in range(len(smiles_list)):
            all_rows.append({
                "uniprot_id": uniprot_id,
                "ligand_smiles": smiles_list[i],
                "true_pki": true_pki[i],
                "pred_pki": pred[i],
                "protein_type": protein_type,
            })

        protein_summaries.append({
            "uniprot_id": uniprot_id,
            "protein_type": protein_type,
            "n_drugs": len(smiles_list),
            "n_anchors": n_anchors,
            "mean_anchor_tm": mean_tm,
            "ci": ci,
            "mse": mse,
        })

        # Running averages every 5 proteins
        if len(protein_summaries) % 5 == 0:
            ps_df = pd.DataFrame(protein_summaries)
            for ptype in ["idp", "ordered"]:
                sub = ps_df[ps_df["protein_type"] == ptype].dropna(subset=["ci"])
                if len(sub) > 0:
                    logger.info("  === RUNNING [%s] (n=%d): CI=%.3f  MSE=%.3f ===",
                                ptype.upper(), len(sub), sub["ci"].mean(), sub["mse"].mean())

    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    summary_df = pd.DataFrame(protein_summaries)
    summary_df.to_csv(output_dir / "protein_summary.csv", index=False)

    logger.info("=" * 60)
    for ptype in ["idp", "ordered", "all"]:
        sub = summary_df if ptype == "all" else summary_df[summary_df["protein_type"] == ptype]
        valid = sub.dropna(subset=["ci"])
        if len(valid) > 0:
            logger.info("%s (n=%d): mean_CI=%.3f  mean_MSE=%.3f",
                        ptype.upper(), len(valid), valid["ci"].mean(), valid["mse"].mean())
    logger.info("=" * 60)
    logger.info("DONE — saved to %s", output_dir)


if __name__ == "__main__":
    main()
