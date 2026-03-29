#!/usr/bin/env python3
"""Train AffinityGAT with pure InfoNCE loss for binder/non-binder discrimination.

Uses domain training data (domain_training_data.csv) with ProteinBatchSampler.
Positives: pKi >= 7, Hard negatives: pKi <= 5, In-batch negatives from other proteins.
No MSE head, no curriculum — pure contrastive.

Usage:
    python scripts/train_contrastive_binder.py \
        --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
        --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
        --training-data data/processed/domain_training_data.csv \
        --device cuda --epochs 200
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from idr_gat.model.affinity_gat import AffinityGAT, infonce_loss, encode_smiles
from idr_gat.data.protein_batch_sampler import ProteinBatchSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_graph(graph_path, node_ranges_path, device):
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    node_ranges = torch.load(node_ranges_path, map_location=device, weights_only=False)
    return graph, node_ranges


def build_training_items(df, node_ranges):
    """Convert domain training CSV to list of dicts for ProteinBatchSampler."""
    items = []
    smiles_set = set()
    skipped = 0

    for _, row in df.iterrows():
        uid = row["uniprot_id"]
        if uid not in node_ranges:
            skipped += 1
            continue
        start_node, end_node = node_ranges[uid]
        anchor_node = start_node  # use first node as anchor
        items.append({
            "uniprot_id": uid,
            "smiles": row["ligand_smiles"],
            "pki": float(row["pki"]),
            "anchors": [{"anchor_node": int(anchor_node), "anchor_tm": 1.0}],
        })
        smiles_set.add(row["ligand_smiles"])

    return items, sorted(smiles_set), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-ranges", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--proteins-per-batch", type=int, default=32)
    parser.add_argument("--pos-per-protein", type=int, default=16)
    parser.add_argument("--hard-neg-per-protein", type=int, default=16)
    parser.add_argument("--pos-threshold", type=float, default=7.0)
    parser.add_argument("--neg-threshold", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models/contrastive_binder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    logger.info("Loading graph...")
    graph, node_ranges_raw = load_graph(args.graph, args.node_ranges, device)
    # Convert node_ranges to dict
    node_ranges = {}
    if isinstance(node_ranges_raw, dict):
        node_ranges = node_ranges_raw
    else:
        node_ranges = node_ranges_raw

    # Load domain training data
    df = pd.read_csv(args.training_data)
    logger.info("Loaded %d domain-drug pairs", len(df))

    # Split 90/10 by protein
    uids = df["uniprot_id"].unique()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uids)
    n_train = int(0.9 * len(uids))
    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:])

    train_df = df[df["uniprot_id"].isin(train_uids)]
    val_df = df[df["uniprot_id"].isin(val_uids)]

    train_items, train_smiles, skip_train = build_training_items(train_df, node_ranges)
    val_items, val_smiles, skip_val = build_training_items(val_df, node_ranges)
    logger.info("Train: %d items (%d skipped), Val: %d items (%d skipped)",
                len(train_items), skip_train, len(val_items), skip_val)

    # Build SMILES lookup
    all_smiles = sorted(set(train_smiles + val_smiles))
    smiles_lookup = {s: i for i, s in enumerate(all_smiles)}
    smiles_tensor = torch.stack([
        torch.tensor(encode_smiles(s), dtype=torch.long) for s in all_smiles
    ]).to(device)
    logger.info("SMILES vocabulary: %d", len(all_smiles))

    # Model
    model = AffinityGAT(proj_dim=args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AffinityGAT (contrastive) parameters: %s", f"{n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Pre-compute node embeddings once per epoch
        with torch.no_grad():
            graph_dev = graph.to(device) if graph.x_3di.device != device else graph
            node_embs = model.encode_graph(graph_dev)

        sampler = ProteinBatchSampler(
            train_items, proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            hard_neg_per_protein=args.hard_neg_per_protein,
            mid_per_protein=0,  # no middle zone for pure contrastive
            pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
        )

        total_nce, n_batches = 0.0, 0
        for batch in sampler:
            anchor_list, smi_list, lbl_list = [], [], []
            for item in batch:
                smi = item["smiles"]
                if smi not in smiles_lookup:
                    continue
                anc = item["anchors"][0]
                anchor_list.append(anc["anchor_node"])
                smi_list.append(smiles_lookup[smi])
                lbl_list.append(item["label"])

            if not anchor_list:
                continue

            b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
            b_smi_enc = smiles_tensor[torch.tensor(smi_list, dtype=torch.long, device=device)]
            b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

            prot_emb = node_embs[b_anchors]
            drug_emb = model.drug_encoder(b_smi_enc)
            interaction = model.cross_attn(prot_emb, drug_emb)

            proj = F.normalize(model.proj_head(interaction), dim=1)
            pos_mask = b_labels == 1
            neg_mask = b_labels == 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                nce = infonce_loss(proj[pos_mask], proj[pos_mask], proj[neg_mask],
                                   temperature=args.temperature)
            else:
                continue

            optimizer.zero_grad()
            nce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nce += nce.item()
            n_batches += 1

        train_loss = total_nce / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            node_embs = model.encode_graph(graph_dev)

        val_sampler = ProteinBatchSampler(
            val_items, proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            hard_neg_per_protein=args.hard_neg_per_protein,
            mid_per_protein=0,
            pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
        )

        val_nce, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_sampler:
                anchor_list, smi_list, lbl_list = [], [], []
                for item in batch:
                    smi = item["smiles"]
                    if smi not in smiles_lookup:
                        continue
                    anc = item["anchors"][0]
                    anchor_list.append(anc["anchor_node"])
                    smi_list.append(smiles_lookup[smi])
                    lbl_list.append(item["label"])

                if not anchor_list:
                    continue

                b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
                b_smi_enc = smiles_tensor[torch.tensor(smi_list, dtype=torch.long, device=device)]
                b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

                prot_emb = node_embs[b_anchors]
                drug_emb = model.drug_encoder(b_smi_enc)
                interaction = model.cross_attn(prot_emb, drug_emb)
                proj = F.normalize(model.proj_head(interaction), dim=1)

                pos_mask = b_labels == 1
                neg_mask = b_labels == 0
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    nce = infonce_loss(proj[pos_mask], proj[pos_mask], proj[neg_mask],
                                       temperature=args.temperature)
                    val_nce += nce.item()
                    val_batches += 1

        val_loss = val_nce / max(val_batches, 1)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "proj_dim": args.proj_dim,
                "temperature": args.temperature,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            logger.info("Epoch %d/%d train_nce=%.4f val_nce=%.4f best=%.4f patience=%d/%d%s",
                        epoch, args.epochs, train_loss, val_loss, best_val_loss,
                        patience_counter, args.patience, " *" if improved else "")

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Done. Best val_nce=%.4f at %s", best_val_loss, output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
