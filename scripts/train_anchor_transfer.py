#!/usr/bin/env python3
"""Train the Anchor Transfer DTA model.

Given (anchor_protein, query_protein, drug) → binary binding + pKi regression.

For each (query, drug, pKi) training sample, a random known binder of that
drug is selected as the anchor. The anchor is re-sampled every epoch for
data augmentation.

Usage:
  python scripts/train_anchor_transfer.py \
    --graph data/processed/whole_protein_graph/global_graph.pt \
    --interactions data/processed/dtc_training_interactions.csv \
    --output-dir models/anchor_transfer \
    --device cuda --epochs 100
"""

import argparse
import json
import logging
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from idr_gat.model.anchor_transfer import AnchorTransferDTA, encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class AnchorTransferDataset(Dataset):
    """Dataset for anchor transfer training.

    Anchor = strongest known binder per drug (highest pKi).  If the strongest
    binder is the query itself, the second-strongest is used.  Samples with no
    valid anchor are dropped entirely.
    """

    def __init__(
        self,
        interactions_df: pd.DataFrame,
        esm2_embeddings: dict[str, torch.Tensor],
        drug_to_anchor: dict[str, str],
        drug_to_second: dict[str, str],
        pos_threshold: float = 7.0,
        neg_threshold: float = 5.0,
    ):
        self.esm2 = esm2_embeddings
        valid_proteins = set(esm2_embeddings.keys())
        df = interactions_df[interactions_df["uniprot_id"].isin(valid_proteins)].copy()
        logger.info("Filtered to %d interactions with ESM-2 embeddings (from %d)",
                     len(df), len(interactions_df))

        queries = df["uniprot_id"].values
        smiles_arr = df["ligand_smiles"].values
        pkis = df["pki"].values
        self.sample_queries = []
        self.sample_smiles = []
        self.sample_pkis = []
        self.sample_labels = []
        self.sample_anchors = []
        n_skipped_no_anchor = 0
        n_skipped_self = 0
        for i in range(len(df)):
            smi = smiles_arr[i]
            if smi not in drug_to_anchor:
                n_skipped_no_anchor += 1
                continue
            anchor = drug_to_anchor[smi]
            query = queries[i]
            if anchor == query:
                anchor = drug_to_second.get(smi)
                if not anchor or anchor not in valid_proteins:
                    n_skipped_self += 1
                    continue
            if anchor not in valid_proteins:
                n_skipped_no_anchor += 1
                continue
            pki = float(pkis[i])
            if pki >= pos_threshold:
                label = 1
            elif pki <= neg_threshold:
                label = 0
            else:
                label = -1
            self.sample_queries.append(query)
            self.sample_smiles.append(smi)
            self.sample_pkis.append(pki)
            self.sample_labels.append(label)
            self.sample_anchors.append(anchor)

        n_samples = len(self.sample_queries)
        logger.info("Dataset: %d samples, %d unique drugs, %d unique proteins",
                     n_samples, len(set(self.sample_smiles)),
                     len(set(self.sample_queries)))
        logger.info("Skipped: %d no-anchor, %d self-anchor-only",
                     n_skipped_no_anchor, n_skipped_self)

        n_pos = sum(1 for l in self.sample_labels if l == 1)
        n_neg = sum(1 for l in self.sample_labels if l == 0)
        n_mid = sum(1 for l in self.sample_labels if l == -1)
        logger.info("Labels: %d pos (pKi>=%.1f), %d neg (pKi<=%.1f), %d middle",
                     n_pos, pos_threshold, n_neg, neg_threshold, n_mid)

        unique_smiles = list(set(self.sample_smiles))
        self._smi_to_idx = {s: i for i, s in enumerate(unique_smiles)}
        self._encoded_smiles = torch.stack([
            torch.tensor(encode_smiles(s), dtype=torch.long) for s in unique_smiles
        ])
        self._sample_smi_indices = [self._smi_to_idx[s] for s in self.sample_smiles]

    def __len__(self):
        return len(self.sample_queries)

    def __getitem__(self, idx):
        return {
            "anchor_esm2": self.esm2[self.sample_anchors[idx]],
            "query_esm2": self.esm2[self.sample_queries[idx]],
            "drug_indices": self._encoded_smiles[self._sample_smi_indices[idx]],
            "pki": self.sample_pkis[idx],
            "label": self.sample_labels[idx],
        }


def collate_fn(batch):
    """Collate anchor transfer samples into batched tensors."""
    anchor_esm2 = torch.stack([b["anchor_esm2"] for b in batch])
    query_esm2 = torch.stack([b["query_esm2"] for b in batch])
    drug_indices = torch.stack([b["drug_indices"] for b in batch])
    pki = torch.tensor([b["pki"] for b in batch], dtype=torch.float)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    binding_mask = labels >= 0
    binding_labels = labels.clamp(min=0)

    return {
        "anchor_esm2": anchor_esm2,
        "query_esm2": query_esm2,
        "drug_indices": drug_indices,
        "pki": pki,
        "binding_labels": binding_labels,
        "binding_mask": binding_mask,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, device, alpha=1.0):
    model.train()
    total_loss = 0
    total_bce = 0
    total_mse = 0
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        anchor = batch["anchor_esm2"].to(device)
        query = batch["query_esm2"].to(device)
        drugs = batch["drug_indices"].to(device)
        pki = batch["pki"].to(device)
        b_labels = batch["binding_labels"].to(device)
        b_mask = batch["binding_mask"].to(device)

        out = model.compute_loss(anchor, query, drugs, pki, b_labels, b_mask, alpha=alpha)

        optimizer.zero_grad()
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += out["loss"].item()
        total_bce += out["bce_loss"].item()
        total_mse += out["mse_loss"].item()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
        "mse": total_mse / max(n_batches, 1),
    }


@torch.no_grad()
def validate(model, loader, device, alpha=1.0):
    model.eval()
    total_loss = 0
    total_bce = 0
    total_mse = 0
    all_preds = []
    all_targets = []
    all_probs = []
    all_labels = []
    all_masks = []
    n_batches = 0

    for batch in loader:
        if batch is None:
            continue
        anchor = batch["anchor_esm2"].to(device)
        query = batch["query_esm2"].to(device)
        drugs = batch["drug_indices"].to(device)
        pki = batch["pki"].to(device)
        b_labels = batch["binding_labels"].to(device)
        b_mask = batch["binding_mask"].to(device)

        out = model.compute_loss(anchor, query, drugs, pki, b_labels, b_mask, alpha=alpha)

        total_loss += out["loss"].item()
        total_bce += out["bce_loss"].item()
        total_mse += out["mse_loss"].item()
        n_batches += 1

        all_preds.append(out["pki_pred"].cpu())
        all_targets.append(pki.cpu())
        all_probs.append(out["binding_prob"].cpu())
        all_labels.append(b_labels.cpu())
        all_masks.append(b_mask.cpu())

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    probs = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    masks = torch.cat(all_masks)

    # Concordance Index
    ci = concordance_index(targets.numpy(), preds.numpy())

    # Pearson r
    if len(preds) > 1:
        pearson_r = float(np.corrcoef(preds.numpy(), targets.numpy())[0, 1])
    else:
        pearson_r = 0.0

    # Binary accuracy (on labeled samples only)
    if masks.any():
        binary_preds = (probs[masks] >= 0.5).long()
        acc = (binary_preds == labels[masks]).float().mean().item()
    else:
        acc = 0.0

    return {
        "loss": total_loss / max(n_batches, 1),
        "bce": total_bce / max(n_batches, 1),
        "mse": total_mse / max(n_batches, 1),
        "ci": ci,
        "pearson_r": pearson_r,
        "binary_acc": acc,
    }


def concordance_index(y_true, y_pred):
    """Compute concordance index."""
    n = len(y_true)
    if n < 2:
        return 0.5
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            if (y_true[i] > y_true[j] and y_pred[i] > y_pred[j]) or \
               (y_true[i] < y_true[j] and y_pred[i] < y_pred[j]):
                concordant += 1
            else:
                discordant += 1
    total = concordant + discordant
    return concordant / total if total > 0 else 0.5


def concordance_index(y_true, y_pred):
    """Fast vectorized concordance index using random sampling for large N."""
    n = len(y_true)
    if n < 2:
        return 0.5

    # For large datasets, sample pairs
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_esm2_from_graph(graph_path: str) -> dict[str, torch.Tensor]:
    """Load ESM-2 embeddings from a whole-protein graph or a plain dict.

    Accepts either:
      - PyG Data with .protein_ids and .x_esm2
      - Plain dict of {protein_id: tensor}
    """
    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    # Plain dict (e.g., from standalone ESM-2 computation)
    if isinstance(data, dict):
        logger.info("Loaded ESM-2 embeddings dict: %d proteins (dim=%d)",
                     len(data), next(iter(data.values())).shape[0] if data else 0)
        return data

    # PyG graph
    protein_ids = data.protein_ids
    x_esm2 = data.x_esm2

    esm2_dict = {}
    for i, pid in enumerate(protein_ids):
        esm2_dict[pid] = x_esm2[i]

    logger.info("Loaded ESM-2 embeddings for %d proteins (dim=%d)",
                len(esm2_dict), x_esm2.shape[1])
    return esm2_dict


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




# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Anchor Transfer DTA")
    parser.add_argument("--graph", required=True, help="Path to global_graph.pt")
    parser.add_argument("--interactions", required=True, help="CSV: uniprot_id,ligand_smiles,pki")
    parser.add_argument("--output-dir", default="models/anchor_transfer")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=1.0, help="MSE weight relative to BCE")
    parser.add_argument("--pos-threshold", type=float, default=7.0)
    parser.add_argument("--neg-threshold", type=float, default=5.0)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--test-split", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm2-dim", type=int, default=480)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading ESM-2 embeddings from graph...")
    esm2_dict = sanitize_esm2_embeddings(load_esm2_from_graph(args.graph))

    logger.info("Loading interactions...")
    df = pd.read_csv(args.interactions)
    df = df[~df["uniprot_id"].str.contains(",", na=False)]
    logger.info("Loaded %d interactions", len(df))

    # 80/10/10 protein-level split (matches paper protocol)
    all_proteins = sorted(set(df["uniprot_id"]) & set(esm2_dict.keys()))
    random.shuffle(all_proteins)
    n_test = max(1, int(len(all_proteins) * args.test_split))
    n_val = max(1, int(len(all_proteins) * args.val_split))
    test_proteins = set(all_proteins[:n_test])
    val_proteins = set(all_proteins[n_test:n_test + n_val])
    train_proteins = set(all_proteins[n_test + n_val:])
    logger.info("Split: %d train, %d val, %d test proteins",
                len(train_proteins), len(val_proteins), len(test_proteins))

    train_df = df[df["uniprot_id"].isin(train_proteins)]
    val_df = df[df["uniprot_id"].isin(val_proteins)]

    # Build drug → strongest binder anchor from TRAINING proteins only
    train_with_emb = train_df[train_df["uniprot_id"].isin(esm2_dict)].copy()
    idx = train_with_emb.groupby("ligand_smiles")["pki"].idxmax()
    drug_to_anchor = dict(zip(
        train_with_emb.loc[idx, "ligand_smiles"],
        train_with_emb.loc[idx, "uniprot_id"],
    ))
    drug_to_second = {}
    for smi, grp in train_with_emb.groupby("ligand_smiles"):
        v = grp.sort_values("pki", ascending=False)
        if len(v) > 1:
            drug_to_second[smi] = v.iloc[1]["uniprot_id"]
    logger.info("Anchor pool: %d drugs with strongest binder, %d with second",
                len(drug_to_anchor), len(drug_to_second))

    train_dataset = AnchorTransferDataset(
        train_df, esm2_dict, drug_to_anchor, drug_to_second,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
    )
    val_dataset = AnchorTransferDataset(
        val_df, esm2_dict, drug_to_anchor, drug_to_second,
        pos_threshold=args.pos_threshold,
        neg_threshold=args.neg_threshold,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0, pin_memory=True,
    )

    # Model
    model = AnchorTransferDTA(esm2_dim=args.esm2_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_epoch(model, train_loader, optimizer, device, alpha=args.alpha)
        val_metrics = validate(model, val_loader, device, alpha=args.alpha)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %3d/%d [%.0fs] lr=%.2e | "
            "Train loss=%.4f (bce=%.4f mse=%.4f) | "
            "Val loss=%.4f (bce=%.4f mse=%.4f) CI=%.4f r=%.4f acc=%.4f",
            epoch, args.epochs, elapsed, lr,
            train_metrics["loss"], train_metrics["bce"], train_metrics["mse"],
            val_metrics["loss"], val_metrics["bce"], val_metrics["mse"],
            val_metrics["ci"], val_metrics["pearson_r"], val_metrics["binary_acc"],
        )

        history.append({
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
        })

        # Checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "args": vars(args),
            }, output_dir / "best_model.pt")
            logger.info("  → Saved best model (val_loss=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (patience=%d)", epoch, args.patience)
                break

    # Save history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("Training complete. Best val loss: %.4f", best_val_loss)
    logger.info("Model saved to %s", output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
