"""Single-run training loop for DrugBAN paper replication."""
from __future__ import annotations

import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from scripts.drugban_paper.dataset import (
    AnchorBinaryDTIDataset,
    BinaryDTIDataset,
    OracleAnchorDTIDataset,
    SubsetBinaryDTIDataset,
    build_graph_cache,
    collate_anchor_binary,
    collate_binary,
    load_split,
)
from scripts.drugban_paper.metrics import compute_metrics

log = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _evaluate(model, loader, model_type: str, device) -> dict:
    """Run evaluation on a dataloader and return metrics."""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            labels = batch["label"]

            if model_type in ("drugban", "drugban_anchor_subset", "drugban_oracle_subset"):
                protein = batch["protein"].to(device)
                logits = model(graph, protein)
            else:
                anchor_prot = batch["anchor_prot"].to(device)
                query_prot = batch["query_prot"].to(device)
                logits = model(graph, anchor_prot, query_prot)

            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(labels.numpy())

    return compute_metrics(np.array(all_labels), np.array(all_logits))


def train_one_run(
    dataset: str,
    split: str,
    model_type: str,
    seed: int,
    data_dir: str = "data/drugban_paper",
    output_dir: str = "models/drugban_paper",
    batch_size: int = 64,
    lr: float = 5e-5,
    epochs: int = 100,
    patience: int = 20,
    hidden_dim: int = 128,
    dropout: float = 0.2,
    graph_cache: dict | None = None,
) -> dict:
    """Train one model on one dataset/split with one seed.

    Args:
        dataset: "bindingdb", "biosnap", or "human".
        split: "random", "cold", or "cluster".
        model_type: "drugban" or "anchor_drugban".
        seed: random seed.
        data_dir: path to downloaded DrugBAN data.
        output_dir: path to save model checkpoints.
        batch_size: training batch size.
        lr: learning rate.
        epochs: max training epochs.
        patience: early stopping patience on val AUROC.
        hidden_dim: model hidden dimension.
        dropout: dropout rate.
        graph_cache: optional pre-built graph cache (shared across runs).

    Returns:
        Dict with test metrics and run metadata.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Cap workers to avoid fd exhaustion in containers with limited ulimits
    num_workers = min(os.cpu_count() or 4, 4)
    pin_memory = torch.cuda.is_available()
    run_name = f"{dataset}_{split}_{model_type}_seed{seed}"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"=== {run_name} on {device} (workers={num_workers}) ===")

    # 1. Load data
    train_df, val_df, test_df = load_split(data_dir, dataset, split)

    # 2. Build graph cache
    all_smiles = sorted(
        set(
            train_df["SMILES"].tolist()
            + val_df["SMILES"].tolist()
            + test_df["SMILES"].tolist()
        )
    )
    gc = build_graph_cache(all_smiles, existing_cache=graph_cache)

    # 3. Create datasets and model
    if model_type in ("drugban", "drugban_anchor_subset", "drugban_oracle_subset"):
        from anchor_transfer.model.drugban import DrugBANModel

        # Check if subset exists — train on same data as anchor model
        import json as _json
        if model_type == "drugban_oracle_subset":
            subset_dir = Path(output_dir) / "oracle_subsets"
        else:
            subset_dir = Path(output_dir) / "anchor_subsets"
        subset_test_path = subset_dir / f"{dataset}_{split}_test_pairs.json"
        use_subset = model_type in ("drugban_anchor_subset", "drugban_oracle_subset") and subset_test_path.exists()

        if use_subset:
            train_pairs = set(tuple(p) for p in _json.load(open(subset_dir / f"{dataset}_{split}_train_pairs.json")))
            val_pairs = set(tuple(p) for p in _json.load(open(subset_dir / f"{dataset}_{split}_val_pairs.json")))
            test_pairs = set(tuple(p) for p in _json.load(open(subset_test_path)))
            train_ds = SubsetBinaryDTIDataset(train_df, gc, train_pairs)
            val_ds = SubsetBinaryDTIDataset(val_df, gc, val_pairs)
            test_ds = SubsetBinaryDTIDataset(test_df, gc, test_pairs)
            log.info(f"  Using anchor subset: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
        else:
            train_ds = BinaryDTIDataset(train_df, gc)
            val_ds = BinaryDTIDataset(val_df, gc)
            test_ds = BinaryDTIDataset(test_df, gc)

        collate_fn = collate_binary
        model = DrugBANModel(
            hidden_dim=hidden_dim, dropout=dropout, binary=True
        ).to(device)

    elif model_type == "anchor_drugban":
        from anchor_transfer.model.anchor_drugban import AnchorDrugBAN
        from scripts.drugban_paper.anchor import AnchorIndex

        anchor_idx = AnchorIndex(
            train_smiles=train_df["SMILES"].tolist(),
            train_proteins=train_df["Protein"].tolist(),
            train_labels=train_df["Y"].tolist(),
        )
        cache_dir = str(Path(output_dir) / "anchor_caches")
        train_ds = AnchorBinaryDTIDataset(
            train_df, gc, anchor_idx,
            cache_dir=cache_dir, cache_key=f"{dataset}_{split}_train",
        )
        val_ds = AnchorBinaryDTIDataset(
            val_df, gc, anchor_idx,
            cache_dir=cache_dir, cache_key=f"{dataset}_{split}_val",
        )
        test_ds = AnchorBinaryDTIDataset(
            test_df, gc, anchor_idx,
            cache_dir=cache_dir, cache_key=f"{dataset}_{split}_test",
        )
        collate_fn = collate_anchor_binary
        model = AnchorDrugBAN(
            hidden_dim=hidden_dim, dropout=dropout, binary=True
        ).to(device)

        # Save anchor-available pairs for DrugBAN same-subset training + evaluation
        import json as _json
        subset_dir = Path(output_dir) / "anchor_subsets"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for name, ds_obj in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            p = subset_dir / f"{dataset}_{split}_{name}_pairs.json"
            _json.dump([list(x) for x in ds_obj.kept_pairs], open(p, "w"))
        log.info(
            f"  Saved anchor subsets: "
            f"train={len(train_ds.kept_pairs)}, "
            f"val={len(val_ds.kept_pairs)}, "
            f"test={len(test_ds.kept_pairs)}"
        )

    elif model_type == "anchor_drugban_oracle":
        from anchor_transfer.model.anchor_drugban import AnchorDrugBAN

        # Oracle: use ALL positives (train+val+test) for anchor lookup
        all_pos = pd.concat([train_df, val_df, test_df], ignore_index=True)
        all_pos = all_pos[all_pos["Y"] == 1][["SMILES", "Protein"]]

        train_ds = OracleAnchorDTIDataset(train_df, gc, all_pos)
        val_ds = OracleAnchorDTIDataset(val_df, gc, all_pos)
        test_ds = OracleAnchorDTIDataset(test_df, gc, all_pos)
        collate_fn = collate_anchor_binary
        model = AnchorDrugBAN(
            hidden_dim=hidden_dim, dropout=dropout, binary=True
        ).to(device)

        # Save oracle subsets for fair comparison
        import json as _json
        subset_dir = Path(output_dir) / "oracle_subsets"
        subset_dir.mkdir(parents=True, exist_ok=True)
        for name, ds_obj in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
            p = subset_dir / f"{dataset}_{split}_{name}_pairs.json"
            _json.dump([list(x) for x in ds_obj.kept_pairs], open(p, "w"))
        log.info(
            f"  Saved oracle subsets: "
            f"train={len(train_ds.kept_pairs)}, "
            f"val={len(val_ds.kept_pairs)}, "
            f"test={len(test_ds.kept_pairs)}"
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"Model: {model_type}, params: {n_params:,}")
    log.info(
        f"Datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # 4. Data loaders — max CPU/GPU utilization
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    # 5. Optimizer (matching paper: Adam, lr=5e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 6. Training loop
    best_val_auroc = -1.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, n_samples = 0.0, 0

        for batch in tqdm(train_loader, desc=f"Ep {epoch:3d}", leave=False):
            graph = batch["graph"].to(device)
            labels = batch["label"].to(device)

            if model_type in ("drugban", "drugban_anchor_subset", "drugban_oracle_subset"):
                protein = batch["protein"].to(device)
                logits = model(graph, protein)
            else:
                anchor_prot = batch["anchor_prot"].to(device)
                query_prot = batch["query_prot"].to(device)
                logits = model(graph, anchor_prot, query_prot)

            loss = F.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(labels)
            n_samples += len(labels)

        # 7. Validation
        val_metrics = _evaluate(model, val_loader, model_type, device)
        elapsed = time.time() - t0

        improved = val_metrics["auroc"] > best_val_auroc
        if improved:
            best_val_auroc = val_metrics["auroc"]
            patience_counter = 0
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch},
                run_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        log.info(
            f"Ep {epoch:3d} [{elapsed:.0f}s] "
            f"loss={total_loss / n_samples:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f} "
            f"val_auprc={val_metrics['auprc']:.4f} "
            f"{'*' if improved else f'p={patience_counter}'}"
        )

        if patience_counter >= patience:
            log.info("Early stopping")
            break

    # 8. Test with best model
    ckpt = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = _evaluate(model, test_loader, model_type, device)
    test_metrics["seed"] = seed
    test_metrics["dataset"] = dataset
    test_metrics["split"] = split
    test_metrics["model"] = model_type
    test_metrics["best_epoch"] = ckpt["epoch"]
    test_metrics["test_size"] = len(test_ds)

    log.info(
        f"TEST {run_name}: "
        f"AUROC={test_metrics['auroc']:.4f} "
        f"AUPRC={test_metrics['auprc']:.4f} "
        f"Acc={test_metrics['accuracy']:.4f} "
        f"Sens={test_metrics['sensitivity']:.4f} "
        f"Spec={test_metrics['specificity']:.4f} "
        f"(n={len(test_ds)})"
    )

    return test_metrics
