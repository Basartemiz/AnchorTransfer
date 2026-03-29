#!/usr/bin/env python3
"""Train domain-native DeepDTA on domain-drug pairs.

Usage:
    python scripts/train_domain_deepdta.py \
        --training-data data/processed/domain_training_data.csv \
        --epochs 100 --batch-size 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from domain_deepdta_model import DomainDeepDTA, encode_domain, DOMAIN_MAX_LEN
from deepdta_encoding import encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DomainDTADataset(Dataset):
    def __init__(self, smiles_encoded, domain_encoded, pkis):
        self.smiles = smiles_encoded
        self.domains = domain_encoded
        self.pkis = pkis

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, idx):
        return (torch.tensor(self.smiles[idx], dtype=torch.long),
                torch.tensor(self.domains[idx], dtype=torch.long),
                torch.tensor(self.pkis[idx], dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True,
                        help="CSV with domain_sequence, ligand_smiles, pki")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models/domain_deepdta")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load domain training data
    df = pd.read_csv(args.training_data)
    logger.info("Loaded %d domain-drug pairs (%d unique domains)",
                len(df), df["domain_name"].nunique())

    # Encode
    smiles_encoded = [encode_smiles(smi) for smi in df["ligand_smiles"]]
    domain_encoded = [encode_domain(seq) for seq in df["domain_sequence"]]
    pkis = df["pki"].values.astype(np.float32)

    # 90/10 split by protein (avoid leakage)
    uids = df["uniprot_id"].unique()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uids)
    n_train_uids = int(0.9 * len(uids))
    train_uids = set(uids[:n_train_uids])

    train_mask = df["uniprot_id"].isin(train_uids).values
    val_mask = ~train_mask

    train_ds = DomainDTADataset(
        [smiles_encoded[i] for i in range(len(df)) if train_mask[i]],
        [domain_encoded[i] for i in range(len(df)) if train_mask[i]],
        pkis[train_mask],
    )
    val_ds = DomainDTADataset(
        [smiles_encoded[i] for i in range(len(df)) if val_mask[i]],
        [domain_encoded[i] for i in range(len(df)) if val_mask[i]],
        pkis[val_mask],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train: %d (%d proteins), Val: %d (%d proteins)",
                len(train_ds), n_train_uids, len(val_ds), len(uids) - n_train_uids)

    model = DomainDeepDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("DomainDeepDTA parameters: %s", f"{n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for s, d, y in train_loader:
            s, d, y = s.to(device), d.to(device), y.to(device)
            pred = model(s, d)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for s, d, y in val_loader:
                s, d, y = s.to(device), d.to(device), y.to(device)
                val_total += F.mse_loss(model(s, d), y).item() * len(y)
        val_loss = val_total / len(val_ds)

        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "domain_max_len": DOMAIN_MAX_LEN,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            logger.info("Epoch %d/%d train=%.4f val=%.4f best=%.4f patience=%d/%d%s",
                        epoch, args.epochs, train_loss, val_loss, best_val_loss,
                        patience_counter, args.patience, " *" if improved else "")

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Done. Best val=%.4f. Model at %s", best_val_loss, output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
