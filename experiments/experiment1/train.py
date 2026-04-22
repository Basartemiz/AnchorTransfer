"""Training loop for ESM-DTA vs AnchorTransfer v2 comparison."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.experiment1.eval import evaluate_model

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _macro_metrics(protein_metrics: dict) -> tuple[float, float, float]:
    """Return (macro_CI, macro_RMSE, combined_score) across proteins.

    combined_score = macro_CI - macro_RMSE (higher is better).
    Proteins with <2 samples contribute to RMSE but not CI.
    """
    cis = [m["ci"] for m in protein_metrics.values() if "ci" in m]
    rmses = [m["rmse"] for m in protein_metrics.values()]
    if not rmses:
        return float("nan"), float("nan"), float("-inf")
    macro_rmse = sum(rmses) / len(rmses)
    if not cis:
        return float("nan"), macro_rmse, float("-inf")
    macro_ci = sum(cis) / len(cis)
    return macro_ci, macro_rmse, macro_ci - macro_rmse


def _macro_auroc(protein_metrics: dict) -> tuple[float, int]:
    """Return (macro_AUROC, n_proteins_with_auroc). Proteins without both
    binder/non-binder samples are skipped."""
    aurocs = [m["auroc"] for m in protein_metrics.values() if "auroc" in m]
    if not aurocs:
        return float("nan"), 0
    return sum(aurocs) / len(aurocs), len(aurocs)


def train_models(
    data,
    val_loader,
    epochs: int = 10,
    esm_dta_model=None,
    anchor_dta_model=None,
) -> None:
    adam_weight_decay = 1e-4
    esm_dta_optimizer = torch.optim.AdamW(
        esm_dta_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )
    anchor_dta_optimizer = torch.optim.AdamW(
        anchor_dta_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_esm_score = float("-inf")
    best_anchor_score = float("-inf")
    device = next(esm_dta_model.parameters()).device

    for epoch in range(epochs):
        total_esm, total_anchor, n_batches = 0.0, 0.0, 0
        for batch in data:
            drug_indices = batch["drug_indices"].to(device)
            protein_esm2 = batch["protein_esm2"].to(device)
            pki_targets = batch["pki"].to(device)
            anchor_esm2 = batch["anchor_protein_esm2"].to(device)
            anchor_pki = batch["anchor_pki"].to(device)

            # ESM-DTA
            esm_dta_preds = esm_dta_model(drug_indices, protein_esm2)
            esm_dta_loss = F.mse_loss(esm_dta_preds, pki_targets)

            # AnchorTransfer v2
            anchor_dta_preds = anchor_dta_model(
                anchor_esm2=anchor_esm2,
                query_esm2=protein_esm2,
                drug_indices=drug_indices,
                anchor_pki=anchor_pki,
            )["pki_pred"]
            anchor_dta_loss = F.mse_loss(anchor_dta_preds, pki_targets)

            esm_dta_optimizer.zero_grad()
            esm_dta_loss.backward()
            esm_dta_optimizer.step()

            anchor_dta_optimizer.zero_grad()
            anchor_dta_loss.backward()
            anchor_dta_optimizer.step()

            total_esm += esm_dta_loss.item()
            total_anchor += anchor_dta_loss.item()
            n_batches += 1

        avg_loss_esm_dta = total_esm / max(n_batches, 1)
        avg_loss_anchor_dta = total_anchor / max(n_batches, 1)

        torch.save(esm_dta_model.state_dict(), CHECKPOINT_DIR / f"esm_dta_epoch_{epoch}.pt")
        torch.save(anchor_dta_model.state_dict(), CHECKPOINT_DIR / f"anchor_dta_epoch_{epoch}.pt")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"ESM-DTA avg loss: {avg_loss_esm_dta:.4f} - "
            f"AnchorTransfer v2 avg loss: {avg_loss_anchor_dta:.4f}"
        )

        device = next(esm_dta_model.parameters()).device
        esm_metrics = evaluate_model(esm_dta_model, val_loader, device)
        anchor_metrics = evaluate_model(anchor_dta_model, val_loader, device)

        esm_ci, esm_rmse, esm_score = _macro_metrics(esm_metrics)
        anchor_ci, anchor_rmse, anchor_score = _macro_metrics(anchor_metrics)

        print(
            f"  Val ESM-DTA:           macro_CI={esm_ci:.4f}  macro_RMSE={esm_rmse:.4f}  score={esm_score:.4f}"
        )
        print(
            f"  Val AnchorTransfer v2: macro_CI={anchor_ci:.4f}  macro_RMSE={anchor_rmse:.4f}  score={anchor_score:.4f}"
        )

        if esm_score > best_esm_score:
            best_esm_score = esm_score
            torch.save(esm_dta_model.state_dict(), CHECKPOINT_DIR / "esm_dta_best.pt")
            print(f"  [best] ESM-DTA score improved → saved esm_dta_best.pt")

        if anchor_score > best_anchor_score:
            best_anchor_score = anchor_score
            torch.save(anchor_dta_model.state_dict(), CHECKPOINT_DIR / "anchor_dta_best.pt")
            print(f"  [best] AnchorTransfer v2 score improved → saved anchor_dta_best.pt")
