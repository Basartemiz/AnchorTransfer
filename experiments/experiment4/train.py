"""Training loop for ConciseDTA vs ConciseAnchorBilinear comparison on DTC/BDB."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear
from anchor_transfer.model.concise_dta import ConciseDTA

from experiments.experiment4.eval import evaluate_model

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def _macro_metrics(protein_metrics: dict) -> tuple[float, float, float]:
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
    aurocs = [m["auroc"] for m in protein_metrics.values() if "auroc" in m]
    if not aurocs:
        return float("nan"), 0
    return sum(aurocs) / len(aurocs), len(aurocs)


def _model_predict(model, batch, device) -> torch.Tensor:
    """Dispatch forward call based on model type; return (B,) pKi predictions."""
    drug_fp = batch["drug_fp"].to(device)
    if isinstance(model, ConciseAnchorBilinear):
        anchor_emb = batch["anchor_emb"].to(device)
        query_emb = batch["query_emb"].to(device)
        return model(drug_fp, anchor_emb, query_emb)
    if isinstance(model, ConciseDTA):
        query_emb = batch["query_emb"].to(device)
        return model(drug_fp, query_emb)
    raise TypeError(f"Unsupported model type: {type(model).__name__}")


def train_models(
    data,
    val_loader,
    epochs: int = 10,
    dta_model: ConciseDTA | None = None,
    anchor_model: ConciseAnchorBilinear | None = None,
) -> None:
    adam_weight_decay = 1e-4
    dta_optimizer = torch.optim.AdamW(
        dta_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )
    anchor_optimizer = torch.optim.AdamW(
        anchor_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_dta_score = float("-inf")
    best_anchor_score = float("-inf")
    device = next(dta_model.parameters()).device

    for epoch in range(epochs):
        dta_model.train()
        anchor_model.train()
        total_dta, total_anchor, n_batches = 0.0, 0.0, 0
        for batch in data:
            pki_targets = batch["pki"].to(device)

            dta_preds = _model_predict(dta_model, batch, device)
            dta_loss = F.mse_loss(dta_preds, pki_targets)

            anchor_preds = _model_predict(anchor_model, batch, device)
            anchor_loss = F.mse_loss(anchor_preds, pki_targets)

            dta_optimizer.zero_grad()
            dta_loss.backward()
            dta_optimizer.step()

            anchor_optimizer.zero_grad()
            anchor_loss.backward()
            anchor_optimizer.step()

            total_dta += dta_loss.item()
            total_anchor += anchor_loss.item()
            n_batches += 1

        avg_dta = total_dta / max(n_batches, 1)
        avg_anchor = total_anchor / max(n_batches, 1)

        torch.save(dta_model.state_dict(), CHECKPOINT_DIR / f"concise_dta_epoch_{epoch}.pt")
        torch.save(anchor_model.state_dict(), CHECKPOINT_DIR / f"concise_anchor_epoch_{epoch}.pt")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"ConciseDTA avg loss: {avg_dta:.4f} - "
            f"ConciseAnchorBilinear avg loss: {avg_anchor:.4f}"
        )

        dta_metrics = evaluate_model(dta_model, val_loader, device)
        anchor_metrics = evaluate_model(anchor_model, val_loader, device)

        d_ci, d_rmse, d_score = _macro_metrics(dta_metrics)
        a_ci, a_rmse, a_score = _macro_metrics(anchor_metrics)

        print(f"  Val ConciseDTA:    macro_CI={d_ci:.4f}  macro_RMSE={d_rmse:.4f}  score={d_score:.4f}")
        print(f"  Val ConciseAnchorBilinear: macro_CI={a_ci:.4f}  macro_RMSE={a_rmse:.4f}  score={a_score:.4f}")

        if d_score > best_dta_score:
            best_dta_score = d_score
            torch.save(dta_model.state_dict(), CHECKPOINT_DIR / "concise_dta_best.pt")
            print(f"  [best] ConciseDTA score improved → saved concise_dta_best.pt")

        if a_score > best_anchor_score:
            best_anchor_score = a_score
            torch.save(anchor_model.state_dict(), CHECKPOINT_DIR / "concise_anchor_best.pt")
            print(f"  [best] ConciseAnchorBilinear score improved → saved concise_anchor_best.pt")
