"""Training loop for DrugBAN vs AnchorDrugBAN comparison on DTC/BDB."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from anchor_transfer.model.anchor_drugban import AnchorDrugBAN
from anchor_transfer.model.drugban import DrugBAN

from experiments.experiment2.eval import evaluate_model

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
    graph = batch["drug_graph"].to(device)
    if isinstance(model, AnchorDrugBAN):
        v_a = batch["anchor_tokens"].to(device)
        v_q = batch["query_tokens"].to(device)
        # train-mode tuple: (v_d, v_a, v_q, f_anchor, f_query, score).
        score = model(graph, v_a, v_q, mode="train")[-1]
    elif isinstance(model, DrugBAN):
        v_p = batch["query_tokens"].to(device)
        score = model(graph, v_p, mode="train")[-1]
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    return score.squeeze(-1)


def train_models(
    data,
    val_loader,
    epochs: int = 10,
    drugban_model: DrugBAN | None = None,
    anchor_model: AnchorDrugBAN | None = None,
) -> None:
    adam_weight_decay = 1e-4
    drugban_optimizer = torch.optim.AdamW(
        drugban_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )
    anchor_optimizer = torch.optim.AdamW(
        anchor_model.parameters(), lr=1e-4, weight_decay=adam_weight_decay
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_drugban_score = float("-inf")
    best_anchor_score = float("-inf")
    device = next(drugban_model.parameters()).device

    for epoch in range(epochs):
        drugban_model.train()
        anchor_model.train()
        total_drugban, total_anchor, n_batches = 0.0, 0.0, 0
        for batch in data:
            pki_targets = batch["pki"].to(device)

            drugban_preds = _model_predict(drugban_model, batch, device)
            drugban_loss = F.mse_loss(drugban_preds, pki_targets)

            anchor_preds = _model_predict(anchor_model, batch, device)
            anchor_loss = F.mse_loss(anchor_preds, pki_targets)

            drugban_optimizer.zero_grad()
            drugban_loss.backward()
            drugban_optimizer.step()

            anchor_optimizer.zero_grad()
            anchor_loss.backward()
            anchor_optimizer.step()

            total_drugban += drugban_loss.item()
            total_anchor += anchor_loss.item()
            n_batches += 1

        avg_drugban = total_drugban / max(n_batches, 1)
        avg_anchor = total_anchor / max(n_batches, 1)

        torch.save(drugban_model.state_dict(), CHECKPOINT_DIR / f"drugban_epoch_{epoch}.pt")
        torch.save(anchor_model.state_dict(), CHECKPOINT_DIR / f"anchor_drugban_epoch_{epoch}.pt")

        print(
            f"Epoch {epoch+1}/{epochs} - "
            f"DrugBAN avg loss: {avg_drugban:.4f} - "
            f"AnchorDrugBAN avg loss: {avg_anchor:.4f}"
        )

        drugban_metrics = evaluate_model(drugban_model, val_loader, device)
        anchor_metrics = evaluate_model(anchor_model, val_loader, device)

        d_ci, d_rmse, d_score = _macro_metrics(drugban_metrics)
        a_ci, a_rmse, a_score = _macro_metrics(anchor_metrics)

        print(f"  Val DrugBAN:       macro_CI={d_ci:.4f}  macro_RMSE={d_rmse:.4f}  score={d_score:.4f}")
        print(f"  Val AnchorDrugBAN: macro_CI={a_ci:.4f}  macro_RMSE={a_rmse:.4f}  score={a_score:.4f}")

        if d_score > best_drugban_score:
            best_drugban_score = d_score
            torch.save(drugban_model.state_dict(), CHECKPOINT_DIR / "drugban_best.pt")
            print(f"  [best] DrugBAN score improved → saved drugban_best.pt")

        if a_score > best_anchor_score:
            best_anchor_score = a_score
            torch.save(anchor_model.state_dict(), CHECKPOINT_DIR / "anchor_drugban_best.pt")
            print(f"  [best] AnchorDrugBAN score improved → saved anchor_drugban_best.pt")
