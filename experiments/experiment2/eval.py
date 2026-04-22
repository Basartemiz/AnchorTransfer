"""Per-protein / per-quartile evaluation for DrugBAN and AnchorDrugBAN."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from anchor_transfer.model.anchor_drugban import AnchorDrugBAN
from anchor_transfer.model.drugban import DrugBAN


def _model_predict_eval(model, batch, device) -> torch.Tensor:
    graph = batch["drug_graph"].to(device)
    if isinstance(model, AnchorDrugBAN):
        v_a = batch["anchor_tokens"].to(device)
        v_q = batch["query_tokens"].to(device)
        # eval-mode tuple: (v_d, v_a, v_q, score, att_a, att_q).
        score = model(graph, v_a, v_q, mode="eval")[3]
    elif isinstance(model, DrugBAN):
        v_p = batch["query_tokens"].to(device)
        # eval-mode tuple: (v_d, v_p, score, att).
        score = model(graph, v_p, mode="eval")[2]
    else:
        raise TypeError(f"Unsupported model type: {type(model).__name__}")
    return score.squeeze(-1)


def evaluate_model(model, data_loader, device, return_records: bool = False):
    """Per-protein RMSE / CI / AUROC metrics."""
    model.eval()

    records: list[dict] = []
    with torch.no_grad():
        for batch in data_loader:
            pki_targets = batch["pki"].to(device)
            preds = _model_predict_eval(model, batch, device)

            anchor_pki_batch = batch.get("anchor_pki")
            for i in range(len(batch["protein_id"])):
                records.append({
                    "protein_id": batch["protein_id"][i],
                    "drug_id": batch["drug_id"][i],
                    "pred": preds[i].item(),
                    "true": pki_targets[i].item(),
                    "anchor_pki": (
                        float(anchor_pki_batch[i].item()) if anchor_pki_batch is not None else float("nan")
                    ),
                })

    by_protein: dict[str, dict[str, list[float]]] = {}
    for rec in records:
        by_protein.setdefault(rec["protein_id"], {})[rec["drug_id"]] = [rec["pred"], rec["true"]]

    protein_metrics: dict[str, dict] = {}
    for protein_id, drug_preds in by_protein.items():
        y_true = torch.tensor([t for (_, t) in drug_preds.values()], dtype=torch.float)
        y_pred = torch.tensor([p for (p, _) in drug_preds.values()], dtype=torch.float)

        rmse = torch.sqrt(F.mse_loss(y_pred, y_true))
        protein_metrics[protein_id] = {"rmse": rmse.item(), "n_samples": len(y_true)}

        if len(y_true) >= 2:
            pearson = torch.corrcoef(torch.stack([y_true, y_pred]))[0, 1]
            if not torch.isnan(pearson):
                protein_metrics[protein_id]["pearson_corr"] = pearson.item()
            protein_metrics[protein_id]["ci"] = _concordance_index(y_true, y_pred)

        auroc = _get_auroc(y_true, y_pred)
        if auroc == auroc:
            protein_metrics[protein_id]["auroc"] = auroc

    if return_records:
        return protein_metrics, records
    return protein_metrics


def _concordance_index(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Vectorized CI: pairwise diffs over the upper triangle, no Python loop."""
    import numpy as np

    yt = y_true.detach().cpu().numpy() if isinstance(y_true, torch.Tensor) else np.asarray(y_true)
    yp = y_pred.detach().cpu().numpy() if isinstance(y_pred, torch.Tensor) else np.asarray(y_pred)
    n = yt.shape[0]
    if n < 2:
        return float("nan")
    dt = yt[:, None] - yt[None, :]
    dp = yp[:, None] - yp[None, :]
    prod = (dt * dp)[np.triu_indices(n, k=1)]
    concordant = int((prod > 0).sum())
    discordant = int((prod < 0).sum())
    tied = int((prod == 0).sum())
    total = concordant + discordant + tied
    return (concordant + 0.5 * tied) / total if total else float("nan")


def _get_auroc(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    binder_threshold: float = 7.0,
    nonbinder_threshold: float = 5.0,
) -> float:
    """Binder = pKi > 7; non-binder = pKi <= 5; ambiguous (5,7] dropped."""
    from sklearn.metrics import roc_auc_score

    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()
    binder = y_true_np > binder_threshold
    nonbinder = y_true_np <= nonbinder_threshold
    mask = binder | nonbinder
    if not mask.any():
        return float("nan")
    labels = binder[mask].astype(int)
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, y_pred_np[mask]))


def quartile_metrics(
    records: list[dict],
    by: str = "anchor_pki",
    edges: list[float] | None = None,
) -> list[dict]:
    import math

    vals = [r[by] for r in records if not math.isnan(r.get(by, float("nan")))]
    if not vals:
        return []

    if edges is None:
        sorted_v = sorted(vals)
        n = len(sorted_v)
        edges = [sorted_v[n // 4], sorted_v[n // 2], sorted_v[(3 * n) // 4]]

    labels = [f"≤{edges[0]:.2f}"]
    for k in range(len(edges) - 1):
        labels.append(f"({edges[k]:.2f}, {edges[k+1]:.2f}]")
    labels.append(f">{edges[-1]:.2f}")

    bins: list[list[dict]] = [[] for _ in range(len(edges) + 1)]
    for rec in records:
        v = rec.get(by, float("nan"))
        if math.isnan(v):
            continue
        placed = False
        for k, e in enumerate(edges):
            if v <= e:
                bins[k].append(rec); placed = True; break
        if not placed:
            bins[-1].append(rec)

    out: list[dict] = []
    for label, members in zip(labels, bins):
        n = len(members)
        if n == 0:
            out.append({"bin": label, "n": 0, "rmse": float("nan"), "ci": float("nan")})
            continue
        true_t = torch.tensor([r["true"] for r in members], dtype=torch.float)
        pred_t = torch.tensor([r["pred"] for r in members], dtype=torch.float)
        rmse = torch.sqrt(F.mse_loss(pred_t, true_t)).item()
        ci = _concordance_index(true_t, pred_t) if n >= 2 else float("nan")
        out.append({"bin": label, "n": n, "rmse": rmse, "ci": ci})
    return out
