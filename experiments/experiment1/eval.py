"""Per-protein and per-quartile evaluation for ESM-DTA and AnchorTransfer v2."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2


def evaluate_model(
    model,
    data_loader,
    device,
    return_records: bool = False,
):
    """Compute per-protein RMSE / Pearson / CI on the given loader.

    Dispatches forward pass by model type: AnchorTransferDTAv2 uses
    (anchor, query, drug) and returns a dict; EsmDTAModel takes (drug, protein).

    If `return_records=True`, also returns a list of per-sample dicts
    (`protein_id`, `drug_id`, `pred`, `true`, `anchor_pki`) for downstream
    stratified analysis (e.g. anchor-pKi quartile breakdowns).
    """
    model.eval()
    is_anchor = isinstance(model, AnchorTransferDTAv2)

    records: list[dict] = []
    with torch.no_grad():
        for batch in data_loader:
            drug_indices = batch["drug_indices"].to(device)
            protein_esm2 = batch["protein_esm2"].to(device)
            pki_targets = batch["pki"].to(device)

            if is_anchor:
                preds = model(
                    anchor_esm2=batch["anchor_protein_esm2"].to(device),
                    query_esm2=protein_esm2,
                    drug_indices=drug_indices,
                    anchor_pki=batch["anchor_pki"].to(device),
                )["pki_pred"]
            else:
                preds = model(drug_indices, protein_esm2)

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

    # Group by protein for per-protein metrics. Drugs are de-duplicated per protein
    # (last prediction wins) to match the prior behavior.
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
        if auroc == auroc:  # not NaN — both classes present
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
    """AUROC with pKi binarization.

    Binder (positive) = pKi > binder_threshold (default 7).
    Non-binder (negative) = pKi <= nonbinder_threshold (default 5) — inclusive
    to capture Davis-style detection-limit values that sit exactly at pKi=5.
    Samples in (5, 7] are ambiguous and dropped. Returns NaN if only one
    class is present after filtering.
    """
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
    """Bin records by `records[i][by]` and compute RMSE + flat-pool CI per bin.

    If `edges` is None, uses data-driven quartiles (25/50/75 percentile of the
    non-NaN values). Returns a list of dicts, one per bin, each with keys
    {bin, n, rmse, ci, true, pred}.
    """
    import math

    vals = [r[by] for r in records if not math.isnan(r.get(by, float("nan")))]
    if not vals:
        return []

    if edges is None:
        sorted_v = sorted(vals)
        n = len(sorted_v)
        edges = [sorted_v[n // 4], sorted_v[n // 2], sorted_v[(3 * n) // 4]]

    # Build bins: (-inf, e1], (e1, e2], (e2, e3], (e3, +inf)
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
