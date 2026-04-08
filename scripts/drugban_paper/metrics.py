"""Binary classification metrics matching DrugBAN paper Table 1."""
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    """Compute AUROC, AUPRC, Accuracy, Sensitivity, Specificity at best-F1 threshold.

    Args:
        labels: (N,) binary ground truth (0 or 1).
        logits: (N,) raw model output (before sigmoid).

    Returns:
        Dict with keys: auroc, auprc, accuracy, sensitivity, specificity, threshold.
    """
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))

    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    # Find threshold at best F1
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1s)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    preds = (probs >= best_thresh).astype(int)
    accuracy = accuracy_score(labels, preds)

    tp = np.sum((preds == 1) & (labels == 1))
    fn = np.sum((preds == 0) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "accuracy": float(accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "threshold": float(best_thresh),
    }
