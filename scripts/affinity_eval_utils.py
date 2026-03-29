from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PAIR_KEY_COLUMNS = ("uniprot_id", "ligand_smiles")


def jsonable_float(value: float | int | None):
    if value is None:
        return None
    value = float(value)
    if np.isnan(value) or np.isinf(value):
        return None
    return value


def safe_stat(func, y_true, y_pred, min_points: int = 2) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < min_points:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = func(y_true, y_pred)[0]
    return float(value)


def concordance_index(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size < 2:
        return float("nan")

    order = np.argsort(y_true)
    y_true = y_true[order]
    y_pred = y_pred[order]

    ranks = np.unique(y_pred, return_inverse=True)[1] + 1
    tree = np.zeros(int(ranks.max()) + 1, dtype=np.int64)

    def bit_add(index: int, value: int) -> None:
        while index < tree.size:
            tree[index] += value
            index += index & -index

    def bit_sum(index: int) -> int:
        total = 0
        while index > 0:
            total += tree[index]
            index -= index & -index
        return total

    concordant = 0.0
    tied = 0.0
    permissible = 0.0
    inserted = 0
    start = 0
    n = y_true.size
    while start < n:
        end = start + 1
        while end < n and abs(y_true[end] - y_true[start]) < 1e-10:
            end += 1

        group_ranks = ranks[start:end]
        for rank in group_ranks:
            less = bit_sum(int(rank) - 1)
            at_most = bit_sum(int(rank))
            equal = at_most - less
            permissible += inserted
            concordant += less
            tied += equal

        for rank in group_ranks:
            bit_add(int(rank), 1)
            inserted += 1
        start = end

    if permissible == 0:
        return float("nan")
    return (concordant + 0.5 * tied) / permissible


def normalize_pair_columns(df: pd.DataFrame, key_cols: tuple[str, str] = PAIR_KEY_COLUMNS) -> pd.DataFrame:
    normalized = df.copy()
    protein_col, smiles_col = key_cols
    normalized[protein_col] = normalized[protein_col].astype(str).str.strip()
    normalized[smiles_col] = normalized[smiles_col].astype(str).str.split(" |", regex=False).str[0].str.strip()
    return normalized


def pair_key_set(df: pd.DataFrame, key_cols: tuple[str, str] = PAIR_KEY_COLUMNS) -> set[tuple[str, str]]:
    normalized = normalize_pair_columns(df, key_cols=key_cols)
    return set(normalized.loc[:, list(key_cols)].itertuples(index=False, name=None))


def intersect_pair_keys(frames: Iterable[pd.DataFrame], key_cols: tuple[str, str] = PAIR_KEY_COLUMNS) -> set[tuple[str, str]]:
    frame_list = list(frames)
    if not frame_list:
        return set()
    common = pair_key_set(frame_list[0], key_cols=key_cols)
    for frame in frame_list[1:]:
        common &= pair_key_set(frame, key_cols=key_cols)
    return common


def filter_to_pair_keys(
    df: pd.DataFrame,
    keys: set[tuple[str, str]],
    key_cols: tuple[str, str] = PAIR_KEY_COLUMNS,
) -> pd.DataFrame:
    normalized = normalize_pair_columns(df, key_cols=key_cols)
    if not keys:
        return normalized.iloc[0:0].copy()
    pair_index = pd.MultiIndex.from_frame(normalized.loc[:, list(key_cols)])
    wanted_index = pd.MultiIndex.from_tuples(sorted(keys), names=list(key_cols))
    return normalized.loc[pair_index.isin(wanted_index)].copy()


def threshold_binary_pairs(
    df: pd.DataFrame,
    *,
    positive_pki_threshold: float = 6.0,
    negative_pki_threshold: float = 5.0,
    true_col: str = "true_pki",
    pred_col: str = "pred_pki",
) -> pd.DataFrame:
    """Return a thresholded binary classification frame with ambiguous pairs dropped.

    Pairs are labeled as:
    - binder: true pX >= positive_pki_threshold
    - non-binder: true pX <= negative_pki_threshold
    - ambiguous: negative_pki_threshold < true pX < positive_pki_threshold
    """

    if positive_pki_threshold <= negative_pki_threshold:
        raise ValueError("positive_pki_threshold must be greater than negative_pki_threshold")

    labeled = df.copy()
    labeled[true_col] = pd.to_numeric(labeled[true_col], errors="coerce")
    labeled[pred_col] = pd.to_numeric(labeled[pred_col], errors="coerce")
    labeled = labeled.dropna(subset=[true_col, pred_col]).copy()

    binder_mask = labeled[true_col] >= positive_pki_threshold
    non_binder_mask = labeled[true_col] <= negative_pki_threshold
    labeled["binary_label"] = np.where(binder_mask, 1, np.where(non_binder_mask, 0, np.nan))
    labeled["is_ambiguous"] = labeled["binary_label"].isna()

    thresholded = labeled.loc[~labeled["is_ambiguous"]].copy()
    thresholded["binary_label"] = thresholded["binary_label"].astype(int)
    return thresholded


def compute_thresholded_classification_metrics(
    df: pd.DataFrame,
    *,
    positive_pki_threshold: float = 6.0,
    negative_pki_threshold: float = 5.0,
    true_col: str = "true_pki",
    pred_col: str = "pred_pki",
) -> dict:
    """Compute AUROC/AUPRC after dropping ambiguous pairs between two pKi cutoffs."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    thresholded = threshold_binary_pairs(
        df,
        positive_pki_threshold=positive_pki_threshold,
        negative_pki_threshold=negative_pki_threshold,
        true_col=true_col,
        pred_col=pred_col,
    )
    n_pairs_total = int(len(df))
    n_pairs_evaluated = int(len(thresholded))
    n_binders = int((thresholded["binary_label"] == 1).sum())
    n_non_binders = int((thresholded["binary_label"] == 0).sum())
    n_ambiguous = int(n_pairs_total - n_pairs_evaluated)

    if n_binders == 0 or n_non_binders == 0:
        return {
            "n_pairs_total": n_pairs_total,
            "n_pairs_evaluated": n_pairs_evaluated,
            "n_binders": n_binders,
            "n_non_binders": n_non_binders,
            "n_ambiguous": n_ambiguous,
            "auroc": None,
            "auprc": None,
            "positive_pki_threshold": float(positive_pki_threshold),
            "negative_pki_threshold": float(negative_pki_threshold),
        }

    y_true = thresholded["binary_label"].to_numpy(dtype=int)
    y_score = thresholded[pred_col].to_numpy(dtype=float)
    return {
        "n_pairs_total": n_pairs_total,
        "n_pairs_evaluated": n_pairs_evaluated,
        "n_binders": n_binders,
        "n_non_binders": n_non_binders,
        "n_ambiguous": n_ambiguous,
        "auroc": jsonable_float(float(roc_auc_score(y_true, y_score))),
        "auprc": jsonable_float(float(average_precision_score(y_true, y_score))),
        "positive_pki_threshold": float(positive_pki_threshold),
        "negative_pki_threshold": float(negative_pki_threshold),
    }


def compute_per_protein_thresholded_classification(
    df: pd.DataFrame,
    *,
    positive_pki_threshold: float = 6.0,
    negative_pki_threshold: float = 5.0,
    true_col: str = "true_pki",
    pred_col: str = "pred_pki",
    group_col: str = "uniprot_id",
) -> dict:
    """Compute mean AUROC/AUPRC across proteins after dropping ambiguous pairs."""

    from sklearn.metrics import average_precision_score, roc_auc_score

    n_proteins_total = int(df[group_col].nunique()) if len(df) else 0
    n_proteins_with_pairs = 0
    aurocs = []
    auprcs = []

    for _, group_df in df.groupby(group_col, sort=True):
        thresholded = threshold_binary_pairs(
            group_df,
            positive_pki_threshold=positive_pki_threshold,
            negative_pki_threshold=negative_pki_threshold,
            true_col=true_col,
            pred_col=pred_col,
        )
        if len(thresholded) == 0:
            continue
        n_proteins_with_pairs += 1
        y_true = thresholded["binary_label"].to_numpy(dtype=int)
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            continue
        y_score = thresholded[pred_col].to_numpy(dtype=float)
        aurocs.append(float(roc_auc_score(y_true, y_score)))
        auprcs.append(float(average_precision_score(y_true, y_score)))

    return {
        "n_proteins_total": n_proteins_total,
        "n_proteins_with_thresholded_pairs": int(n_proteins_with_pairs),
        "n_proteins_evaluated": int(len(aurocs)),
        "mean_auroc": jsonable_float(float(np.mean(aurocs))) if aurocs else None,
        "mean_auprc": jsonable_float(float(np.mean(auprcs))) if aurocs else None,
        "positive_pki_threshold": float(positive_pki_threshold),
        "negative_pki_threshold": float(negative_pki_threshold),
    }


def compute_regression_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {
            "n_pairs": 0,
            "n_proteins": 0,
            "n_proteins_mse": 0,
            "n_proteins_ci": 0,
            "n_proteins_pearson": 0,
            "n_proteins_spearman": 0,
            "mse": None,
            "rmse": None,
            "pearson": None,
            "spearman": None,
            "ci": None,
            "pooled": {
                "mse": None,
                "rmse": None,
                "pearson": None,
                "spearman": None,
            },
        }

    y_true = df["true_pki"].to_numpy(dtype=float)
    y_pred = df["pred_pki"].to_numpy(dtype=float)
    pooled_mse = float(np.mean((y_true - y_pred) ** 2))
    pooled_rmse = float(np.sqrt(pooled_mse))

    mses = []
    cis = []
    pearsons = []
    spearmans = []
    for _, grp in df.groupby("uniprot_id", sort=True):
        yt = grp["true_pki"].to_numpy(dtype=float)
        yp = grp["pred_pki"].to_numpy(dtype=float)
        mses.append(float(np.mean((yt - yp) ** 2)))
        ci = concordance_index(yt, yp)
        if not np.isnan(ci):
            cis.append(ci)
        pearson = safe_stat(pearsonr, yt, yp, min_points=3)
        if not np.isnan(pearson):
            pearsons.append(pearson)
        spearman = safe_stat(spearmanr, yt, yp, min_points=3)
        if not np.isnan(spearman):
            spearmans.append(spearman)

    macro_mse = float(np.mean(mses)) if mses else float("nan")

    return {
        "n_pairs": int(len(df)),
        "n_proteins": int(df["uniprot_id"].nunique()),
        "n_proteins_mse": int(len(mses)),
        "n_proteins_ci": int(len(cis)),
        "n_proteins_pearson": int(len(pearsons)),
        "n_proteins_spearman": int(len(spearmans)),
        "mse": jsonable_float(macro_mse),
        "rmse": jsonable_float(np.sqrt(macro_mse) if not np.isnan(macro_mse) else float("nan")),
        "pearson": jsonable_float(float(np.mean(pearsons)) if pearsons else float("nan")),
        "spearman": jsonable_float(float(np.mean(spearmans)) if spearmans else float("nan")),
        "ci": jsonable_float(float(np.mean(cis)) if cis else float("nan")),
        "pooled": {
            "mse": jsonable_float(pooled_mse),
            "rmse": jsonable_float(pooled_rmse),
            "pearson": jsonable_float(safe_stat(pearsonr, y_true, y_pred)),
            "spearman": jsonable_float(safe_stat(spearmanr, y_true, y_pred)),
        },
    }
