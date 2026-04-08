"""Compare prot_knn vs ConciseAnchor on the SAME subset of interactions.

Loads both result CSVs, intersects to common interactions,
computes per-protein CI/AUROC/RMSE + Q1-Q4 quartile plots.
"""
import sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_auc_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

PROJECT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT / "results"
N_WORKERS = cpu_count()


# ── per-protein metrics ──────────────────────────────────────────

def _ci_one(yt, yp):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 3 or yp.std() < 1e-8:
        return 0.5 if len(yt) >= 3 else np.nan
    dt = yt[:, None] - yt[None, :]
    dp = yp[:, None] - yp[None, :]
    idx = np.triu_indices(len(yt), k=1)
    dt_f, dp_f = dt[idx], dp[idx]
    nz = dt_f != 0
    if nz.sum() == 0:
        return 0.5
    dt_nz, dp_nz = dt_f[nz], dp_f[nz]
    return float(((dt_nz * dp_nz > 0).sum() + 0.5 * (dp_nz == 0).sum()) / len(dt_nz))


def _auroc_one(yt, yp, hi=7.0, lo=5.0):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    mask = (yt >= hi) | (yt <= lo)
    if mask.sum() < 2:
        return np.nan
    labels = (yt[mask] >= hi).astype(int)
    if len(set(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, yp[mask])


def _rmse_one(yt, yp):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    return np.sqrt(mean_squared_error(yt, yp)) if len(yt) >= 3 else np.nan


def _metrics_worker(args):
    uid, yt, yp = args
    return uid, _ci_one(yt, yp), _auroc_one(yt, yp), _rmse_one(yt, yp)


def pp_metrics(df, col):
    tasks = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        tasks.append((uid, s.pki.values, np.array(s[col].values, dtype=float)))
    with Pool(N_WORKERS) as pool:
        results = pool.map(_metrics_worker, tasks)
    cis = [r[1] for r in results if not np.isnan(r[1])]
    aucs = [r[2] for r in results if not np.isnan(r[2])]
    rmses = [r[3] for r in results if not np.isnan(r[3])]
    return np.array(cis), np.array(aucs), np.array(rmses)


def pp_metrics_by_protein(df, col):
    """Return per-protein dict of {uid: (ci, auroc, rmse)}."""
    tasks = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        tasks.append((uid, s.pki.values, np.array(s[col].values, dtype=float)))
    with Pool(N_WORKERS) as pool:
        results = pool.map(_metrics_worker, tasks)
    return {r[0]: (r[1], r[2], r[3]) for r in results}


# ── main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load results
    ca_df = pd.read_csv(RESULTS_DIR / "concise_anchor_dtc_test.csv")
    knn_df = pd.read_csv(RESULTS_DIR / "prot_knn_dtc_test.csv")

    log.info(f"ConciseAnchor: {len(ca_df)} interactions, {ca_df.uniprot_id.nunique()} proteins")
    log.info(f"prot_knn full: {len(knn_df)} interactions, {knn_df.uniprot_id.nunique()} proteins")

    # Merge on (uniprot_id, ligand_smiles) to get common subset
    # ConciseAnchor df has: uniprot_id, ligand_smiles, pki, anchor_uid, concise_anchor
    # kNN df has: uniprot_id, ligand_smiles, pki, prot_knn_k1, prot_knn_k5
    merged = ca_df.merge(
        knn_df[["uniprot_id", "ligand_smiles", "prot_knn_k1", "prot_knn_k5"]],
        on=["uniprot_id", "ligand_smiles"],
        how="inner"
    )
    log.info(f"Common subset: {len(merged)} interactions, {merged.uniprot_id.nunique()} proteins")

    # Compute metrics on common subset
    methods = {
        "prot_knn_k1": "prot_knn_k1",
        "prot_knn_k5": "prot_knn_k5",
        "ConciseAnchor": "concise_anchor",
    }

    log.info(f"\n{'='*70}")
    log.info(f"COMPARISON — Same {len(merged)} interactions, {merged.uniprot_id.nunique()} proteins")
    log.info(f"{'='*70}")
    log.info(f"  {'Method':<20s} {'CI':>7s} {'AUROC':>7s} {'RMSE':>7s}")
    log.info(f"  {'-'*50}")

    summary_rows = []
    for name, col in methods.items():
        ci, auc, rmse = pp_metrics(merged, col)
        log.info(f"  {name:<20s} {ci.mean():.4f}  {auc.mean() if len(auc) else 0:.4f}  "
                 f"{rmse.mean() if len(rmse) else 0:.4f}")
        summary_rows.append({"method": name, "ci": ci.mean(),
                            "auroc": auc.mean() if len(auc) else np.nan,
                            "rmse": rmse.mean() if len(rmse) else np.nan,
                            "n_interactions": len(merged),
                            "n_proteins": merged.uniprot_id.nunique()})

    # ── Q1-Q4 by pKi quartile ──
    q25, q50, q75 = np.percentile(merged.pki.values, [25, 50, 75])
    merged["pki_quartile"] = pd.cut(merged.pki, bins=[-np.inf, q25, q50, q75, np.inf],
                                     labels=["Q1", "Q2", "Q3", "Q4"])
    log.info(f"\n  pKi quartiles: Q1<={q25:.2f}, Q2<={q50:.2f}, Q3<={q75:.2f}, Q4>{q75:.2f}")

    quartile_data = {}  # {method: {quartile: (ci, auroc, rmse)}}
    for name, col in methods.items():
        quartile_data[name] = {}
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = merged[merged.pki_quartile == q]
            if len(sub) >= 10:
                ci_q, auc_q, rmse_q = pp_metrics(sub, col)
                quartile_data[name][q] = (ci_q.mean(), auc_q.mean() if len(auc_q) else np.nan,
                                          rmse_q.mean() if len(rmse_q) else np.nan)
                log.info(f"    {name:<20s} {q}: CI={ci_q.mean():.4f}, "
                         f"AUROC={auc_q.mean() if len(auc_q) else 0:.4f}, "
                         f"RMSE={rmse_q.mean() if len(rmse_q) else 0:.4f}, n={len(sub)}")

    # ── Plots ──
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})
    quartiles = ["Q1", "Q2", "Q3", "Q4"]
    colors = {"prot_knn_k1": "#7fb3d3", "prot_knn_k5": "#3a7ebf", "ConciseAnchor": "#d45b5b"}
    labels = {"prot_knn_k1": "Prot-kNN (k=1)", "prot_knn_k5": "Prot-kNN (k=5)",
              "ConciseAnchor": "ConciseAnchor"}

    # ── Figure 1: CI by quartile (grouped bar) ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, metric_idx, ylabel in zip(
        axes, ["CI", "AUROC", "RMSE"], [0, 1, 2], ["Per-protein CI", "Per-protein AUROC", "Per-protein RMSE"]
    ):
        x = np.arange(len(quartiles))
        width = 0.25
        for i, (name, _) in enumerate(methods.items()):
            vals = [quartile_data[name].get(q, (np.nan, np.nan, np.nan))[metric_idx] for q in quartiles]
            bars = ax.bar(x + i * width, vals, width, label=labels[name], color=colors[name],
                         edgecolor="white", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f"{v:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xlabel("pKi Quartile")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{metric} by pKi Quartile")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{q}\n(≤{t:.1f})" if q != "Q4" else f"Q4\n(>{q75:.1f})"
                           for q, t in zip(quartiles, [q25, q50, q75, np.inf])])
        if metric == "CI":
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
            ax.set_ylim(0.45, max(0.75, max(v for v in [quartile_data[n].get(q, (0,))[0]
                        for n in methods for q in quartiles] if not np.isnan(v)) + 0.05))
        elif metric == "AUROC":
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random (0.5)")
            ax.set_ylim(0.4, 1.0)
        ax.legend(fontsize=9)

    plt.suptitle(f"prot_knn vs ConciseAnchor — DTC Cold-Protein Test\n"
                 f"Same {len(merged)} interactions, {merged.uniprot_id.nunique()} proteins",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(RESULTS_DIR / f"knn_vs_concise_quartiles.{ext}", bbox_inches="tight")
    log.info(f"Saved knn_vs_concise_quartiles.png/pdf")
    plt.close()

    # ── Figure 2: Overall comparison bar chart ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    method_names = list(methods.keys())

    for ax, metric, metric_idx, ylabel in zip(
        axes, ["CI", "AUROC", "RMSE"], [0, 1, 2],
        ["Per-protein CI", "Per-protein AUROC", "Per-protein RMSE"]
    ):
        vals = [summary_rows[i][metric.lower()] for i in range(len(method_names))]
        bar_colors = [colors[n] for n in method_names]
        bars = ax.bar([labels[n] for n in method_names], vals, color=bar_colors,
                     edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                   f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_title(metric)
        if metric in ("CI", "AUROC"):
            ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
            ax.set_ylim(0.4, max(vals) + 0.1)
        else:
            ax.set_ylim(0, max(vals) + 0.3)

    plt.suptitle(f"Overall: prot_knn vs ConciseAnchor — Same Subset\n"
                 f"{len(merged)} interactions, {merged.uniprot_id.nunique()} proteins",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(RESULTS_DIR / f"knn_vs_concise_overall.{ext}", bbox_inches="tight")
    log.info(f"Saved knn_vs_concise_overall.png/pdf")
    plt.close()

    # ── Figure 3: Per-protein scatter (CI) ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, knn_col, knn_label in zip(axes, ["prot_knn_k1", "prot_knn_k5"],
                                       ["Prot-kNN (k=1)", "Prot-kNN (k=5)"]):
        ca_pp = pp_metrics_by_protein(merged, "concise_anchor")
        knn_pp = pp_metrics_by_protein(merged, knn_col)
        common_prots = set(ca_pp.keys()) & set(knn_pp.keys())

        ca_cis = [ca_pp[p][0] for p in common_prots if not np.isnan(ca_pp[p][0]) and not np.isnan(knn_pp[p][0])]
        knn_cis = [knn_pp[p][0] for p in common_prots if not np.isnan(ca_pp[p][0]) and not np.isnan(knn_pp[p][0])]

        ax.scatter(knn_cis, ca_cis, alpha=0.5, s=30, color="#3a7ebf", edgecolors="white", linewidth=0.3)
        ax.plot([0.3, 1], [0.3, 1], "k--", alpha=0.4)
        ax.set_xlabel(f"{knn_label} CI")
        ax.set_ylabel("ConciseAnchor CI")
        ax.set_title(f"Per-protein CI: ConciseAnchor vs {knn_label}")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(0.3, 1.0)
        ax.set_aspect("equal")

        # Count wins
        wins_ca = sum(1 for c, k in zip(ca_cis, knn_cis) if c > k)
        wins_knn = sum(1 for c, k in zip(ca_cis, knn_cis) if k > c)
        ax.text(0.05, 0.95, f"ConciseAnchor wins: {wins_ca}\n{knn_label} wins: {wins_knn}",
               transform=ax.transAxes, fontsize=9, va="top",
               bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.suptitle(f"Per-protein CI Comparison — {len(common_prots)} proteins",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(RESULTS_DIR / f"knn_vs_concise_scatter.{ext}", bbox_inches="tight")
    log.info(f"Saved knn_vs_concise_scatter.png/pdf")
    plt.close()

    # Save merged CSV and summary
    merged.to_csv(RESULTS_DIR / "knn_vs_concise_common_subset.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(RESULTS_DIR / "knn_vs_concise_summary.csv", index=False)

    log.info(f"\n{'='*70}")
    log.info("DONE — all plots and CSVs saved")
