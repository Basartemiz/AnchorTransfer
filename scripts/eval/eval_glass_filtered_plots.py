"""GLASS2 evaluation: 30% homolog filtered + per-protein plots.

Filters GLASS2 proteins to <30% k-mer identity to MooDengDB training proteins.
Produces per-protein CI scatter plot (CoNCISE vs ConciseAnchor).
"""
import os, sys, json, logging, hashlib, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import combinations
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load data
# ================================================================
log.info("Loading data...")
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
glass_seqs = json.load(open("data/raw/glass/glass2_sequences.json"))

# MooDengDB training sequences for homolog filtering
train_raw = pd.read_csv("data/moodeng-v1/train.csv", low_memory=False)
train_seqs = {}
for seq in train_raw["Target Sequence"].unique():
    pid = seq_to_id(seq)
    train_seqs[pid] = seq
log.info(f"GLASS2: {glass.uniprot_id.nunique()} prot, MooDengDB train: {len(train_seqs)} prot")

# ================================================================
# 30% k-mer identity filtering
# ================================================================
log.info("Computing 30% k-mer identity (GLASS2 vs MooDengDB train)...")

# Precompute train k-mer sets
train_kmer_sets = {}
for pid, seq in train_seqs.items():
    s = seq.upper()
    train_kmer_sets[pid] = set(s[i:i+3] for i in range(len(s)-2))

glass_kmer_sets = {}
for uid, seq in glass_seqs.items():
    s = seq.upper()
    glass_kmer_sets[uid] = set(s[i:i+3] for i in range(len(s)-2))

def check_homolog(g_uid):
    if g_uid not in glass_kmer_sets: return None
    gk = glass_kmer_sets[g_uid]
    if not gk: return None
    for tk in train_kmer_sets.values():
        if not tk: continue
        jacc = len(gk & tk) / len(gk | tk)
        if jacc >= 0.30: return g_uid
    return None

glass_prots = list(glass.uniprot_id.unique())
log.info(f"  Checking {len(glass_prots)} GLASS2 proteins...")
with Pool(min(cpu_count(), 32)) as pool:
    results = pool.map(check_homolog, glass_prots)
homologs = {r for r in results if r is not None}
log.info(f"  Found {len(homologs)} homologs (>=30% identity)")
novel_prots = set(glass_prots) - homologs
log.info(f"  Novel proteins (<30%): {len(novel_prots)}")

# ================================================================
# Load predictions from previous eval
# ================================================================
log.info("Loading predictions...")
pred_df = pd.read_csv(RESULTS / "glass2_concise_anchor_results.csv")
log.info(f"  {len(pred_df)} interactions with predictions")

# Filter to novel proteins
pred_novel = pred_df[pred_df.uniprot_id.isin(novel_prots)].copy()
log.info(f"  Novel (<30%): {len(pred_novel)} int, {pred_novel.uniprot_id.nunique()} prot")

# ================================================================
# Per-protein CI computation
# ================================================================
def compute_per_protein_ci(df, col):
    results = {}
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        if len(yt) < 3 or yp.std() < 1e-8:
            if len(yt) >= 3: results[uid] = 0.5
            continue
        c = d = t = 0
        for i, j in combinations(range(len(yt)), 2):
            if yt[i] == yt[j]: continue
            if (yp[i]>yp[j]) == (yt[i]>yt[j]): c += 1
            elif yp[i] == yp[j]: t += 1
            else: d += 1
        tot = c+d+t
        results[uid] = (c+0.5*t)/tot if tot else 0.5
    return results

from sklearn.metrics import roc_auc_score

def compute_per_protein_auroc(df, col, hi=7.0, lo=5.0):
    results = {}
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        mask = (yt >= hi) | (yt <= lo)
        if mask.sum() < 2: continue
        labels = (yt[mask] >= hi).astype(int)
        if len(set(labels)) < 2: continue
        results[uid] = roc_auc_score(labels, yp[mask])
    return results

# ================================================================
# Evaluate on ALL, NOVEL, and fair subset
# ================================================================
for label, df in [("All GLASS2", pred_df), ("Novel <30%", pred_novel)]:
    log.info(f"\n{'='*60}")
    log.info(f"  {label}: {len(df)} int, {df.uniprot_id.nunique()} prot")
    log.info(f"{'='*60}")

    for col, name in [("concise_pred", "CoNCISE"), ("anchor_pred", "ConciseAnchor")]:
        valid = df[~df[col].isna()]
        ci_dict = compute_per_protein_ci(valid, col)
        ci_vals = np.array(list(ci_dict.values()))
        auroc_dict = compute_per_protein_auroc(valid, col)
        auroc_vals = np.array(list(auroc_dict.values()))
        log.info(f"  {name:<22s} CI={ci_vals.mean():.3f}±{ci_vals.std():.3f}  "
                 f"AUROC={auroc_vals.mean():.3f}  ({len(valid)}/{len(df)} int, "
                 f"{valid.uniprot_id.nunique()} prot)")

    # Fair comparison
    both = df.dropna(subset=["concise_pred", "anchor_pred"])
    if len(both) > 10:
        log.info(f"\n  Fair subset ({len(both)} int, {both.uniprot_id.nunique()} prot):")
        for col, name in [("concise_pred", "CoNCISE"), ("anchor_pred", "ConciseAnchor")]:
            ci_dict = compute_per_protein_ci(both, col)
            ci_vals = np.array(list(ci_dict.values()))
            auroc_dict = compute_per_protein_auroc(both, col)
            auroc_vals = np.array(list(auroc_dict.values()))
            log.info(f"    {name:<22s} CI={ci_vals.mean():.3f}±{ci_vals.std():.3f}  "
                     f"AUROC={auroc_vals.mean():.3f}")

# ================================================================
# Per-protein scatter plots
# ================================================================
log.info("\nGenerating per-protein plots...")

# Use novel proteins with both predictions
both_novel = pred_novel.dropna(subset=["concise_pred", "anchor_pred"])
if len(both_novel) < 10:
    both_novel = pred_df.dropna(subset=["concise_pred", "anchor_pred"])
    plot_label = "All GLASS2"
else:
    plot_label = "Novel GLASS2 (<30% identity)"

ci_concise = compute_per_protein_ci(both_novel, "concise_pred")
ci_anchor = compute_per_protein_ci(both_novel, "anchor_pred")

# Get proteins with both CIs
common_prots = sorted(set(ci_concise.keys()) & set(ci_anchor.keys()))
ci_c = np.array([ci_concise[p] for p in common_prots])
ci_a = np.array([ci_anchor[p] for p in common_prots])
n_int = np.array([len(both_novel[both_novel.uniprot_id == p]) for p in common_prots])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: Per-protein CI scatter
sc = ax1.scatter(ci_c, ci_a, c=np.log10(n_int + 1), cmap="viridis", s=40,
                  edgecolor="black", linewidth=0.3, alpha=0.8)
ax1.plot([0.3, 1], [0.3, 1], "k--", linewidth=0.8, alpha=0.5)
ax1.fill_between([0.3, 1], [0.3, 1], [1, 1], alpha=0.05, color="blue")
ax1.fill_between([0.3, 1], [0.3, 1], [0.3, 0.3], alpha=0.05, color="red")

wins_a = (ci_a > ci_c).sum()
wins_c = (ci_c > ci_a).sum()
ax1.text(0.35, 0.92, f"ConciseAnchor\nwins: {wins_a}/{len(common_prots)}",
         fontsize=9, color="#2E86AB", fontweight="bold")
ax1.text(0.7, 0.35, f"CoNCISE\nwins: {wins_c}/{len(common_prots)}",
         fontsize=9, color="#E8998D", fontweight="bold")

cb = plt.colorbar(sc, ax=ax1, shrink=0.8)
cb.set_label("log10(n interactions)", fontsize=8)
ax1.set_xlabel("CoNCISE per-protein CI")
ax1.set_ylabel("ConciseAnchor per-protein CI")
ax1.set_title(f"A) Per-protein CI — {plot_label}\n({len(common_prots)} proteins)")
ax1.set_xlim(0.3, 1.0); ax1.set_ylim(0.3, 1.0)
ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
ax1.axvline(0.5, color="gray", linestyle=":", alpha=0.3)

# Panel B: CI distribution comparison (box + strip)
data = [ci_c, ci_a]
labels = ["CoNCISE", "ConciseAnchor"]
colors = ["#E8998D", "#2E86AB"]
bp = ax2.boxplot(data, positions=[1, 2], widths=0.5, patch_artist=True,
                  showfliers=False, medianprops=dict(color="black", linewidth=1.5))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.6)

for i, (vals, c) in enumerate(zip(data, colors)):
    jitter = np.random.uniform(-0.12, 0.12, len(vals))
    ax2.scatter(i + 1 + jitter, vals, s=20, color=c, edgecolor="black",
                linewidth=0.3, alpha=0.7, zorder=3)

for i, vals in enumerate(data):
    ax2.text(i + 1, 1.02, f"mean={vals.mean():.3f}", ha="center", fontsize=8, fontweight="bold")

ax2.set_xticks([1, 2]); ax2.set_xticklabels(labels)
ax2.set_ylabel("Per-protein CI")
ax2.set_title(f"B) CI Distribution — {plot_label}")
ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="Random")
ax2.set_ylim(0.2, 1.1)

fig.suptitle(f"ConciseAnchor vs CoNCISE on GLASS2 Cross-Dataset Evaluation",
             fontsize=12, fontweight="bold", y=1.02)
plt.tight_layout()
out = RESULTS / "glass2_per_protein_comparison.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(str(out).replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
log.info(f"Saved {out}")
plt.close()

log.info("Done!")
