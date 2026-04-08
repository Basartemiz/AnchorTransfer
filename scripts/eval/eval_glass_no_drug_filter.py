"""GLASS2 eval: protein-overlap excluded only, NO drug filtering.
Also generates comparison plots across all 3 protocols."""
import os, sys, json, pickle, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")
from concise.model.concise import Concise
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConciseFixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Concise(
            drug_layers=[[32], [32], [32]], ligand_dim=2048, residue_dim=1280,
            drug_dim=128, proj_dim=256, nheads=32, activation="tanh",
            cosine_prediction=False,
        )
        self.backbone.final = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 1),
        )

    def forward(self, d, p):
        return self.backbone(d, p, is_morgan_fingerprint=True)["binding"]


# Load models
ckpt = torch.load("models/concise_bdb_fixed/best_model.pt", map_location="cpu", weights_only=False)
concise = ConciseFixed()
concise.load_state_dict(ckpt["model_state_dict"])
concise = concise.eval().to(DEVICE)

ckpt2 = torch.load("models/concise_anchor_bdb/best_model.pt", map_location="cpu", weights_only=False)
anchor_m = ConciseAnchorBilinear()
anchor_m.load_state_dict(ckpt2["model_state_dict"])
anchor_m = anchor_m.eval().to(DEVICE)
log.info("Models loaded")

# Load data
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
raygun = torch.load("data/processed/raygun_bdb_embeddings.pt", map_location="cpu", weights_only=False)
morgan = pickle.load(open("data/processed/concise_bdb_morgan_fp.pkl", "rb"))
seqs = json.load(open("data/processed/merged_sequences.json"))

# BDB train split
bdb = pd.read_csv("data/processed/bindingdb_interactions.csv")
bdb_clean = bdb[~bdb.uniprot_id.str.contains(",", na=False)]
prots = sorted(bdb_clean.uniprot_id.unique())
np.random.seed(42)
np.random.shuffle(prots)
nt = int(len(prots) * 0.1)
nv = int(len(prots) * 0.1)
train_prots = set(prots[nt + nv:])
bdb_train = bdb_clean[bdb_clean.uniprot_id.isin(train_prots)].copy()

# Protein-only filtering (NO drug filtering)
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def canonical(smi):
    try:
        mol = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else str(smi)
    except:
        return str(smi)


bdb_train_seqs = {seqs.get(uid, "") for uid in train_prots if uid in seqs}
bdb_train_seqs.discard("")
prot_overlap = set()
for uid in glass.uniprot_id.unique():
    if uid in train_prots or (uid in seqs and seqs[uid] in bdb_train_seqs):
        prot_overlap.add(uid)

glass_prot_only = glass[~glass.uniprot_id.isin(prot_overlap)]
log.info("Protein-only filter: %d -> %d int, %d -> %d prot" % (
    len(glass), len(glass_prot_only), glass.uniprot_id.nunique(),
    glass_prot_only.uniprot_id.nunique()))

# Coverage filter
glass_prot_only = glass_prot_only[
    glass_prot_only.uniprot_id.isin(raygun) & glass_prot_only.ligand_smiles.isin(morgan)
]
log.info("After coverage: %d int, %d prot" % (
    len(glass_prot_only), glass_prot_only.uniprot_id.nunique()))

# Tanimoto anchor retrieval from BDB (including overlapping drugs)
bdb_train["canon"] = bdb_train.ligand_smiles.apply(canonical)
glass_prot_only = glass_prot_only.copy()
glass_prot_only["canon"] = glass_prot_only.ligand_smiles.apply(canonical)
bdb_strong = bdb_train[bdb_train.pki >= 7]

strong_fps = {}
for smi in bdb_strong.canon.unique():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        strong_fps[smi] = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
log.info("Anchor pool: %d drugs" % len(strong_fps))

amap = {}
eval_drugs = list(glass_prot_only.canon.unique())
for i, es in enumerate(eval_drugs):
    mol = Chem.MolFromSmiles(es)
    if not mol:
        continue
    fq = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)
    bs, bsmi = -1, None
    for cs, cf in strong_fps.items():
        s = DataStructs.TanimotoSimilarity(fq, cf)
        if s > bs:
            bs, bsmi = s, cs
    if bsmi:
        rows = bdb_strong[bdb_strong.canon == bsmi]
        best = rows.loc[rows.pki.idxmax()]
        amap[es] = {"uid": best.uniprot_id, "pki": best.pki, "tc": bs}
    if (i + 1) % 1000 == 0:
        log.info("  Anchors: %d/%d" % (i + 1, len(eval_drugs)))

glass_eval = glass_prot_only[glass_prot_only.canon.isin(amap)].copy()
pki_vals = [amap[s]["pki"] for s in glass_eval.canon]
q1, q2, q3 = np.percentile(pki_vals, [25, 50, 75])
glass_eval["anchor_uid"] = [amap[s]["uid"] for s in glass_eval.canon]
glass_eval["anchor_pki"] = pki_vals
glass_eval["tanimoto"] = [amap[s]["tc"] for s in glass_eval.canon]
glass_eval["anchor_q"] = glass_eval.anchor_pki.apply(
    lambda v: "Q1 weakest" if v <= q1 else "Q2" if v <= q2 else "Q3" if v <= q3 else "Q4 strongest"
)
log.info("Eval set (protein-only filter): %d int, %d prot" % (
    len(glass_eval), glass_eval.uniprot_id.nunique()))

# Batched prediction
BATCH = 512
fp_list = [torch.tensor(np.array(morgan[r.ligand_smiles], dtype=np.float32))
           for _, r in glass_eval.iterrows()]
qe_list = [raygun[r.uniprot_id] for _, r in glass_eval.iterrows()]
ae_list = [raygun[r.anchor_uid] for _, r in glass_eval.iterrows()]

cp_all, ap_all = [], []
t0 = time.time()
with torch.no_grad():
    for start in range(0, len(fp_list), BATCH):
        end = min(start + BATCH, len(fp_list))
        fp_b = torch.stack(fp_list[start:end]).to(DEVICE)
        qe_b = torch.stack(qe_list[start:end]).to(DEVICE)
        ae_b = torch.stack(ae_list[start:end]).to(DEVICE)
        cp_all.extend(concise(fp_b, qe_b).cpu().tolist())
        ap_all.extend(anchor_m(fp_b, ae_b, qe_b).cpu().tolist())
        if end % 5000 < BATCH:
            log.info("  Predicted %d/%d (%.0fs)" % (end, len(fp_list), time.time() - t0))

glass_eval["concise_pred"] = cp_all
glass_eval["anchor_pred"] = ap_all
log.info("Predictions done in %.0fs" % (time.time() - t0))


# Metrics
def pp_metrics(df, col):
    cis, aurocs, rmses = [], [], []
    for p in df.uniprot_id.unique():
        s = df[df.uniprot_id == p]
        if len(s) < 3:
            continue
        yt, yp = s.pki.values, s[col].values
        if yp.std() < 1e-8:
            cis.append(0.5)
        else:
            c = d = t = 0
            for i, j in combinations(range(len(yt)), 2):
                if yt[i] == yt[j]:
                    continue
                if (yp[i] > yp[j]) == (yt[i] > yt[j]):
                    c += 1
                elif yp[i] == yp[j]:
                    t += 1
                else:
                    d += 1
            tot = c + d + t
            cis.append((c + 0.5 * t) / tot if tot > 0 else 0.5)
        rmses.append(np.sqrt(mean_squared_error(yt, yp)))
        mask = (yt >= 7) | (yt <= 5)
        if mask.sum() >= 2:
            labels = (yt[mask] >= 7).astype(int)
            if len(set(labels)) == 2:
                aurocs.append(roc_auc_score(labels, yp[mask]))
    return np.array(cis), np.array(aurocs), np.array(rmses)


def pp_ci_dict(df, col):
    r = {}
    for p in df.uniprot_id.unique():
        s = df[df.uniprot_id == p]
        if len(s) < 3:
            continue
        yt, yp = s.pki.values, s[col].values
        if yp.std() < 1e-8:
            r[p] = 0.5
            continue
        c = d = t = 0
        for i, j in combinations(range(len(yt)), 2):
            if yt[i] == yt[j]:
                continue
            if (yp[i] > yp[j]) == (yt[i] > yt[j]):
                c += 1
            elif yp[i] == yp[j]:
                t += 1
            else:
                d += 1
        tot = c + d + t
        r[p] = (c + 0.5 * t) / tot if tot > 0 else 0.5
    return r


log.info("")
log.info("===== GLASS2 PROTEIN-ONLY FILTER (no drug exclusion) =====")
hdr = "  %-16s %5s  %6s %6s %6s  %6s %6s %6s" % (
    "Quartile", "n", "C_CI", "C_AUC", "C_RMS", "A_CI", "A_AUC", "A_RMS")
log.info(hdr)
for q in sorted(glass_eval.anchor_q.unique()):
    sub = glass_eval[glass_eval.anchor_q == q]
    cci, cau, crm = pp_metrics(sub, "concise_pred")
    aci, aau, arm = pp_metrics(sub, "anchor_pred")
    log.info("  %-16s %5d  %6.3f %6.3f %6.3f  %6.3f %6.3f %6.3f" % (
        q, len(sub), cci.mean(),
        cau.mean() if len(cau) > 0 else 0, crm.mean(),
        aci.mean(), aau.mean() if len(aau) > 0 else 0, arm.mean()))
cci, cau, crm = pp_metrics(glass_eval, "concise_pred")
aci, aau, arm = pp_metrics(glass_eval, "anchor_pred")
log.info("  %-16s %5d  %6.3f %6.3f %6.3f  %6.3f %6.3f %6.3f" % (
    "Overall", len(glass_eval), cci.mean(),
    cau.mean() if len(cau) > 0 else 0, crm.mean(),
    aci.mean(), aau.mean() if len(aau) > 0 else 0, arm.mean()))

glass_eval.to_csv("results/glass2_prot_only_filter_predictions.csv", index=False)
log.info("Saved predictions")

# ===== COMPARISON PLOTS =====
log.info("Generating comparison plots...")

glass_tanimoto = pd.read_csv("results/bdb_to_glass_predictions.csv")
glass_oracle = pd.read_csv("results/glass2_oracle_predictions.csv")

protocols = [
    ("Tanimoto\n(prot+drug filter)\n%d prot" % glass_tanimoto.uniprot_id.nunique(), glass_tanimoto),
    ("Tanimoto\n(prot-only filter)\n%d prot" % glass_eval.uniprot_id.nunique(), glass_eval),
    ("Oracle\n(no filter)\n%d prot" % glass_oracle.uniprot_id.nunique(), glass_oracle),
]

# CI distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (title, df) in zip(axes, protocols):
    ci_c = list(pp_ci_dict(df, "concise_pred").values())
    ci_a = list(pp_ci_dict(df, "anchor_pred").values())
    if ci_c:
        vp = ax.violinplot([ci_c], positions=[0], showmeans=True, showmedians=True, widths=0.8)
        for pc in vp["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.7)
    if ci_a:
        vp = ax.violinplot([ci_a], positions=[1], showmeans=True, showmedians=True, widths=0.8)
        for pc in vp["bodies"]:
            pc.set_facecolor("indianred")
            pc.set_alpha(0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["CoNCISE", "ConciseAnchor"], fontsize=11)
    ax.set_ylabel("Per-protein CI", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    if ci_c:
        ax.text(0, np.mean(ci_c) + 0.02, "%.3f" % np.mean(ci_c),
                ha="center", fontsize=10, color="steelblue", fontweight="bold")
    if ci_a:
        ax.text(1, np.mean(ci_a) + 0.02, "%.3f" % np.mean(ci_a),
                ha="center", fontsize=10, color="indianred", fontweight="bold")
plt.suptitle("GLASS2: CoNCISE vs ConciseAnchor across evaluation protocols",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fig_glass2_protocol_comparison_ci.png", dpi=150, bbox_inches="tight")
log.info("Saved fig_glass2_protocol_comparison_ci.png")
plt.close()

# Quartile CI comparison
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (title, df) in zip(axes, protocols):
    quartiles = sorted(df.anchor_q.unique())
    for i, q in enumerate(quartiles):
        sub = df[df.anchor_q == q]
        ci_c = list(pp_ci_dict(sub, "concise_pred").values())
        ci_a = list(pp_ci_dict(sub, "anchor_pred").values())
        pos_c = i * 3
        pos_a = i * 3 + 1
        if ci_c:
            vp = ax.violinplot([ci_c], positions=[pos_c], showmeans=True, widths=0.8)
            for pc in vp["bodies"]:
                pc.set_facecolor("steelblue")
                pc.set_alpha(0.7)
        if ci_a:
            vp = ax.violinplot([ci_a], positions=[pos_a], showmeans=True, widths=0.8)
            for pc in vp["bodies"]:
                pc.set_facecolor("indianred")
                pc.set_alpha(0.7)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(quartiles))])
    ax.set_xticklabels([q.replace(" ", "\n") for q in quartiles], fontsize=9)
    ax.set_ylabel("Per-protein CI", fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(handles=[
        Patch(facecolor="steelblue", alpha=0.7, label="CoNCISE"),
        Patch(facecolor="indianred", alpha=0.7, label="ConciseAnchor"),
    ], loc="upper left", fontsize=9)
plt.suptitle("GLASS2: Quartile CI across evaluation protocols",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fig_glass2_protocol_quartile_ci.png", dpi=150, bbox_inches="tight")
log.info("Saved fig_glass2_protocol_quartile_ci.png")
plt.close()

# Summary bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metric_names = ["CI", "AUROC", "RMSE"]
for ax, metric in zip(axes, metric_names):
    c_vals, a_vals, labels = [], [], []
    for title, df in protocols:
        cci, cau, crm = pp_metrics(df, "concise_pred")
        aci, aau, arm = pp_metrics(df, "anchor_pred")
        short = title.split("\n")[1].strip("()")
        labels.append(short)
        if metric == "CI":
            c_vals.append(cci.mean())
            a_vals.append(aci.mean())
        elif metric == "AUROC":
            c_vals.append(cau.mean() if len(cau) > 0 else 0)
            a_vals.append(aau.mean() if len(aau) > 0 else 0)
        else:
            c_vals.append(crm.mean())
            a_vals.append(arm.mean())
    x = np.arange(len(labels))
    ax.bar(x - 0.18, c_vals, 0.35, color="steelblue", alpha=0.8, label="CoNCISE")
    ax.bar(x + 0.18, a_vals, 0.35, color="indianred", alpha=0.8, label="ConciseAnchor")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(metric, fontsize=13)
    ax.legend(fontsize=9)
    if metric != "RMSE":
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3)
    for i, (cv, av) in enumerate(zip(c_vals, a_vals)):
        ax.text(i - 0.18, cv + 0.01, "%.3f" % cv, ha="center", fontsize=8, color="steelblue")
        ax.text(i + 0.18, av + 0.01, "%.3f" % av, ha="center", fontsize=8, color="indianred")
plt.suptitle("GLASS2: Protocol comparison summary", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/fig_glass2_protocol_summary_bar.png", dpi=150, bbox_inches="tight")
log.info("Saved fig_glass2_protocol_summary_bar.png")
plt.close()
log.info("All done")
