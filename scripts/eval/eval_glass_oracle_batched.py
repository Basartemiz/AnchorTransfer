"""GLASS2 oracle anchor eval — batched GPU predictions."""
import os, sys, json, pickle, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

from concise.model.concise import Concise
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")


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

glass = glass[glass.uniprot_id.isin(raygun) & glass.ligand_smiles.isin(morgan)]
log.info(f"GLASS2 covered: {len(glass)} int, {glass.uniprot_id.nunique()} prot")

# Oracle anchors
oracle_map = {}
for smi, grp in glass.groupby("ligand_smiles"):
    s = grp.sort_values("pki", ascending=False)
    candidates = [(u, p) for u, p in zip(s.uniprot_id.values, s.pki.values)
                   if p >= 7.0 and u in raygun]
    if candidates:
        oracle_map[smi] = candidates
log.info(f"Oracle anchors for {len(oracle_map)}/{glass.ligand_smiles.nunique()} drugs")

# Build evaluation dataframe (exclude self-predictions)
rows = []
for _, r in glass.iterrows():
    if r.ligand_smiles not in oracle_map:
        continue
    for au, ap in oracle_map[r.ligand_smiles]:
        if au != r.uniprot_id:
            rows.append({**r.to_dict(), "anchor_uid": au, "anchor_pki": ap})
            break
glass_ora = pd.DataFrame(rows)

# Quartiles
pki_vals = glass_ora.anchor_pki.values
q1, q2, q3 = np.percentile(pki_vals, [25, 50, 75])
glass_ora["anchor_q"] = glass_ora.anchor_pki.apply(
    lambda v: "Q1 weakest" if v <= q1 else "Q2" if v <= q2 else "Q3" if v <= q3 else "Q4 strongest"
)
log.info(f"Oracle subset: {len(glass_ora)} int, {glass_ora.uniprot_id.nunique()} prot")

# Batched prediction
BATCH = 512
fp_list = [torch.tensor(np.array(morgan[r.ligand_smiles], dtype=np.float32)) for _, r in glass_ora.iterrows()]
qe_list = [raygun[r.uniprot_id] for _, r in glass_ora.iterrows()]
ae_list = [raygun[r.anchor_uid] for _, r in glass_ora.iterrows()]

cp_all, ap_all = [], []
t0 = time.time()
with torch.no_grad():
    for start in range(0, len(fp_list), BATCH):
        end = min(start + BATCH, len(fp_list))
        fp_batch = torch.stack(fp_list[start:end]).to(DEVICE)
        qe_batch = torch.stack(qe_list[start:end]).to(DEVICE)
        ae_batch = torch.stack(ae_list[start:end]).to(DEVICE)

        cp_all.extend(concise(fp_batch, qe_batch).cpu().tolist())
        ap_all.extend(anchor_m(fp_batch, ae_batch, qe_batch).cpu().tolist())

        if (end) % 5000 < BATCH:
            log.info(f"  Predicted {end}/{len(fp_list)} ({time.time()-t0:.0f}s)")

glass_ora["concise_pred"] = cp_all
glass_ora["anchor_pred"] = ap_all
log.info(f"Predictions done in {time.time()-t0:.0f}s")


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


log.info("")
log.info("===== GLASS2 ORACLE ANCHORS =====")
log.info("  %-16s %5s  %6s %6s %6s  %6s %6s %6s" % (
    "Quartile", "n", "C_CI", "C_AUC", "C_RMS", "A_CI", "A_AUC", "A_RMS"))
for q in sorted(glass_ora.anchor_q.unique()):
    sub = glass_ora[glass_ora.anchor_q == q]
    cci, cau, crm = pp_metrics(sub, "concise_pred")
    aci, aau, arm = pp_metrics(sub, "anchor_pred")
    log.info("  %-16s %5d  %6.3f %6.3f %6.3f  %6.3f %6.3f %6.3f" % (
        q, len(sub), cci.mean(),
        cau.mean() if len(cau) > 0 else 0, crm.mean(),
        aci.mean(), aau.mean() if len(aau) > 0 else 0, arm.mean()))

cci, cau, crm = pp_metrics(glass_ora, "concise_pred")
aci, aau, arm = pp_metrics(glass_ora, "anchor_pred")
log.info("  %-16s %5d  %6.3f %6.3f %6.3f  %6.3f %6.3f %6.3f" % (
    "Overall", len(glass_ora), cci.mean(),
    cau.mean() if len(cau) > 0 else 0, crm.mean(),
    aci.mean(), aau.mean() if len(aau) > 0 else 0, arm.mean()))

glass_ora.to_csv("results/glass2_oracle_predictions.csv", index=False)
log.info("Saved results/glass2_oracle_predictions.csv")
