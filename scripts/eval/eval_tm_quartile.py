"""Evaluate V2 AUROC by Foldseek TM quartile."""
import json, pandas as pd, torch, numpy as np, logging
from sklearn.metrics import roc_auc_score
from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2, encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
device = torch.device("cuda")

esm2 = torch.load("data/processed/esm2_35m_dtc_proteins_full.pt", map_location="cpu", weights_only=False)
bench_emb = torch.load("data/processed/esm2_35m_benchmark.pt", map_location="cpu", weights_only=False)
try:
    fs_extra = torch.load("data/processed/esm2_35m_foldseek_anchors.pt", map_location="cpu", weights_only=False)
except Exception:
    fs_extra = {}
try:
    fs_extra2 = torch.load("data/processed/esm2_35m_foldseek_anchors_all.pt", map_location="cpu", weights_only=False)
except Exception:
    fs_extra2 = {}
esm2.update(bench_emb)
esm2.update(fs_extra)
esm2.update(fs_extra2)

fs_hits = json.load(open("results/foldseek_idp_hits/foldseek_hits.json"))
bench = pd.read_csv("data/raw/benchmark_affinity.csv")

ckpt = torch.load("models/anchor_transfer_v2_strongest/best_model.pt", map_location=device, weights_only=False)
model = AnchorTransferDTAv2(esm2_dim=480).to(device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
dtc_v = dtc[dtc.uniprot_id.isin(esm2)].copy()
idx = dtc_v.groupby("ligand_smiles")["pki"].idxmax()
drug_strongest = dict(zip(dtc_v.loc[idx].ligand_smiles, dtc_v.loc[idx].uniprot_id))

# TM values per IDP
tm_values = {}
for uid, info in fs_hits.items():
    tm_values[uid] = info["top10"][0]["alntm"] if info["top10"] else 0

tms = np.array(list(tm_values.values()))
q1, q2, q3 = np.percentile(tms, [25, 50, 75])

def get_quartile(tm):
    if tm <= q1:
        return "Q1"
    elif tm <= q2:
        return "Q2"
    elif tm <= q3:
        return "Q3"
    else:
        return "Q4"

results = []
for uid, grp in bench.groupby("uniprot_id"):
    ptype = grp.protein_type.iloc[0]
    if uid not in esm2 or ptype != "idp":
        continue

    q_emb = esm2[uid]
    fs_anchor = None
    best_tm = 0
    if uid in fs_hits:
        best_tm = fs_hits[uid]["top10"][0]["alntm"] if fs_hits[uid]["top10"] else 0
        for h in fs_hits[uid]["top10"]:
            if h["target"] in esm2 and h["target"] != uid:
                fs_anchor = h["target"]
                break

    quartile = get_quartile(best_tm) if uid in tm_values else "no_fs"

    preds, probs, trues = [], [], []
    for _, row in grp.iterrows():
        smi = row["ligand_smiles"]
        anchor = None
        if smi in drug_strongest:
            a = drug_strongest[smi]
            if a != uid and a in esm2:
                anchor = a
        if not anchor and fs_anchor:
            anchor = fs_anchor
        if not anchor:
            anchor = uid

        a_t = esm2[anchor].unsqueeze(0).to(device)
        q_t = q_emb.unsqueeze(0).to(device)
        d_t = torch.tensor([encode_smiles(smi)], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(a_t, q_t, d_t)
        preds.append(out["pki_pred"].item())
        probs.append(out["binding_prob"].item())
        trues.append(row["pki"])

    preds = np.array(preds)
    probs = np.array(probs)
    trues = np.array(trues)
    labels = (trues >= 7.0).astype(int)
    auroc = roc_auc_score(labels, probs) if len(set(labels)) == 2 else float("nan")

    results.append({
        "uid": uid, "quartile": quartile, "best_tm": best_tm,
        "auroc": auroc, "n": len(trues), "has_fs": uid in fs_hits,
    })

df = pd.DataFrame(results)
logger.info("Total IDPs: %d", len(df))

logger.info("=== AUROC BY TM QUARTILE (Q1=%.3f, Q2=%.3f, Q3=%.3f) ===", q1, q2, q3)
for q in ["Q1", "Q2", "Q3", "Q4", "no_fs"]:
    sub = df[df.quartile == q]
    if len(sub) == 0:
        continue
    valid = sub.auroc.dropna()
    if len(valid) > 0:
        logger.info("  %s: n=%d auroc_valid=%d AUROC=%.3f (+/-%.3f) mean_TM=%.3f",
                     q, len(sub), len(valid), valid.mean(), valid.std(), sub.best_tm.mean())
    else:
        logger.info("  %s: n=%d auroc_valid=0", q, len(sub))

valid_all = df.auroc.dropna()
logger.info("  ALL: n=%d auroc_valid=%d AUROC=%.3f", len(df), len(valid_all), valid_all.mean())
