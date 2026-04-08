"""Evaluate pretrained CoNCISE + ConciseAnchor on GLASS2.

GLASS2 has continuous pKi values. We evaluate:
- Per-protein CI (concordance index) for ranking
- AUROC (pKi >= 7 as positive)
- Both models on same subset (drugs with anchors in MooDengDB training)
"""
import os, sys, json, logging, random, time, pickle, hashlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load MooDengDB training info (for anchors)
# ================================================================
log.info("Loading MooDengDB training data (for anchors)...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
train_raw.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
train_raw["prot_id"] = train_raw.sequence.apply(seq_to_id)

# Load cached Raygun + FPs from MooDengDB
raygun_embs = torch.load(RESULTS / "raygun_moodeng_embeddings.pt", map_location="cpu", weights_only=False)
fp_dict = pickle.load(open(RESULTS / "morgan_moodeng_fp.pkl", "rb"))
log.info(f"MooDengDB Raygun: {len(raygun_embs)}, FPs: {len(fp_dict)}")

# Build anchors from training positives
train_pos = train_raw[train_raw.label == 1]
train_pos = train_pos[train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors:
        drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs with positive binders")
del train_raw, train_pos

# ================================================================
# Load GLASS2
# ================================================================
log.info("Loading GLASS2...")
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
glass_seqs = json.load(open("data/raw/glass/glass2_sequences.json"))
log.info(f"GLASS2: {len(glass)} int, {glass.uniprot_id.nunique()} prot, "
         f"{glass.ligand_smiles.nunique()} drugs")

# ================================================================
# Compute GLASS2 Raygun embeddings
# ================================================================
GLASS_RAYGUN = RESULTS / "raygun_glass_embeddings.pt"

if GLASS_RAYGUN.exists():
    glass_raygun = torch.load(GLASS_RAYGUN, map_location="cpu", weights_only=False)
    log.info(f"Loaded GLASS2 Raygun: {len(glass_raygun)} proteins")
else:
    log.info("Computing GLASS2 Raygun embeddings (streaming)...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(DEVICE)
    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)

    items = [(uid, seq[:1022].upper()) for uid, seq in glass_seqs.items() if len(seq) >= 25]
    log.info(f"  {len(items)} GLASS2 proteins")
    glass_raygun = {}
    for idx, (uid, seq) in enumerate(items):
        try:
            _, _, toks = bc([(uid, seq)])
            with torch.no_grad():
                out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
                esm_emb = out["representations"][33][0, 1:len(seq)+1, :]
                ray_out = raygun_model(esm_emb.unsqueeze(0))
                glass_raygun[uid] = ray_out[1].squeeze(0).cpu()  # (50, 1280)
            del out, esm_emb, ray_out, toks
        except Exception as e:
            pass
        if (idx+1) % 100 == 0:
            log.info(f"    {idx+1}/{len(items)} done")
    del esm_model, raygun_model; torch.cuda.empty_cache()
    torch.save(glass_raygun, str(GLASS_RAYGUN))
    log.info(f"  Saved {len(glass_raygun)} GLASS2 Raygun embeddings")

# Compute GLASS2 drug FPs
log.info("GLASS2 drug FPs...")
from rdkit import Chem
from rdkit.Chem import AllChem
glass_fp = {}
for smi in glass.ligand_smiles.unique():
    if smi in fp_dict:
        glass_fp[smi] = fp_dict[smi]
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                glass_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
log.info(f"  {len(glass_fp)}/{glass.ligand_smiles.nunique()} drugs")

# Filter GLASS2
glass = glass[glass.uniprot_id.isin(glass_raygun) & glass.ligand_smiles.isin(glass_fp)].copy()
log.info(f"GLASS2 filtered: {len(glass)} int, {glass.uniprot_id.nunique()} prot")

# ================================================================
# Metrics
# ================================================================
from sklearn.metrics import roc_auc_score

def pp_ci(df, col):
    cis = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        if len(yt) < 3 or yp.std() < 1e-8:
            if len(yt) >= 3: cis.append(0.5)
            continue
        c = d = t = 0
        for i, j in combinations(range(len(yt)), 2):
            if yt[i] == yt[j]: continue
            if (yp[i]>yp[j]) == (yt[i]>yt[j]): c += 1
            elif yp[i] == yp[j]: t += 1
            else: d += 1
        tot = c+d+t
        cis.append((c+0.5*t)/tot if tot else 0.5)
    return np.array(cis)

def pp_auroc(df, col, hi=7.0, lo=5.0):
    out = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        mask = (yt >= hi) | (yt <= lo)
        if mask.sum() < 2: continue
        labels = (yt[mask] >= hi).astype(int)
        if len(set(labels)) < 2: continue
        out.append(roc_auc_score(labels, yp[mask]))
    return np.array(out)

# ================================================================
# Pretrained CoNCISE predictions on GLASS2
# ================================================================
log.info(f"\n{'='*60}")
log.info("Pretrained CoNCISE on GLASS2")
log.info(f"{'='*60}")

concise_model = torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v2", pretrained=True)
concise_model = concise_model.eval().to(DEVICE)

concise_preds = np.full(len(glass), np.nan)
BS = 256
uids = glass.uniprot_id.values
smis = glass.ligand_smiles.values
with torch.no_grad():
    for i in range(0, len(glass), BS):
        batch_uids = uids[i:i+BS]
        batch_smis = smis[i:i+BS]
        valid_idx, fps_list, embs_list = [], [], []
        for j, (uid, smi) in enumerate(zip(batch_uids, batch_smis)):
            if uid in glass_raygun and smi in glass_fp:
                valid_idx.append(i + j)
                fps_list.append(glass_fp[smi])
                embs_list.append(glass_raygun[uid])
        if not fps_list: continue
        fps_t = torch.tensor(np.array(fps_list)).to(DEVICE)
        embs_t = torch.stack(embs_list).to(DEVICE)
        pred = concise_model(fps_t, embs_t, is_morgan_fingerprint=True)["binding"]
        scores = ((pred + 1) / 2).cpu().numpy()
        for j, idx in enumerate(valid_idx):
            concise_preds[idx] = scores[j]
        if (i + BS) % 20000 < BS:
            log.info(f"  {min(i+BS, len(glass))}/{len(glass)}")

glass["concise_pred"] = concise_preds
log.info(f"  CoNCISE: {np.sum(~np.isnan(concise_preds))}/{len(glass)} predictions")
del concise_model; torch.cuda.empty_cache()

# ================================================================
# ConciseAnchor predictions on GLASS2
# ================================================================
log.info("ConciseAnchor on GLASS2...")
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(
            ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)
    def forward(self, drug_fp, anchor_emb, query_emb):
        return self.backbone(drug_fp, anchor_emb, query_emb)

ANCHOR_BEST = Path("models/concise_anchor_moodeng/best_model.pt")
model_a = ConciseAnchorBinary().to(DEVICE)
ckpt = torch.load(str(ANCHOR_BEST), map_location=DEVICE, weights_only=False)
model_a.load_state_dict(ckpt["model_state_dict"]); model_a.eval()

anchor_preds = np.full(len(glass), np.nan)
with torch.no_grad():
    for i in range(0, len(glass), BS):
        batch_uids = uids[i:i+BS]
        batch_smis = smis[i:i+BS]
        valid_idx, fps_list, anc_list, qry_list = [], [], [], []
        for j, (uid, smi) in enumerate(zip(batch_uids, batch_smis)):
            if smi not in drug_to_anchors or smi not in glass_fp or uid not in glass_raygun: continue
            anchor = None
            for a in drug_to_anchors[smi]:
                if a != uid: anchor = a; break
            if anchor is None: continue
            valid_idx.append(i + j)
            fps_list.append(glass_fp[smi])
            anc_list.append(raygun_embs[anchor])
            qry_list.append(glass_raygun[uid])
        if not fps_list: continue
        fps_t = torch.tensor(np.array(fps_list)).to(DEVICE)
        anc_t = torch.stack(anc_list).to(DEVICE)
        qry_t = torch.stack(qry_list).to(DEVICE)
        pred = torch.sigmoid(model_a(fps_t, anc_t, qry_t)).cpu().numpy()
        for j, idx in enumerate(valid_idx):
            anchor_preds[idx] = pred[j]
        if (i + BS) % 20000 < BS:
            log.info(f"  {min(i+BS, len(glass))}/{len(glass)}")

glass["anchor_pred"] = anchor_preds
log.info(f"  ConciseAnchor: {np.sum(~np.isnan(anchor_preds))}/{len(glass)} predictions")
del model_a; torch.cuda.empty_cache()

# ================================================================
# Results
# ================================================================
log.info(f"\n{'='*60}")
log.info(f"GLASS2 RESULTS ({len(glass)} int, {glass.uniprot_id.nunique()} proteins)")
log.info(f"{'='*60}")

methods = ["concise_pred", "anchor_pred"]
names = ["Pretrained CoNCISE", "ConciseAnchor"]

log.info(f"\n  {'Method':<22s} {'CI':>7s} {'AUROC':>7s} {'Coverage':>10s}")
log.info(f"  {'-'*50}")
for col, name in zip(methods, names):
    valid = ~np.isnan(glass[col])
    ci = pp_ci(glass[valid], col)
    auc = pp_auroc(glass[valid], col)
    log.info(f"  {name:<22s} {ci.mean():7.3f} {auc.mean() if len(auc) else 0:7.3f} "
             f"{valid.sum():>5d}/{len(glass)}")

# Fair comparison on same subset
valid_both = ~np.isnan(glass.concise_pred) & ~np.isnan(glass.anchor_pred)
if valid_both.sum() > 10:
    log.info(f"\n  ── Fair comparison (same {valid_both.sum()} interactions) ──")
    for col, name in zip(methods, names):
        ci = pp_ci(glass[valid_both], col)
        auc = pp_auroc(glass[valid_both], col)
        log.info(f"  {name:<22s} {ci.mean():7.3f} {auc.mean() if len(auc) else 0:7.3f}")

# Per-quartile (by pKi)
log.info(f"\n  ── pKi Quartile Breakdown ──")
q_edges = np.quantile(glass.pki.values, [0, 0.25, 0.5, 0.75, 1.0])
q_labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
glass["pki_q"] = pd.cut(glass.pki, bins=q_edges, labels=q_labels, include_lowest=True)
for col, name in zip(methods, names):
    log.info(f"\n  {name}:")
    for ql in q_labels:
        qdf = glass[glass.pki_q == ql]
        valid = ~np.isnan(qdf[col])
        if valid.sum() < 10: continue
        ci = pp_ci(qdf[valid], col)
        log.info(f"    {ql:<16s} CI={ci.mean():.3f} (n={valid.sum()}, {qdf[valid].uniprot_id.nunique()} prot)")

glass.to_csv(RESULTS / "glass2_concise_anchor_results.csv", index=False)
log.info(f"\nSaved results/glass2_concise_anchor_results.csv")
log.info("Done!")
