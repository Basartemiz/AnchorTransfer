"""OOD eval on anchor subset only. CoNCISE + ConciseAnchor + Prot-kNN k=1.
Only evaluates interactions where ConciseAnchor has coverage (same subset for all).
"""
import os, sys, json, logging, hashlib, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
BS = 1024

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load training data
# ================================================================
log.info("Loading MooDengDB training data...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
train_raw.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
train_raw["prot_id"] = train_raw.sequence.apply(seq_to_id)

raygun_embs = torch.load(RESULTS / "raygun_moodeng_embeddings.pt", map_location="cpu", weights_only=False)
fp_dict = pickle.load(open(RESULTS / "morgan_moodeng_fp.pkl", "rb"))

# Load OOD Raygun cache
OOD_RAYGUN = RESULTS / "raygun_ood_embeddings.pt"
if OOD_RAYGUN.exists():
    ood_raygun = torch.load(str(OOD_RAYGUN), map_location="cpu", weights_only=False)
    for pid, emb in ood_raygun.items():
        raygun_embs[pid] = emb
    log.info(f"Merged {len(ood_raygun)} OOD Raygun embeddings")
log.info(f"Total Raygun: {len(raygun_embs)}, FPs: {len(fp_dict)}")

# Anchors
train_pos = train_raw[train_raw.label == 1]
train_pos = train_pos[train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors: drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs")

# ================================================================
# Load OOD test, filter to anchor subset immediately
# ================================================================
log.info("Loading OOD test set...")
ood_raw = pd.read_csv("data/moodeng-v2-extended/test.csv", sep="\t", low_memory=False)
ood_raw.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
ood_raw["prot_id"] = ood_raw.sequence.apply(seq_to_id)
log.info(f"OOD full: {len(ood_raw)} int, {(ood_raw.label==1).sum()} pos")

# OOD drug FPs
from rdkit import Chem
from rdkit.Chem import AllChem
ood_fp = {}
for smi in ood_raw.smiles.unique():
    if smi in fp_dict:
        ood_fp[smi] = fp_dict[smi]
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol: ood_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
log.info(f"OOD FPs: {len(ood_fp)}/{ood_raw.smiles.nunique()}")

# Filter to anchor subset: drug must have anchors, protein must have Raygun, drug must have FP
def has_valid_anchor(row):
    smi, pid = row.smiles, row.prot_id
    if smi not in drug_to_anchors or smi not in ood_fp or pid not in raygun_embs:
        return False
    for a in drug_to_anchors[smi]:
        if a != pid and a in raygun_embs:
            return True
    return False

log.info("Filtering to anchor subset...")
mask = ood_raw.apply(has_valid_anchor, axis=1)
ood_df = ood_raw[mask].copy()
log.info(f"Anchor subset: {len(ood_df)} int ({100*len(ood_df)/len(ood_raw):.1f}%), "
         f"{ood_df.prot_id.nunique()} prot, {ood_df.smiles.nunique()} drugs, "
         f"{(ood_df.label==1).sum()} pos, {(ood_df.label==0).sum()} neg")

del train_raw, train_pos, ood_raw
import gc; gc.collect()

# ================================================================
# kNN pool
# ================================================================
log.info("Building kNN pool...")
train_raw2 = pd.read_csv(MOODENG / "train.csv", low_memory=False)
train_raw2.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
train_raw2["prot_id"] = train_raw2.sequence.apply(seq_to_id)
train_pos2 = train_raw2[(train_raw2.label == 1) & train_raw2.prot_id.isin(raygun_embs) & train_raw2.smiles.isin(fp_dict)]

train_prot_ids = sorted(set(train_pos2.prot_id))
train_prot_embs = np.array([raygun_embs[p].mean(dim=0).numpy() for p in train_prot_ids])
train_prot_normed = train_prot_embs / (np.linalg.norm(train_prot_embs, axis=1, keepdims=True) + 1e-10)

train_drug_ids = sorted(set(train_pos2.smiles) & set(fp_dict.keys()))
train_drug_fps = np.array([fp_dict[d] for d in train_drug_ids])
train_drug_idx = {d: i for i, d in enumerate(train_drug_ids)}
train_prot_idx = {p: i for i, p in enumerate(train_prot_ids)}

INT_MAT = np.zeros((len(train_drug_ids), len(train_prot_ids)), dtype=np.float32)
for _, r in train_pos2.iterrows():
    di = train_drug_idx.get(r.smiles, -1)
    pi = train_prot_idx.get(r.prot_id, -1)
    if di >= 0 and pi >= 0: INT_MAT[di, pi] = 1.0
log.info(f"  INT_MAT: {INT_MAT.shape}, nnz={np.count_nonzero(INT_MAT)}")
del train_raw2, train_pos2; gc.collect()

# ================================================================
# Predictions (all on anchor subset only)
# ================================================================
prot_ids = ood_df.prot_id.values
smiles_arr = ood_df.smiles.values
labels = ood_df.label.values.astype(float)

log.info(f"\n{'='*60}")
log.info(f"  OOD Anchor Subset: {len(ood_df)} interactions")
log.info(f"  {int(labels.sum())} pos, {int((1-labels).sum())} neg (1:{(1-labels).sum()/max(labels.sum(),1):.1f})")
log.info(f"{'='*60}")

# CoNCISE
log.info("  CoNCISE predictions...")
concise_model = torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v2", pretrained=True)
concise_model = concise_model.eval().to(DEVICE)
p_concise = np.full(len(prot_ids), np.nan)
with torch.no_grad():
    for i in range(0, len(prot_ids), BS):
        bp = prot_ids[i:i+BS]; bs = smiles_arr[i:i+BS]
        idx, fps, embs = [], [], []
        for j, (pid, smi) in enumerate(zip(bp, bs)):
            if pid in raygun_embs and smi in ood_fp:
                idx.append(i+j); fps.append(ood_fp[smi]); embs.append(raygun_embs[pid])
        if not fps: continue
        pred = concise_model(torch.tensor(np.array(fps)).to(DEVICE),
                             torch.stack(embs).to(DEVICE), is_morgan_fingerprint=True)["binding"]
        scores = ((pred + 1) / 2).cpu().numpy()
        for j, ix in enumerate(idx): p_concise[ix] = scores[j]
        if (i + BS) % 50000 < BS:
            log.info(f"    {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
log.info(f"    {np.sum(~np.isnan(p_concise))}/{len(prot_ids)}")
del concise_model; torch.cuda.empty_cache()

# ConciseAnchor
log.info("  ConciseAnchor predictions...")
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear
class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)
    def forward(self, fp, anc, qry): return self.backbone(fp, anc, qry)

anchor_model = ConciseAnchorBinary().to(DEVICE)
ckpt = torch.load("models/concise_anchor_moodeng/best_model.pt", map_location=DEVICE, weights_only=False)
anchor_model.load_state_dict(ckpt["model_state_dict"]); anchor_model.eval()

p_anchor = np.full(len(prot_ids), np.nan)
with torch.no_grad():
    for i in range(0, len(prot_ids), BS):
        bp = prot_ids[i:i+BS]; bs = smiles_arr[i:i+BS]
        idx, fps, ancs, qrys = [], [], [], []
        for j, (pid, smi) in enumerate(zip(bp, bs)):
            if smi not in drug_to_anchors or smi not in ood_fp or pid not in raygun_embs: continue
            anc = None
            for a in drug_to_anchors[smi]:
                if a != pid and a in raygun_embs: anc = a; break
            if anc is None: continue
            idx.append(i+j); fps.append(ood_fp[smi])
            ancs.append(raygun_embs[anc]); qrys.append(raygun_embs[pid])
        if not fps: continue
        logit = anchor_model(torch.tensor(np.array(fps)).to(DEVICE),
                             torch.stack(ancs).to(DEVICE), torch.stack(qrys).to(DEVICE))
        scores = torch.sigmoid(logit).cpu().numpy()
        for j, ix in enumerate(idx): p_anchor[ix] = scores[j]
        if (i + BS) % 50000 < BS:
            log.info(f"    {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
log.info(f"    {np.sum(~np.isnan(p_anchor))}/{len(prot_ids)}")
del anchor_model; torch.cuda.empty_cache()

# Prot-kNN k=1
log.info("  Prot-kNN k=1...")
unique_prots = list(set(prot_ids))
prot_embs_q = np.array([raygun_embs[p].mean(dim=0).numpy() if isinstance(raygun_embs[p], torch.Tensor)
                         else raygun_embs[p].mean(axis=0) for p in unique_prots])
qn = prot_embs_q / (np.linalg.norm(prot_embs_q, axis=1, keepdims=True) + 1e-10)
ps = qn @ train_prot_normed.T
prot_topk = {}
for i, p in enumerate(unique_prots):
    best = np.argmax(ps[i])
    if ps[i, best] > 0: prot_topk[p] = (best, ps[i, best])
log.info(f"    Protein top-1 computed ({len(unique_prots)} proteins)")

unique_drugs = list(set(smiles_arr))
drug_fps_np = np.array([ood_fp[d] for d in unique_drugs])
drug_idx_map = {d: i for i, d in enumerate(unique_drugs)}

CHUNK = 4000
train_fps_gpu = torch.tensor(train_drug_fps, dtype=torch.float32).to(DEVICE)
train_bits_gpu = train_fps_gpu.sum(1)
ds = np.empty((len(unique_drugs), len(train_drug_ids)), dtype=np.float32)
for ci in range(0, len(unique_drugs), CHUNK):
    chunk = torch.tensor(drug_fps_np[ci:ci+CHUNK], dtype=torch.float32).to(DEVICE)
    inter = chunk @ train_fps_gpu.T
    q_bits = chunk.sum(1, keepdim=True)
    tani = inter / torch.clamp(q_bits + train_bits_gpu.unsqueeze(0) - inter, min=1)
    ds[ci:ci+CHUNK] = tani.cpu().numpy()
    if (ci + CHUNK) % 50000 < CHUNK:
        log.info(f"    Drug Tanimoto: {min(ci+CHUNK, len(unique_drugs))}/{len(unique_drugs)}")
del train_fps_gpu; torch.cuda.empty_cache()
log.info(f"    Drug Tanimoto computed ({len(unique_drugs)} unique drugs)")

p_knn1 = np.full(len(prot_ids), np.nan)
for i in range(len(prot_ids)):
    pid, smi = prot_ids[i], smiles_arr[i]
    if pid not in prot_topk or smi not in drug_idx_map: continue
    best_pi, best_psim = prot_topk[pid]
    bound_mask = INT_MAT[:, best_pi] > 0
    if not bound_mask.any(): continue
    di = drug_idx_map[smi]
    max_sim = ds[di, bound_mask].max()
    if max_sim > 0:
        p_knn1[i] = max_sim * best_psim
    if (i + 1) % 100000 == 0:
        log.info(f"    Scoring: {i+1}/{len(prot_ids)}")
log.info(f"    {np.sum(~np.isnan(p_knn1))}/{len(prot_ids)}")

# ================================================================
# Results (all on same subset)
# ================================================================
methods = {"CoNCISE": p_concise, "ConciseAnchor": p_anchor, "Prot-kNN k=1": p_knn1}

# Fair: all methods have predictions
fair = ~np.isnan(p_concise) & ~np.isnan(p_anchor) & ~np.isnan(p_knn1)
n_fair = fair.sum()
log.info(f"\n  ── Results on fair subset ({n_fair} int, {len(np.unique(prot_ids[fair]))} prot) ──")
log.info(f"  {int(labels[fair].sum())} pos, {int((1-labels[fair]).sum())} neg")
log.info(f"\n  {'Method':<20s} {'AUROC':>8s} {'AUPRC':>8s}")
log.info(f"  {'-'*40}")
for mname, preds in methods.items():
    if n_fair >= 10 and len(set(labels[fair].astype(int))) > 1:
        auroc_val = roc_auc_score(labels[fair], preds[fair])
        aupr_val = average_precision_score(labels[fair], preds[fair])
        log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f}")
log.info(f"  {'-'*40}")

# Also report individual coverage
log.info(f"\n  ── Individual coverage ──")
log.info(f"  {'Method':<20s} {'AUROC':>8s} {'AUPRC':>8s} {'Coverage':>10s}")
log.info(f"  {'-'*50}")
for mname, preds in methods.items():
    valid = ~np.isnan(preds)
    if valid.sum() >= 10 and len(set(labels[valid].astype(int))) > 1:
        auroc_val = roc_auc_score(labels[valid], preds[valid])
        aupr_val = average_precision_score(labels[valid], preds[valid])
        log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f} {valid.sum():>5d}/{len(prot_ids)}")

log.info("\nDone!")
