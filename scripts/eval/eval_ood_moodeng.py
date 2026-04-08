"""Evaluate CoNCISE + ConciseAnchor on MooDengDB v2 OOD test set (LCIdb holdout).

OOD = out-of-distribution ligands not in MooDengDB training.
Binary classification: AUROC, AUPRC, fair subset comparison.
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
BS = 256

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load MooDengDB training data (for anchors + kNN)
# ================================================================
log.info("Loading MooDengDB training data...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
train_raw.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
train_raw["prot_id"] = train_raw.sequence.apply(seq_to_id)

raygun_embs = torch.load(RESULTS / "raygun_moodeng_embeddings.pt", map_location="cpu", weights_only=False)
fp_dict = pickle.load(open(RESULTS / "morgan_moodeng_fp.pkl", "rb"))
log.info(f"Raygun: {len(raygun_embs)}, FPs: {len(fp_dict)}")

# Anchors from training positives
train_pos = train_raw[train_raw.label == 1]
train_pos = train_pos[train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors: drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs with positive binders")

# ================================================================
# Load OOD test set
# ================================================================
log.info("Loading OOD test set (MooDengDB v2 extended)...")
ood_raw = pd.read_csv("data/moodeng-v2-extended/test.csv", sep="\t", low_memory=False)
ood_raw.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
ood_raw["prot_id"] = ood_raw.sequence.apply(seq_to_id)
log.info(f"OOD test: {len(ood_raw)} int, {ood_raw.prot_id.nunique()} prot, "
         f"{ood_raw.smiles.nunique()} drugs, "
         f"{(ood_raw.label==1).sum()} pos, {(ood_raw.label==0).sum()} neg")

# Check SMILES overlap with training
train_smiles = set(train_raw.smiles.unique())
ood_smiles = set(ood_raw.smiles.unique())
overlap = ood_smiles & train_smiles
log.info(f"OOD SMILES overlap with train: {len(overlap)}/{len(ood_smiles)} "
         f"({100*len(overlap)/len(ood_smiles):.1f}%)")

# ================================================================
# Compute OOD Raygun embeddings (reuse cached where possible)
# ================================================================
ood_prots = {seq_to_id(s): s for s in ood_raw.sequence.unique()}
missing_prots = {pid: seq for pid, seq in ood_prots.items() if pid not in raygun_embs}
log.info(f"OOD proteins: {len(ood_prots)} total, {len(missing_prots)} need Raygun embeddings")

OOD_RAYGUN_CACHE = RESULTS / "raygun_ood_embeddings.pt"
if missing_prots:
    if OOD_RAYGUN_CACHE.exists():
        ood_raygun = torch.load(str(OOD_RAYGUN_CACHE), map_location="cpu", weights_only=False)
        log.info(f"Loaded OOD Raygun cache: {len(ood_raygun)} proteins")
        still_missing = {pid: seq for pid, seq in missing_prots.items() if pid not in ood_raygun}
    else:
        ood_raygun = {}
        still_missing = missing_prots

    if still_missing:
        log.info(f"Computing Raygun for {len(still_missing)} OOD proteins...")
        import esm
        esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = alphabet.get_batch_converter()
        esm_model = esm_model.eval().to(DEVICE)
        raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
        raygun_model = raygun_model.eval().to(DEVICE)

        items = [(pid, seq[:1022].upper()) for pid, seq in still_missing.items() if len(seq) >= 25]
        for idx, (pid, seq) in enumerate(items):
            try:
                _, _, toks = bc([(pid, seq)])
                with torch.no_grad():
                    out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
                    esm_emb = out["representations"][33][0, 1:len(seq)+1, :]
                    ray_out = raygun_model(esm_emb.unsqueeze(0))
                    ood_raygun[pid] = ray_out[1].squeeze(0).cpu()
                del out, esm_emb, ray_out, toks
            except Exception as e:
                pass
            if (idx+1) % 100 == 0:
                log.info(f"  {idx+1}/{len(items)} done")
                torch.save(ood_raygun, str(OOD_RAYGUN_CACHE))
        del esm_model, raygun_model; torch.cuda.empty_cache()
        torch.save(ood_raygun, str(OOD_RAYGUN_CACHE))
        log.info(f"Saved {len(ood_raygun)} OOD Raygun embeddings")

    # Merge into main dict
    for pid, emb in ood_raygun.items():
        raygun_embs[pid] = emb

# Compute OOD drug FPs
log.info("OOD drug FPs...")
from rdkit import Chem
from rdkit.Chem import AllChem
ood_fp = {}
for smi in ood_raw.smiles.unique():
    if smi in fp_dict:
        ood_fp[smi] = fp_dict[smi]
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                ood_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
log.info(f"  {len(ood_fp)}/{ood_raw.smiles.nunique()} drugs")

# Filter OOD to those with embeddings + FPs
ood_df = ood_raw[ood_raw.prot_id.isin(raygun_embs) & ood_raw.smiles.isin(ood_fp)].copy()
log.info(f"OOD filtered: {len(ood_df)} int, {ood_df.prot_id.nunique()} prot, "
         f"{ood_df.smiles.nunique()} drugs")

del train_raw, train_pos, ood_raw
import gc; gc.collect()

# ================================================================
# Build Prot-kNN retrieval pool
# ================================================================
log.info("Building Prot-kNN retrieval pool...")
train_raw2 = pd.read_csv(MOODENG / "train.csv", low_memory=False)
train_raw2.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
train_raw2["prot_id"] = train_raw2.sequence.apply(seq_to_id)
train_pos2 = train_raw2[train_raw2.label == 1]
train_pos2 = train_pos2[train_pos2.prot_id.isin(raygun_embs) & train_pos2.smiles.isin(fp_dict)]

train_prot_ids = sorted(set(train_pos2.prot_id) & set(raygun_embs.keys()))
train_prot_embs = np.array([raygun_embs[p].mean(dim=0).numpy() for p in train_prot_ids])
train_prot_norms = np.linalg.norm(train_prot_embs, axis=1, keepdims=True) + 1e-10
train_prot_normed = train_prot_embs / train_prot_norms

train_drug_ids = sorted(set(train_pos2.smiles) & set(fp_dict.keys()))
train_drug_fps = np.array([fp_dict[d] for d in train_drug_ids])
train_drug_idx = {d: i for i, d in enumerate(train_drug_ids)}

INT_MAT = np.zeros((len(train_drug_ids), len(train_prot_ids)), dtype=np.float32)
train_prot_idx = {p: i for i, p in enumerate(train_prot_ids)}
for _, r in train_pos2.iterrows():
    di = train_drug_idx.get(r.smiles, -1)
    pi = train_prot_idx.get(r.prot_id, -1)
    if di >= 0 and pi >= 0: INT_MAT[di, pi] = 1.0
log.info(f"  Interaction matrix: {INT_MAT.shape}, nnz={np.count_nonzero(INT_MAT)}")
del train_raw2, train_pos2; gc.collect()

# ================================================================
# Prediction functions
# ================================================================
def predict_concise(prot_ids, smiles_arr, emb_dict, fp_dict_local, model):
    preds = np.full(len(prot_ids), np.nan)
    for i in range(0, len(prot_ids), BS):
        batch_pids = prot_ids[i:i+BS]
        batch_smis = smiles_arr[i:i+BS]
        idx, fps, embs = [], [], []
        for j, (pid, smi) in enumerate(zip(batch_pids, batch_smis)):
            if pid in emb_dict and smi in fp_dict_local:
                idx.append(i+j); fps.append(fp_dict_local[smi]); embs.append(emb_dict[pid])
        if not fps: continue
        with torch.no_grad():
            pred = model(torch.tensor(np.array(fps)).to(DEVICE),
                         torch.stack(embs).to(DEVICE), is_morgan_fingerprint=True)["binding"]
            scores = ((pred + 1) / 2).cpu().numpy()
        for j, ix in enumerate(idx): preds[ix] = scores[j]
        if (i + BS) % 50000 < BS:
            log.info(f"    {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
    return preds

def predict_anchor(prot_ids, smiles_arr, emb_dict, fp_dict_local, model, anchors):
    preds = np.full(len(prot_ids), np.nan)
    for i in range(0, len(prot_ids), BS):
        batch_pids = prot_ids[i:i+BS]
        batch_smis = smiles_arr[i:i+BS]
        idx, fps, ancs, qrys = [], [], [], []
        for j, (pid, smi) in enumerate(zip(batch_pids, batch_smis)):
            if smi not in anchors or smi not in fp_dict_local or pid not in emb_dict: continue
            anc = None
            for a in anchors[smi]:
                if a != pid: anc = a; break
            if anc is None or anc not in raygun_embs: continue
            idx.append(i+j); fps.append(fp_dict_local[smi])
            ancs.append(raygun_embs[anc]); qrys.append(emb_dict[pid])
        if not fps: continue
        with torch.no_grad():
            logit = model(torch.tensor(np.array(fps)).to(DEVICE),
                          torch.stack(ancs).to(DEVICE), torch.stack(qrys).to(DEVICE))
            scores = torch.sigmoid(logit).cpu().numpy()
        for j, ix in enumerate(idx): preds[ix] = scores[j]
        if (i + BS) % 50000 < BS:
            log.info(f"    {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
    return preds

def run_prot_knn(prot_ids, smiles_list, prot_emb_dict, drug_fp_dict, k):
    preds = np.full(len(prot_ids), np.nan)
    unique_prots = list(set(prot_ids))
    prot_embs = np.array([prot_emb_dict[p].mean(dim=0).numpy() if isinstance(prot_emb_dict[p], torch.Tensor)
                           else prot_emb_dict[p].mean(axis=0) for p in unique_prots])
    qn = prot_embs / (np.linalg.norm(prot_embs, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_prot_normed.T
    prot_topk = {}
    for i, p in enumerate(unique_prots):
        topk_idx = np.argsort(ps[i])[-k:][::-1]
        topk_sims = ps[i, topk_idx]
        valid = topk_sims > 0
        prot_topk[p] = (topk_idx[valid], topk_sims[valid])
    log.info(f"    Protein top-k computed ({len(unique_prots)} unique proteins)")

    unique_drugs = list(set(smiles_list))
    drug_fps_np = np.array([drug_fp_dict[d] for d in unique_drugs])
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

    for i in range(len(prot_ids)):
        pid, smi = prot_ids[i], smiles_list[i]
        if pid not in prot_topk or smi not in drug_idx_map: continue
        topk_idx, topk_sims = prot_topk[pid]
        if len(topk_idx) == 0: continue
        di = drug_idx_map[smi]
        d_sims = ds[di]
        best_scores, best_wts = [], []
        for ki in range(len(topk_idx)):
            bpi = topk_idx[ki]
            bound_mask = INT_MAT[:, bpi] > 0
            if not bound_mask.any(): continue
            max_sim = d_sims[bound_mask].max()
            if max_sim <= 0: continue
            best_scores.append(max_sim)
            best_wts.append(topk_sims[ki])
        if best_scores:
            preds[i] = np.average(best_scores, weights=best_wts)
        if (i + 1) % 100000 == 0:
            log.info(f"    Scoring: {i+1}/{len(prot_ids)}")
    return preds

# ================================================================
# Load models
# ================================================================
log.info("Loading models...")
concise_model = torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v2", pretrained=True)
concise_model = concise_model.eval().to(DEVICE)

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

# ================================================================
# Run predictions
# ================================================================
prot_ids = ood_df.prot_id.values
smiles_arr = ood_df.smiles.values
labels = ood_df.label.values.astype(float)

log.info(f"\n{'='*60}")
log.info(f"  OOD Test: {len(ood_df)} interactions")
log.info(f"  {(labels==1).sum()} pos, {(labels==0).sum()} neg")
log.info(f"{'='*60}")

log.info("  CoNCISE predictions...")
p_concise = predict_concise(prot_ids, smiles_arr, raygun_embs, ood_fp, concise_model)
log.info(f"    {np.sum(~np.isnan(p_concise))}/{len(prot_ids)}")

log.info("  ConciseAnchor predictions...")
p_anchor = predict_anchor(prot_ids, smiles_arr, raygun_embs, ood_fp, anchor_model, drug_to_anchors)
log.info(f"    {np.sum(~np.isnan(p_anchor))}/{len(prot_ids)}")

log.info("  Prot-kNN k=1...")
p_knn1 = run_prot_knn(prot_ids, smiles_arr, raygun_embs, ood_fp, 1)
log.info(f"    {np.sum(~np.isnan(p_knn1))}/{len(prot_ids)}")

log.info("  Prot-kNN k=5...")
p_knn5 = run_prot_knn(prot_ids, smiles_arr, raygun_embs, ood_fp, 5)
log.info(f"    {np.sum(~np.isnan(p_knn5))}/{len(prot_ids)}")

methods = {"CoNCISE": p_concise, "ConciseAnchor": p_anchor,
           "Prot-kNN k=1": p_knn1, "Prot-kNN k=5": p_knn5}

# ================================================================
# Results
# ================================================================
log.info(f"\n  {'Method':<20s} {'AUROC':>8s} {'AUPR':>8s} {'Coverage':>10s}")
log.info(f"  {'-'*50}")
for mname, preds in methods.items():
    valid = ~np.isnan(preds)
    if valid.sum() < 10:
        log.info(f"  {mname:<20s}  too few predictions ({valid.sum()})"); continue
    auroc_val = roc_auc_score(labels[valid], preds[valid])
    aupr_val = average_precision_score(labels[valid], preds[valid])
    log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f} {valid.sum():>5d}/{len(prot_ids)}")
log.info(f"  {'-'*50}")

# Fair subset
fair = ~np.isnan(p_concise) & ~np.isnan(p_anchor) & ~np.isnan(p_knn1) & ~np.isnan(p_knn5)
n_fair = fair.sum()
if n_fair >= 10:
    n_prot = len(np.unique(prot_ids[fair]))
    log.info(f"\n  ── Fair Subset ({n_fair} int, {n_prot} prot) ──")
    log.info(f"  {'Method':<20s} {'AUROC':>8s} {'AUPR':>8s}")
    log.info(f"  {'-'*40}")
    for mname, preds in methods.items():
        auroc_val = roc_auc_score(labels[fair], preds[fair])
        aupr_val = average_precision_score(labels[fair], preds[fair])
        log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f}")
    log.info(f"  {'-'*40}")
else:
    log.info(f"\n  Fair subset too small ({n_fair})")

# Save
out_df = ood_df.copy()
out_df["concise_pred"] = p_concise
out_df["anchor_pred"] = p_anchor
out_df["knn1_pred"] = p_knn1
out_df["knn5_pred"] = p_knn5
out_df.to_csv(RESULTS / "ood_test_results.csv", index=False)
log.info(f"\nSaved results/ood_test_results.csv")

del concise_model, anchor_model; torch.cuda.empty_cache()
log.info("Done!")
