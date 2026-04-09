"""Full evaluation: CoNCISE vs ConciseAnchor vs Prot-kNN on MooDengDB test + GLASS2.

Includes quartile analysis and 30% homolog filtering for GLASS2.
All predictions batched for speed.
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
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load all data
# ================================================================
log.info("Loading data...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
test_raw = pd.read_csv(MOODENG / "test.csv", low_memory=False)
for df in [train_raw, test_raw]:
    df.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
    df["prot_id"] = df.sequence.apply(seq_to_id)

train_seqs = {seq_to_id(s): s for s in train_raw.sequence.unique()}

raygun_embs = torch.load(RESULTS / "raygun_moodeng_embeddings.pt", map_location="cpu", weights_only=False)
fp_dict = pickle.load(open(RESULTS / "morgan_moodeng_fp.pkl", "rb"))
log.info(f"Raygun: {len(raygun_embs)}, FPs: {len(fp_dict)}")

# Filter
test_df = test_raw.loc[test_raw.prot_id.isin(raygun_embs) & test_raw.smiles.isin(fp_dict),
                         ["prot_id", "smiles", "label"]].copy()

# Anchors
train_pos = train_raw[train_raw.label == 1]
train_pos = train_pos[train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors: drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs, Test: {len(test_df)}")

# GLASS2
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
glass_seqs = json.load(open("data/raw/glass/glass2_sequences.json"))
glass_raygun = torch.load(RESULTS / "raygun_glass_embeddings.pt", map_location="cpu", weights_only=False)

# GLASS2 FPs
from rdkit import Chem
from rdkit.Chem import AllChem
glass_fp = {}
for smi in glass.ligand_smiles.unique():
    if smi in fp_dict: glass_fp[smi] = fp_dict[smi]
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol: glass_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
glass = glass[glass.uniprot_id.isin(glass_raygun) & glass.ligand_smiles.isin(glass_fp)].copy()
log.info(f"GLASS2: {len(glass)} int, {glass.uniprot_id.nunique()} prot")

# 30% homolog filtering for GLASS2
log.info("30% homolog filtering GLASS2 vs MooDengDB train...")
train_kmer = {}
for pid, seq in train_seqs.items():
    s = seq.upper()
    train_kmer[pid] = set(s[i:i+3] for i in range(len(s)-2))
glass_kmer = {}
for uid, seq in glass_seqs.items():
    s = seq.upper()
    glass_kmer[uid] = set(s[i:i+3] for i in range(len(s)-2))

def check_homolog(g_uid):
    if g_uid not in glass_kmer: return None
    gk = glass_kmer[g_uid]
    if not gk: return None
    for tk in train_kmer.values():
        if not tk: continue
        if len(gk & tk) / len(gk | tk) >= 0.30: return g_uid
    return None

with Pool(min(cpu_count(), 32)) as pool:
    results = pool.map(check_homolog, list(glass.uniprot_id.unique()))
homologs = {r for r in results if r is not None}
novel_prots = set(glass.uniprot_id.unique()) - homologs
glass_novel = glass[glass.uniprot_id.isin(novel_prots)].copy()
log.info(f"  Homologs: {len(homologs)}, Novel <30%: {len(novel_prots)} prot, {len(glass_novel)} int")

del train_raw, train_seqs, train_kmer, glass_kmer
import gc; gc.collect()

# ================================================================
# Metrics
# ================================================================
def pp_ci(uids, yt_all, yp_all):
    """Per-protein CI. Returns dict uid -> ci."""
    results = {}
    for uid in np.unique(uids):
        mask = uids == uid
        yt, yp = yt_all[mask], yp_all[mask]
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

# ================================================================
# Build Prot-kNN retrieval pool from MooDengDB training positives
# ================================================================
log.info("Building Prot-kNN retrieval pool from MooDengDB train positives...")
# For binary DTI: retrieval pool = all positive (binding) interactions
# For each test protein, find nearest train proteins, predict binding score

# Train protein embeddings (pooled Raygun)
train_prot_ids = sorted(set(train_pos.prot_id) & set(raygun_embs.keys()))
train_prot_embs = np.array([raygun_embs[p].mean(dim=0).numpy() for p in train_prot_ids])
train_prot_norms = np.linalg.norm(train_prot_embs, axis=1, keepdims=True) + 1e-10
train_prot_normed = train_prot_embs / train_prot_norms
train_prot_idx = {p: i for i, p in enumerate(train_prot_ids)}

# Train drug FPs
train_drug_ids = sorted(set(train_pos.smiles) & set(fp_dict.keys()))
train_drug_fps = np.array([fp_dict[d] for d in train_drug_ids])
train_drug_idx = {d: i for i, d in enumerate(train_drug_ids)}

# Interaction matrix (drugs x proteins) — positive interactions only
log.info("  Building interaction matrix...")
INT_MAT = np.zeros((len(train_drug_ids), len(train_prot_ids)), dtype=np.float32)
for _, r in train_pos.iterrows():
    di = train_drug_idx.get(r.smiles, -1)
    pi = train_prot_idx.get(r.prot_id, -1)
    if di >= 0 and pi >= 0: INT_MAT[di, pi] = 1.0
log.info(f"  Interaction matrix: {INT_MAT.shape}, nnz={np.count_nonzero(INT_MAT)}")

def run_prot_knn(prot_ids, smiles_list, prot_emb_dict, drug_fp_dict, k):
    """GPU-batched Prot-kNN: find k nearest train proteins, predict drug binding."""
    preds = np.full(len(prot_ids), np.nan)

    # 1. Protein similarities (small, CPU is fine)
    unique_prots = list(set(prot_ids))
    prot_embs = np.array([prot_emb_dict[p].mean(dim=0).numpy() if isinstance(prot_emb_dict[p], torch.Tensor)
                           else prot_emb_dict[p].mean(axis=0) for p in unique_prots])
    qn = prot_embs / (np.linalg.norm(prot_embs, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_prot_normed.T  # (n_unique_prot, n_train_prot)
    # Pre-compute top-k for each unique protein
    prot_topk = {}
    for i, p in enumerate(unique_prots):
        topk_idx = np.argsort(ps[i])[-k:][::-1]
        topk_sims = ps[i, topk_idx]
        valid = topk_sims > 0
        prot_topk[p] = (topk_idx[valid], topk_sims[valid])
    log.info(f"    Protein top-k computed ({len(unique_prots)} unique proteins)")

    # 2. Drug Tanimoto on GPU (chunked to avoid OOM)
    unique_drugs = list(set(smiles_list))
    drug_fps_np = np.array([drug_fp_dict[d] for d in unique_drugs])
    drug_idx_map = {d: i for i, d in enumerate(unique_drugs)}

    CHUNK = 4000
    train_fps_gpu = torch.tensor(train_drug_fps, dtype=torch.float32).to(DEVICE)
    train_bits_gpu = train_fps_gpu.sum(1)  # (n_train_drugs,)
    ds = np.empty((len(unique_drugs), len(train_drug_ids)), dtype=np.float32)
    for ci in range(0, len(unique_drugs), CHUNK):
        chunk = torch.tensor(drug_fps_np[ci:ci+CHUNK], dtype=torch.float32).to(DEVICE)
        inter = chunk @ train_fps_gpu.T
        q_bits = chunk.sum(1, keepdim=True)
        tani = inter / torch.clamp(q_bits + train_bits_gpu.unsqueeze(0) - inter, min=1)
        ds[ci:ci+CHUNK] = tani.cpu().numpy()
        if (ci + CHUNK) % 20000 < CHUNK:
            log.info(f"    Drug Tanimoto: {min(ci+CHUNK, len(unique_drugs))}/{len(unique_drugs)}")
    del train_fps_gpu; torch.cuda.empty_cache()
    log.info(f"    Drug Tanimoto computed ({len(unique_drugs)} unique drugs)")

    # 3. Per-interaction scoring
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
        if (i + 1) % 50000 == 0:
            log.info(f"    Scoring: {i+1}/{len(prot_ids)}")
    return preds

# ================================================================
# Predict on MooDengDB test + GLASS2
# ================================================================
BS = 256

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
    return preds

# Load models
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
# Evaluate function
# ================================================================
def evaluate(name, prot_ids, smiles_arr, labels_or_pki, emb_dict, fp_local, is_binary=True):
    log.info(f"\n{'='*60}")
    log.info(f"  {name}: {len(prot_ids)} interactions")
    log.info(f"{'='*60}")

    # Predictions
    log.info("  CoNCISE predictions...")
    p_concise = predict_concise(prot_ids, smiles_arr, emb_dict, fp_local, concise_model)
    log.info(f"    {np.sum(~np.isnan(p_concise))}/{len(prot_ids)}")

    log.info("  ConciseAnchor predictions...")
    p_anchor = predict_anchor(prot_ids, smiles_arr, emb_dict, fp_local, anchor_model, drug_to_anchors)
    log.info(f"    {np.sum(~np.isnan(p_anchor))}/{len(prot_ids)}")

    log.info("  Prot-kNN k=1...")
    p_knn1 = run_prot_knn(prot_ids, smiles_arr, emb_dict, fp_local, 1)
    log.info(f"    {np.sum(~np.isnan(p_knn1))}/{len(prot_ids)}")

    log.info("  Prot-kNN k=5...")
    p_knn5 = run_prot_knn(prot_ids, smiles_arr, emb_dict, fp_local, 5)
    log.info(f"    {np.sum(~np.isnan(p_knn5))}/{len(prot_ids)}")

    methods = {"CoNCISE": p_concise, "ConciseAnchor": p_anchor,
               "Prot-kNN k=1": p_knn1, "Prot-kNN k=5": p_knn5}

    # For GLASS2 (continuous pKi): binarize at pKi >= 7 (pos) / pKi < 7 (neg)
    if is_binary:
        labels = labels_or_pki.astype(float)
    else:
        pki = labels_or_pki
        labels = (pki >= 7.0).astype(float)
        n_pos = labels.sum().astype(int)
        log.info(f"  Binary: {n_pos} pos (pKi>=7), {len(labels)-n_pos} neg (pKi<7)")

    log.info(f"\n  {'Method':<20s} {'AUROC':>8s} {'AUPR':>8s} {'CI':>8s} {'Coverage':>10s}")
    log.info(f"  {'-'*60}")
    for mname, preds in methods.items():
        valid = ~np.isnan(preds)
        if valid.sum() < 10:
            log.info(f"  {mname:<20s}  too few predictions ({valid.sum()})"); continue
        auroc_val, aupr_val = 0, 0
        if len(set(labels[valid].astype(int))) > 1:
            auroc_val = roc_auc_score(labels[valid], preds[valid])
            aupr_val = average_precision_score(labels[valid], preds[valid])
        ci_str = "   —"
        if not is_binary:
            ci_dict = pp_ci(np.array(prot_ids)[valid], pki[valid], preds[valid])
            ci_vals = np.array(list(ci_dict.values()))
            ci_str = f"{ci_vals.mean():8.3f}"
        log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f} {ci_str} {valid.sum():>5d}/{len(prot_ids)}")
    log.info(f"  {'-'*60}")

    # ── Fair subset: only interactions where ALL methods have predictions ──
    fair = ~np.isnan(p_concise) & ~np.isnan(p_anchor) & ~np.isnan(p_knn1) & ~np.isnan(p_knn5)
    n_fair = fair.sum()
    if n_fair >= 10:
        n_prot_fair = len(np.unique(np.array(prot_ids)[fair]))
        log.info(f"\n  ── Fair Subset ({n_fair} int, {n_prot_fair} prot) ──")
        log.info(f"  {'Method':<20s} {'AUROC':>8s} {'AUPR':>8s} {'CI':>8s}")
        log.info(f"  {'-'*50}")
        for mname, preds in methods.items():
            auroc_val, aupr_val = 0, 0
            if len(set(labels[fair].astype(int))) > 1:
                auroc_val = roc_auc_score(labels[fair], preds[fair])
                aupr_val = average_precision_score(labels[fair], preds[fair])
            ci_str = "   —"
            if not is_binary:
                ci_dict = pp_ci(np.array(prot_ids)[fair], pki[fair], preds[fair])
                ci_vals = np.array(list(ci_dict.values()))
                ci_str = f"{ci_vals.mean():8.3f}"
            log.info(f"  {mname:<20s} {auroc_val:8.4f} {aupr_val:8.4f} {ci_str}")
        log.info(f"  {'-'*50}")

        # Fair subset quartile analysis
        if not is_binary:
            log.info(f"\n  ── Fair Subset pKi Quartile Breakdown ──")
            fair_pki = pki[fair]
            q_edges_f = np.quantile(fair_pki, [0, 0.25, 0.5, 0.75, 1.0])
            q_labels_f = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
            q_assign_f = np.digitize(fair_pki, q_edges_f[1:-1])
            fair_prot_ids = np.array(prot_ids)[fair]
            fair_labels = labels[fair]
            for mname, preds in methods.items():
                log.info(f"  {mname}:")
                fair_preds = preds[fair]
                for qi, ql in enumerate(q_labels_f):
                    qm = q_assign_f == qi
                    if qm.sum() < 5: continue
                    ci_dict = pp_ci(fair_prot_ids[qm], fair_pki[qm], fair_preds[qm])
                    ci_vals = np.array(list(ci_dict.values()))
                    q_auroc = 0
                    if qm.sum() >= 5 and len(set(fair_labels[qm].astype(int))) > 1:
                        q_auroc = roc_auc_score(fair_labels[qm], fair_preds[qm])
                    log.info(f"    {ql:<15s} CI={ci_vals.mean():.3f} AUROC={q_auroc:.3f} (n={qm.sum()})")
    else:
        log.info(f"\n  ── Fair Subset: too few anchor predictions ({(~np.isnan(p_anchor)).sum()}) ──")

    # Quartile analysis on full data (pKi quartile for GLASS2)
    if not is_binary:
        log.info(f"\n  ── Full Data pKi Quartile Breakdown ──")
        q_edges = np.quantile(pki, [0, 0.25, 0.5, 0.75, 1.0])
        q_labels = ["Q1 weakest", "Q2", "Q3", "Q4 strongest"]
        q_assign = np.digitize(pki, q_edges[1:-1])
        for mname, preds in methods.items():
            valid = ~np.isnan(preds)
            if valid.sum() < 20: continue
            log.info(f"  {mname}:")
            for qi, ql in enumerate(q_labels):
                qm = (q_assign == qi) & valid
                if qm.sum() < 5: continue
                ci_dict = pp_ci(np.array(prot_ids)[qm], pki[qm], preds[qm])
                ci_vals = np.array(list(ci_dict.values()))
                q_auroc = 0
                if qm.sum() >= 5 and len(set(labels[qm].astype(int))) > 1:
                    q_auroc = roc_auc_score(labels[qm], preds[qm])
                log.info(f"    {ql:<15s} CI={ci_vals.mean():.3f} AUROC={q_auroc:.3f} (n={qm.sum()})")

# ================================================================
# Run evaluations
# ================================================================

# MooDengDB test (binary)
log.info("="*60)
log.info("MooDengDB TEST SET")
evaluate("MooDengDB Test", test_df.prot_id.values, test_df.smiles.values,
         test_df.label.values, raygun_embs, fp_dict, is_binary=True)

# GLASS2 all (continuous pKi)
evaluate("GLASS2 All", glass.uniprot_id.values, glass.ligand_smiles.values,
         glass.pki.values, glass_raygun, glass_fp, is_binary=False)

# GLASS2 novel <30%
evaluate("GLASS2 Novel <30%", glass_novel.uniprot_id.values, glass_novel.ligand_smiles.values,
         glass_novel.pki.values, glass_raygun, glass_fp, is_binary=False)

del concise_model, anchor_model; torch.cuda.empty_cache()
log.info("\nAll done!")
