"""Joint drug+protein kNN: DTC as source → BDB as target.

Excludes exact protein matches and homologs (<50% k-mer identity).
Excludes exact canonical drug matches.
Uses ESM-2 650M (1280-dim) for protein cosine, Morgan FP Tanimoto for drugs.
"""
import os, sys, pickle, logging, time
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

N_WORKERS = min(cpu_count(), 32)

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

def pp_rmse(df, col):
    out = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt, yp = s.pki.values, np.array(s[col].values, dtype=float)
        m = ~np.isnan(yp); yt, yp = yt[m], yp[m]
        if len(yt) < 3: continue
        out.append(np.sqrt(mean_squared_error(yt, yp)))
    return np.array(out)


# ── load DTC (source) ────────────────────────────────────────────
log.info("Loading DTC...")
dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
esm_dtc = torch.load("data/processed/esm2_650m_dtc.pt", map_location="cpu", weights_only=False)
esm_dtc_np = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in esm_dtc.items()}

dtc = dtc[dtc.uniprot_id.isin(esm_dtc_np)].copy()
log.info(f"DTC: {len(dtc)} int, {dtc.uniprot_id.nunique()} prot, {dtc.ligand_smiles.nunique()} drugs")

# Build Morgan FPs for DTC drugs
log.info("Computing Morgan FPs for DTC drugs...")
from rdkit import Chem
from rdkit.Chem import AllChem

dtc_fp = {}
for smi in dtc.ligand_smiles.unique():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        dtc_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
dtc = dtc[dtc.ligand_smiles.isin(dtc_fp)].copy()
log.info(f"DTC with FPs: {len(dtc)} int, {dtc.ligand_smiles.nunique()} drugs")

# Index DTC
dtc_drugs = sorted(set(dtc.ligand_smiles))
dtc_drug_idx = {d: i for i, d in enumerate(dtc_drugs)}
dtc_fp_mat = np.array([dtc_fp[d] for d in dtc_drugs])

dtc_prots = sorted(set(dtc.uniprot_id))
dtc_prot_idx = {p: i for i, p in enumerate(dtc_prots)}
dtc_emb_mat = np.array([esm_dtc_np[p] for p in dtc_prots])
dtc_emb_norms = np.linalg.norm(dtc_emb_mat, axis=1, keepdims=True) + 1e-10
dtc_emb_normed = dtc_emb_mat / dtc_emb_norms

d2i = defaultdict(list)  # drug_idx -> [(prot_idx, pki)]
p2i = defaultdict(list)  # prot_idx -> [(drug_idx, pki)]
for _, r in dtc.iterrows():
    di, pi = dtc_drug_idx[r.ligand_smiles], dtc_prot_idx[r.uniprot_id]
    d2i[di].append((pi, r.pki))
    p2i[pi].append((di, r.pki))
log.info(f"DTC indexed: FP {dtc_fp_mat.shape}, Emb {dtc_emb_mat.shape}")


# ── load BDB (target) ───────────────────────────────────────────
log.info("Loading BDB...")
bdb = pd.read_csv("data/processed/bindingdb_interactions.csv")
log.info(f"BDB raw: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot")

# BDB protein embeddings — use ESM DTC embeddings where available
bdb = bdb[bdb.uniprot_id.isin(esm_dtc_np)].copy()
log.info(f"BDB with ESM embeddings: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot")

# BDB drug FPs
bdb_fp = {}
for smi in bdb.ligand_smiles.unique():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        bdb_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
bdb = bdb[bdb.ligand_smiles.isin(bdb_fp)].copy()
log.info(f"BDB with FPs: {len(bdb)} int, {bdb.ligand_smiles.nunique()} drugs")

# ── Overlap exclusion ────────────────────────────────────────────
# 1. Exact protein overlap
prot_overlap = set(dtc.uniprot_id) & set(bdb.uniprot_id)
log.info(f"Protein overlap (exact): {len(prot_overlap)}")

# 2. Homolog filtering — compute k-mer identity
log.info("Computing k-mer identity for homolog filtering...")
seqs = {}
if os.path.exists("data/processed/merged_sequences.json"):
    import json
    seqs = json.load(open("data/processed/merged_sequences.json"))

def kmer_identity(s1, s2, k=3):
    if not s1 or not s2: return 0.0
    k1 = set(s1[i:i+k] for i in range(len(s1)-k+1))
    k2 = set(s2[i:i+k] for i in range(len(s2)-k+1))
    if not k1 or not k2: return 0.0
    return len(k1 & k2) / len(k1 | k2)

dtc_seqs = {p: seqs[p] for p in dtc.uniprot_id.unique() if p in seqs}
# Precompute DTC k-mer sets once
dtc_kmer_sets = {p: set(s[i:i+3] for i in range(len(s)-2)) for p, s in dtc_seqs.items()}

def _check_homolog(bp):
    """Check if a BDB protein is a homolog of any DTC protein."""
    if bp not in seqs: return None
    bs = seqs[bp]
    bk = set(bs[i:i+3] for i in range(len(bs)-2))
    if not bk: return None
    for dk in dtc_kmer_sets.values():
        if len(bk & dk) / len(bk | dk) >= 0.5:
            return bp
    return None

bdb_eval_prots = sorted(set(bdb.uniprot_id.unique()) - prot_overlap)
log.info(f"Checking {len(bdb_eval_prots)} BDB proteins for homologs ({N_WORKERS} workers)...")
with Pool(N_WORKERS) as pool:
    results = pool.map(_check_homolog, bdb_eval_prots)
homolog_prots = {r for r in results if r is not None}
log.info(f"Found {len(homolog_prots)} homologs")

all_excl_prots = prot_overlap | homolog_prots
log.info(f"Total excluded proteins: {len(all_excl_prots)} ({len(prot_overlap)} exact + {len(homolog_prots)} homologs)")

# 3. Drug overlap — raw string match (fast, conservative)
dtc_drug_set = set(dtc.ligand_smiles.unique())
excl_bdb_drugs = set(bdb.ligand_smiles.unique()) & dtc_drug_set
all_excl_drugs = excl_bdb_drugs
log.info(f"Drug overlap (raw string): {len(excl_bdb_drugs)}/{bdb.ligand_smiles.nunique()} BDB drugs")

# Filter BDB
bdb_eval = bdb[~bdb.uniprot_id.isin(all_excl_prots) & ~bdb.ligand_smiles.isin(excl_bdb_drugs)].copy()
log.info(f"BDB eval (after exclusion): {len(bdb_eval)} int, "
         f"{bdb_eval.uniprot_id.nunique()} prot, {bdb_eval.ligand_smiles.nunique()} drugs")


# ── Similarity ───────────────────────────────────────────────────

def drug_sims(query_fps):
    inter = query_fps @ dtc_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = dtc_fp_mat.sum(1, keepdims=True).T
    return inter / np.maximum(q_bits + db_bits - inter, 1)

def prot_sims(query_embs):
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    return qn @ dtc_emb_normed.T


# ── kNN methods ──────────────────────────────────────────────────

def _joint_chunk(args):
    """Joint drug+protein kNN: find k nearest DTC drugs, for each get pKi
    with closest DTC protein to query."""
    chunk, ds, ps, dm, pm, d2i_d, k = args
    preds = []
    for smi, uid in chunk:
        di, pi = dm.get(smi, -1), pm.get(uid, -1)
        if di < 0 or pi < 0: preds.append(np.nan); continue
        top = np.argsort(ds[di])[-k:][::-1]
        vals, wts = [], []
        for bdi in top:
            if ds[di, bdi] <= 0: continue
            best_pki, best_ps = None, -1
            for bpi, pki in d2i_d.get(bdi, []):
                if ps[pi, bpi] > best_ps:
                    best_ps = ps[pi, bpi]; best_pki = pki
            if best_pki is not None:
                vals.append(best_pki); wts.append(float(ds[di, bdi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def _prot_only_chunk(args):
    """Pure protein kNN: k nearest DTC proteins, avg their mean pKi."""
    chunk, ds, ps, dm, pm, prot_mean, k = args
    preds = []
    for smi, uid in chunk:
        pi = pm.get(uid, -1)
        if pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-k:][::-1]
        vals, wts = [], []
        for bpi in top:
            if ps[pi, bpi] <= 0: continue
            if bpi in prot_mean:
                vals.append(prot_mean[bpi]); wts.append(float(ps[pi, bpi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def par_run(fn, rows, ds, ps, dm, pm, lookup, k):
    sz = max(1, len(rows) // N_WORKERS)
    chunks = [(rows[i:i+sz], ds, ps, dm, pm, lookup, k) for i in range(0, len(rows), sz)]
    with Pool(N_WORKERS) as pool:
        res = pool.map(fn, chunks)
    return [p for r in res for p in r]


# ── Evaluate ─────────────────────────────────────────────────────

if len(bdb_eval) < 10:
    log.info("Too few interactions after filtering!")
    sys.exit(1)

ud = sorted(bdb_eval.ligand_smiles.unique())
up = sorted(bdb_eval.uniprot_id.unique())
dm = {d: i for i, d in enumerate(ud)}
pm = {p: i for i, p in enumerate(up)}

qfp = np.array([bdb_fp[d] for d in ud])
qemb = np.array([esm_dtc_np[p] for p in up])

t0 = time.time()
ds = drug_sims(qfp)
ps = prot_sims(qemb)

# Exclude overlapping drugs from similarity
for d in all_excl_drugs:
    if d in dtc_drug_idx:
        ds[:, dtc_drug_idx[d]] = -1
# Exclude overlapping/homolog proteins
for p in all_excl_prots:
    if p in dtc_prot_idx:
        ps[:, dtc_prot_idx[p]] = -1

log.info(f"Sims: {time.time()-t0:.1f}s (ds {ds.shape}, ps {ps.shape})")

rows = list(zip(bdb_eval.ligand_smiles.values, bdb_eval.uniprot_id.values))
methods = {}

# Joint drug+protein kNN
for k in [1, 5, 10]:
    log.info(f"  joint_knn k={k} ({N_WORKERS} workers)...")
    methods[f'joint_knn_k{k}'] = par_run(_joint_chunk, rows, ds, ps, dm, pm, dict(d2i), k)

# Pure protein kNN
prot_mean = {pi: np.mean(pkis) for pi, pkis in
             {i: [pki for _, pki in p2i[i]] for i in range(len(dtc_prots))}.items()}
for k in [1, 5]:
    log.info(f"  prot_only_knn k={k}...")
    methods[f'prot_only_k{k}'] = par_run(_prot_only_chunk, rows, ds, ps, dm, pm, prot_mean, k)

log.info(f"Done: {time.time()-t0:.0f}s")

for m, p in methods.items():
    bdb_eval[m] = p

log.info(f"\n{'='*60}")
log.info(f"  DTC → BDB ({bdb_eval.uniprot_id.nunique()} novel proteins, "
         f"{bdb_eval.ligand_smiles.nunique()} novel drugs)")
log.info(f"  {'Method':<22s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s} {'NaN%':>5s}")
log.info(f"  {'-'*47}")
for m in methods:
    p = np.array(methods[m], dtype=float)
    ci = pp_ci(bdb_eval, m); auc = pp_auroc(bdb_eval, m); rmse = pp_rmse(bdb_eval, m)
    log.info(f"  {m:<22s} {ci.mean():6.3f} "
             f"{auc.mean() if len(auc) else 0:6.3f} "
             f"{rmse.mean() if len(rmse) else 0:6.3f} "
             f"{np.isnan(p).mean()*100:5.1f}")

bdb_eval.to_csv("results/knn_dtc_to_bdb.csv", index=False)
log.info(f"Saved results/knn_dtc_to_bdb.csv")
log.info("DONE")
