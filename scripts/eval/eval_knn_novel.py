"""kNN baselines on NOVEL proteins only (<50% k-mer identity to BDB training).

Uses pre-computed kNN predictions from eval_knn_baselines.py results,
filters to novel proteins, and recomputes per-protein metrics.
If kNN predictions don't exist yet, runs them from scratch.
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── metrics ──────────────────────────────────────────────────────

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


def report(name, df, methods):
    log.info(f"\n  {name} ({df.uniprot_id.nunique()} proteins, {len(df)} int)")
    log.info(f"  {'Method':<18s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s}")
    log.info(f"  {'-'*38}")
    for m in methods:
        if m not in df.columns: continue
        ci = pp_ci(df, m); auc = pp_auroc(df, m); rmse = pp_rmse(df, m)
        log.info(f"  {m:<18s} {ci.mean():6.3f} "
                 f"{auc.mean() if len(auc) else 0:6.3f} "
                 f"{rmse.mean() if len(rmse) else 0:6.3f}")


# ── load data ────────────────────────────────────────────────────

log.info("Loading BDB data...")
bdb = pd.read_csv("data/processed/bindingdb_interactions.csv")
morgan = pickle.load(open("data/processed/concise_bdb_morgan_fp.pkl", "rb"))
raygun_raw = torch.load("data/processed/raygun_bdb_embeddings.pt",
                         map_location="cpu", weights_only=False)
raygun_pooled = {}
for k, v in raygun_raw.items():
    t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    raygun_pooled[k] = t.mean(dim=0).numpy()
del raygun_raw

bdb = bdb[bdb.uniprot_id.isin(raygun_pooled) & bdb.ligand_smiles.isin(morgan)].copy()
log.info(f"BDB: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot, "
         f"{bdb.ligand_smiles.nunique()} drugs")

# Index structures
bdb_drugs = sorted(set(bdb.ligand_smiles))
bdb_drug_idx = {d: i for i, d in enumerate(bdb_drugs)}
bdb_fp_mat = np.array([np.array(morgan[d], dtype=np.float32) for d in bdb_drugs])

bdb_prots = sorted(set(bdb.uniprot_id))
bdb_prot_idx = {p: i for i, p in enumerate(bdb_prots)}
bdb_emb_mat = np.array([raygun_pooled[p] for p in bdb_prots])
bdb_emb_norms = np.linalg.norm(bdb_emb_mat, axis=1, keepdims=True) + 1e-10
bdb_emb_normed = bdb_emb_mat / bdb_emb_norms

d2i = defaultdict(list)
p2i = defaultdict(list)
for _, r in bdb.iterrows():
    di, pi = bdb_drug_idx[r.ligand_smiles], bdb_prot_idx[r.uniprot_id]
    d2i[di].append((pi, r.pki))
    p2i[pi].append((di, r.pki))
log.info("Lookups built")


# ── similarity ───────────────────────────────────────────────────

def drug_sims(query_fps, exclude=None):
    inter = query_fps @ bdb_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = bdb_fp_mat.sum(1, keepdims=True).T
    sims = inter / np.maximum(q_bits + db_bits - inter, 1)
    if exclude:
        for d in exclude:
            if d in bdb_drug_idx: sims[:, bdb_drug_idx[d]] = -1
    return sims

def prot_sims(query_embs, exclude=None):
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    sims = qn @ bdb_emb_normed.T
    if exclude:
        for p in exclude:
            if p in bdb_prot_idx: sims[:, bdb_prot_idx[p]] = -1
    return sims


# ── kNN methods (multiprocessing) ────────────────────────────────

N_WORKERS = min(cpu_count(), 16)

def _drug_chunk(args):
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
                if ps[pi, bpi] > best_ps: best_ps = ps[pi, bpi]; best_pki = pki
            if best_pki is not None: vals.append(best_pki); wts.append(float(ds[di, bdi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def _prot_chunk(args):
    chunk, ds, ps, dm, pm, p2i_d, k = args
    preds = []
    for smi, uid in chunk:
        di, pi = dm.get(smi, -1), pm.get(uid, -1)
        if di < 0 or pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-k:][::-1]
        vals, wts = [], []
        for bpi in top:
            if ps[pi, bpi] <= 0: continue
            best_pki, best_ds = None, -1
            for bdi, pki in p2i_d.get(bpi, []):
                if ds[di, bdi] > best_ds: best_ds = ds[di, bdi]; best_pki = pki
            if best_pki is not None: vals.append(best_pki); wts.append(float(ps[pi, bpi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def _anchor_chunk(args):
    chunk, ds, ps, dm, pm, d2i_d = args
    preds = []
    for smi, uid in chunk:
        di, pi = dm.get(smi, -1), pm.get(uid, -1)
        if di < 0 or pi < 0: preds.append(np.nan); continue
        top_drugs = np.argsort(ds[di])[-10:][::-1]
        found = False
        for bdi in top_drugs:
            if ds[di, bdi] <= 0: break
            strong = [(bpi, pki) for bpi, pki in d2i_d.get(bdi, []) if pki >= 7.0]
            if not strong: continue
            best_pki, best_ps = None, -1
            for bpi, pki in strong:
                if ps[pi, bpi] > best_ps: best_ps = ps[pi, bpi]; best_pki = pki
            if best_pki is not None: preds.append(best_pki); found = True; break
        if not found: preds.append(np.nan)
    return preds

def _par(fn, rows, ds, ps, dm, pm, lookup, extra=None):
    sz = max(1, len(rows) // N_WORKERS)
    chunks = []
    for i in range(0, len(rows), sz):
        c = rows[i:i+sz]
        chunks.append((c, ds, ps, dm, pm, lookup, extra) if extra is not None
                       else (c, ds, ps, dm, pm, lookup))
    with Pool(N_WORKERS) as pool:
        res = pool.map(fn, chunks)
    return [p for r in res for p in r]

def run_drug_knn(rows, ds, ps, dm, pm, k):
    return _par(_drug_chunk, rows, ds, ps, dm, pm, dict(d2i), k)

def run_prot_knn(rows, ds, ps, dm, pm, k):
    return _par(_prot_chunk, rows, ds, ps, dm, pm, dict(p2i), k)

def run_anchor_knn(rows, ds, ps, dm, pm):
    return _par(_anchor_chunk, rows, ds, ps, dm, pm, dict(d2i))

def run_joint_knn(rows, ds, ps, dm, pm, k, alpha=0.5):
    i_di = np.array([bdb_drug_idx[r] for r in bdb.ligand_smiles.values])
    i_pi = np.array([bdb_prot_idx[r] for r in bdb.uniprot_id.values])
    i_pki = bdb.pki.values
    preds = []
    for smi, uid in rows:
        di, pi = dm.get(smi, -1), pm.get(uid, -1)
        if di < 0 or pi < 0: preds.append(np.nan); continue
        scores = alpha * ds[di][i_di] + (1-alpha) * ps[pi][i_pi]
        topk = np.argpartition(scores, -k)[-k:]
        s, p = scores[topk], i_pki[topk]
        m = s > 0
        preds.append(np.average(p[m], weights=s[m]) if m.sum() else np.nan)
    return preds


def run_all_knn(df, emb_dict, fp_dict, excl_drugs=None, excl_prots=None):
    """Run all kNN methods on a dataframe. Returns df with prediction columns."""
    df = df[df.uniprot_id.isin(emb_dict) & df.ligand_smiles.isin(fp_dict)].copy()
    if excl_drugs: df = df[~df.ligand_smiles.isin(excl_drugs)].copy()
    if excl_prots: df = df[~df.uniprot_id.isin(excl_prots)].copy()
    if len(df) < 10: return df

    ud = sorted(df.ligand_smiles.unique())
    up = sorted(df.uniprot_id.unique())
    dm = {d: i for i, d in enumerate(ud)}
    pm = {p: i for i, p in enumerate(up)}

    qfp = np.array([np.array(fp_dict[d], dtype=np.float32) for d in ud])
    qemb = np.array([emb_dict[p] for p in up])

    ds = drug_sims(qfp, excl_drugs)
    ps = prot_sims(qemb, excl_prots)

    rows = list(zip(df.ligand_smiles.values, df.uniprot_id.values))

    for k in [1, 5]:
        log.info(f"    drug_knn k={k}...")
        df[f'drug_knn_k{k}'] = run_drug_knn(rows, ds, ps, dm, pm, k)
    for k in [1, 5]:
        log.info(f"    prot_knn k={k}...")
        df[f'prot_knn_k{k}'] = run_prot_knn(rows, ds, ps, dm, pm, k)
    log.info("    joint_knn k=10...")
    df['joint_knn_k10'] = run_joint_knn(rows, ds, ps, dm, pm, 10)
    log.info("    anchor_knn...")
    df['anchor_knn'] = run_anchor_knn(rows, ds, ps, dm, pm)
    return df


# ── canonical SMILES ─────────────────────────────────────────────

try:
    from rdkit import Chem
    def canon(s):
        try: m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else s
        except: return s
except ImportError:
    def canon(s): return s


# ── Load embeddings ──────────────────────────────────────────────

# Davis Raygun (cached from eval_knn_baselines.py)
CACHE = "results/raygun_davis_pooled.pt"
if os.path.exists(CACHE):
    davis_emb = torch.load(CACHE, map_location="cpu", weights_only=False)
    log.info(f"Loaded Davis Raygun cache: {len(davis_emb)}")
else:
    log.info("ERROR: Run eval_knn_baselines.py first to generate Davis Raygun cache")
    sys.exit(1)

all_emb = {**raygun_pooled, **davis_emb}

# FPs
all_fp = dict(morgan)
if os.path.exists("results/concise_davis_fp.pkl"):
    all_fp.update(pickle.load(open("results/concise_davis_fp.pkl", "rb")))
missing = set()
davis_raw = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis_raw.rename(columns={"protein_name": "uniprot_id",
                                   "drug_smiles": "ligand_smiles"})
for d in davis.ligand_smiles.unique():
    if d not in all_fp: missing.add(d)
if missing:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    for s in missing:
        m = Chem.MolFromSmiles(s)
        if m: all_fp[s] = np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)

all_fp_glass = dict(morgan)
if os.path.exists("results/concise_glass_fp.pkl"):
    all_fp_glass.update(pickle.load(open("results/concise_glass_fp.pkl", "rb")))


# ── Overlap sets ─────────────────────────────────────────────────

c_bdb = {d: canon(d) for d in bdb.ligand_smiles.unique()}
bdb_cs = set(c_bdb.values())

# Davis
c_davis = {d: canon(d) for d in davis.ligand_smiles.unique()}
ov_d_davis = {d for d, c in c_davis.items() if c in bdb_cs}
ov_b_davis = {d for d, c in c_bdb.items() if c in {c_davis[x] for x in ov_d_davis}}
excl_drugs_davis = ov_d_davis | ov_b_davis
excl_prots_davis = set(davis.uniprot_id) & set(bdb.uniprot_id)

# Novel Davis proteins (<50% k-mer to BDB)
davis_homologs = set()
hom_file = "results/davis_bdb_homologs_50.txt"
if os.path.exists(hom_file):
    with open(hom_file) as f:
        davis_homologs = {l.strip() for l in f if l.strip()}
davis_novel = set(davis.uniprot_id.unique()) - davis_homologs
log.info(f"Davis novel proteins: {len(davis_novel)}/{davis.uniprot_id.nunique()}")

# GLASS2
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
c_glass = {d: canon(d) for d in glass.ligand_smiles.unique()}
ov_d_glass = {d for d, c in c_glass.items() if c in bdb_cs}
ov_b_glass = {d for d, c in c_bdb.items() if c in {c_glass[x] for x in ov_d_glass}}
excl_drugs_glass = ov_d_glass | ov_b_glass
excl_prots_glass = set(glass.uniprot_id) & set(bdb.uniprot_id)

glass_homologs = set()
ghom_file = "results/glass_bdb_homologs_50.txt"
if os.path.exists(ghom_file):
    with open(ghom_file) as f:
        glass_homologs = {l.strip() for l in f if l.strip()}
glass_novel = set(glass.uniprot_id.unique()) - glass_homologs
log.info(f"GLASS2 novel proteins: {len(glass_novel)}/{glass.uniprot_id.nunique()}")

KNN_METHODS = ['drug_knn_k1', 'drug_knn_k5', 'prot_knn_k1', 'prot_knn_k5',
               'joint_knn_k10', 'anchor_knn']

# ═══════════════════════════════════════════════════════════════
# 1. Davis — all proteins (protein novelty only, all 68 drugs)
# ═══════════════════════════════════════════════════════════════
log.info(f"\n{'='*60}")
log.info("DAVIS — ALL PROTEINS (no drug exclusion)")
t0 = time.time()
df_davis = run_all_knn(davis, all_emb, all_fp,
                       excl_drugs=None, excl_prots=excl_prots_davis)
log.info(f"  Predictions: {time.time()-t0:.0f}s")
report("Davis All (442 prot)", df_davis, KNN_METHODS)
df_davis.to_csv("results/knn_davis_all.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# 2. Davis — novel proteins only (<50% k-mer)
# ═══════════════════════════════════════════════════════════════
log.info(f"\n{'='*60}")
log.info("DAVIS — NOVEL PROTEINS ONLY (<50% k-mer to BDB)")
df_davis_novel = df_davis[df_davis.uniprot_id.isin(davis_novel)].copy()
report("Davis Novel", df_davis_novel, KNN_METHODS)
df_davis_novel.to_csv("results/knn_davis_novel.csv", index=False)

# ═══════════════════════════════════════════════════════════════
# 3. Davis — strict (drug + protein exclusion)
# ═══════════════════════════════════════════════════════════════
log.info(f"\n{'='*60}")
log.info("DAVIS — STRICT (drug + protein exclusion)")
t0 = time.time()
df_davis_strict = run_all_knn(davis, all_emb, all_fp,
                              excl_drugs=excl_drugs_davis, excl_prots=excl_prots_davis)
log.info(f"  Predictions: {time.time()-t0:.0f}s")
report("Davis Strict (10 drugs)", df_davis_strict, KNN_METHODS)

# Novel subset of strict
df_strict_novel = df_davis_strict[df_davis_strict.uniprot_id.isin(davis_novel)].copy()
report("Davis Strict Novel", df_strict_novel, KNN_METHODS)

# ═══════════════════════════════════════════════════════════════
# 4. GLASS2 — novel proteins + canonical drug exclusion
# ═══════════════════════════════════════════════════════════════
log.info(f"\n{'='*60}")
log.info("GLASS2 — NOVEL (canonical drug + protein exclusion)")
t0 = time.time()
df_glass = run_all_knn(glass, raygun_pooled, all_fp_glass,
                       excl_drugs=excl_drugs_glass, excl_prots=excl_prots_glass)
if len(df_glass) >= 10:
    log.info(f"  Predictions: {time.time()-t0:.0f}s")
    report("GLASS2 Novel", df_glass, KNN_METHODS)
    df_glass.to_csv("results/knn_glass2_novel.csv", index=False)

    # Further filter to novel proteins (<50% k-mer)
    df_glass_novel = df_glass[df_glass.uniprot_id.isin(glass_novel)].copy()
    if len(df_glass_novel) >= 10:
        report("GLASS2 Novel (<50% k-mer)", df_glass_novel, KNN_METHODS)
else:
    log.info("  GLASS2: too few interactions after filtering")

# ═══════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════
log.info(f"\n{'='*60}")
log.info("REFERENCE (ConciseAnchor vs CoNCISE from paper):")
log.info("  Davis all:      CA=0.624  CoNCISE=0.527  Retrieval=0.520")
log.info("  Davis novel:    CA=0.717  CoNCISE=0.589")
log.info("  GLASS2:         CA=0.598  CoNCISE=0.547  Retrieval=0.575")
log.info("  GLASS2 oracle:  CA=0.651  CoNCISE=0.585")
log.info("=" * 60 + "\nDONE")
