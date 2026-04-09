"""Protein-kNN baseline for DTA: find k nearest BDB proteins, predict pKi.

Fast — prot_knn and anchor_knn only. No slow drug iteration.
Evaluates Davis (all + novel) and GLASS2 (novel).
"""
import os, sys, pickle, logging, time
import numpy as np
import pandas as pd
import torch
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ── load ─────────────────────────────────────────────────────────
log.info("Loading BDB...")
bdb = pd.read_csv("data/processed/bindingdb_interactions.csv")
morgan = pickle.load(open("data/processed/concise_bdb_morgan_fp.pkl", "rb"))
raygun_raw = torch.load("data/processed/raygun_bdb_embeddings.pt",
                         map_location="cpu", weights_only=False)
raygun_pooled = {k: (v.mean(dim=0).numpy() if isinstance(v, torch.Tensor)
                     else torch.tensor(v).mean(dim=0).numpy())
                 for k, v in raygun_raw.items()}
del raygun_raw

bdb = bdb[bdb.uniprot_id.isin(raygun_pooled) & bdb.ligand_smiles.isin(morgan)].copy()
log.info(f"BDB: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot")

bdb_prots = sorted(set(bdb.uniprot_id))
bdb_prot_idx = {p: i for i, p in enumerate(bdb_prots)}
bdb_emb_mat = np.array([raygun_pooled[p] for p in bdb_prots])
bdb_emb_norms = np.linalg.norm(bdb_emb_mat, axis=1, keepdims=True) + 1e-10
bdb_emb_normed = bdb_emb_mat / bdb_emb_norms
log.info(f"Emb matrix: {bdb_emb_mat.shape}")

# prot → mean pKi (for simple prot-kNN)
prot_mean_pki = bdb.groupby("uniprot_id").pki.mean().to_dict()
# prot → all pKi values
prot_all_pki = defaultdict(list)
for _, r in bdb.iterrows():
    prot_all_pki[bdb_prot_idx[r.uniprot_id]].append(r.pki)
log.info("Lookups built")


# ── prot similarity ──────────────────────────────────────────────

def prot_cosine(query_embs, exclude=None):
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    sims = qn @ bdb_emb_normed.T
    if exclude:
        for p in exclude:
            if p in bdb_prot_idx: sims[:, bdb_prot_idx[p]] = -1
    return sims


# ── kNN methods ──────────────────────────────────────────────────

def prot_knn_mean(rows, ps, pm, k):
    """Prot-kNN (mean): k nearest proteins by cosine, weighted avg of their mean pKi."""
    preds = []
    for uid in [r[1] for r in rows]:
        pi = pm.get(uid, -1)
        if pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-k:][::-1]
        vals, wts = [], []
        for bpi in top:
            if ps[pi, bpi] <= 0: continue
            mean_pki = np.mean(prot_all_pki.get(bpi, []))
            vals.append(mean_pki)
            wts.append(float(ps[pi, bpi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def prot_knn_all(rows, ps, pm, k):
    """Prot-kNN (all): k nearest proteins, weighted avg of ALL their interaction pKi values."""
    preds = []
    for uid in [r[1] for r in rows]:
        pi = pm.get(uid, -1)
        if pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-k:][::-1]
        vals, wts = [], []
        for bpi in top:
            if ps[pi, bpi] <= 0: continue
            for pki in prot_all_pki.get(bpi, []):
                vals.append(pki)
                wts.append(float(ps[pi, bpi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def prot_knn_max(rows, ps, pm, k):
    """Prot-kNN (max): k nearest proteins, use max pKi from each."""
    preds = []
    for uid in [r[1] for r in rows]:
        pi = pm.get(uid, -1)
        if pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-k:][::-1]
        vals, wts = [], []
        for bpi in top:
            if ps[pi, bpi] <= 0: continue
            pkis = prot_all_pki.get(bpi, [])
            if pkis:
                vals.append(max(pkis))
                wts.append(float(ps[pi, bpi]))
        preds.append(np.average(vals, weights=wts) if vals else np.nan)
    return preds

def anchor_knn(rows, ps, pm):
    """Anchor-kNN: nearest protein with pKi>=7 binder, return that pKi."""
    preds = []
    for uid in [r[1] for r in rows]:
        pi = pm.get(uid, -1)
        if pi < 0: preds.append(np.nan); continue
        top = np.argsort(ps[pi])[-20:][::-1]
        found = False
        for bpi in top:
            if ps[pi, bpi] <= 0: break
            pkis = prot_all_pki.get(bpi, [])
            strong = [p for p in pkis if p >= 7.0]
            if strong:
                preds.append(max(strong))
                found = True; break
        if not found: preds.append(np.nan)
    return preds


# ── evaluate ─────────────────────────────────────────────────────

def evaluate(name, df, emb_dict, excl_prots=None):
    log.info(f"\n{'='*60}\n  {name}")
    df = df[df.uniprot_id.isin(emb_dict)].copy()
    if excl_prots: df = df[~df.uniprot_id.isin(excl_prots)].copy()
    log.info(f"  {len(df)} int, {df.uniprot_id.nunique()} prot")
    if len(df) < 10: log.info("  Skip"); return df

    up = sorted(df.uniprot_id.unique())
    pm = {p: i for i, p in enumerate(up)}
    qemb = np.array([emb_dict[p] for p in up])

    t0 = time.time()
    ps = prot_cosine(qemb, excl_prots)
    log.info(f"  Cosine sims: {time.time()-t0:.1f}s")

    rows = list(zip(df.ligand_smiles.values, df.uniprot_id.values))
    methods = {}

    for k in [1, 3, 5, 10]:
        methods[f'prot_knn_mean_k{k}'] = prot_knn_mean(rows, ps, pm, k)
    for k in [1, 5]:
        methods[f'prot_knn_all_k{k}'] = prot_knn_all(rows, ps, pm, k)
    for k in [1, 5]:
        methods[f'prot_knn_max_k{k}'] = prot_knn_max(rows, ps, pm, k)
    methods['anchor_knn'] = anchor_knn(rows, ps, pm)

    log.info(f"  Done: {time.time()-t0:.0f}s")

    for m, p in methods.items(): df[m] = p

    log.info(f"\n  {'Method':<22s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s}")
    log.info(f"  {'-'*42}")
    for m in methods:
        ci = pp_ci(df, m); auc = pp_auroc(df, m); rmse = pp_rmse(df, m)
        log.info(f"  {m:<22s} {ci.mean():6.3f} "
                 f"{auc.mean() if len(auc) else 0:6.3f} "
                 f"{rmse.mean() if len(rmse) else 0:6.3f}")
    return df


# ── canonical SMILES ─────────────────────────────────────────────
try:
    from rdkit import Chem
    def canon(s):
        try: m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else s
        except: return s
except ImportError:
    def canon(s): return s

c_bdb = {d: canon(d) for d in bdb.ligand_smiles.unique()}
bdb_cs = set(c_bdb.values())

# ── Davis ────────────────────────────────────────────────────────
davis_raw = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis_raw.rename(columns={"protein_name": "uniprot_id",
                                   "drug_smiles": "ligand_smiles"})
excl_prots_davis = set(davis.uniprot_id) & set(bdb.uniprot_id)

CACHE = "results/raygun_davis_pooled.pt"
if os.path.exists(CACHE):
    davis_emb = torch.load(CACHE, map_location="cpu", weights_only=False)
    log.info(f"Davis Raygun cache: {len(davis_emb)}")
else:
    log.info("Computing Raygun for Davis...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()
    davis_seqs = {r.protein_name: r.protein_sequence
                  for _, r in davis_raw.drop_duplicates("protein_name").iterrows()}
    esm_embs = {}
    items = list(davis_seqs.items())
    for i in range(0, len(items), 8):
        batch = [(u, s[:1022]) for u, s in items[i:i+8]]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
        for j, (u, s) in enumerate(batch):
            esm_embs[u] = out["representations"][33][j, 1:len(s)+1, :].cpu()
    del esm_model; torch.cuda.empty_cache()
    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)
    davis_emb = {}
    for uid, emb in esm_embs.items():
        with torch.no_grad():
            out = raygun_model(emb.unsqueeze(0).to(DEVICE))
            proj = out[0] if isinstance(out, tuple) else out
            davis_emb[uid] = proj.squeeze(0).mean(dim=0).cpu().numpy()
    del raygun_model; torch.cuda.empty_cache()
    torch.save(davis_emb, CACHE)

all_emb = {**raygun_pooled, **davis_emb}

# Novel proteins
davis_homologs = set()
if os.path.exists("results/davis_bdb_homologs_50.txt"):
    with open("results/davis_bdb_homologs_50.txt") as f:
        davis_homologs = {l.strip() for l in f if l.strip()}
davis_novel_set = set(davis.uniprot_id.unique()) - davis_homologs
log.info(f"Davis novel: {len(davis_novel_set)}/{davis.uniprot_id.nunique()}")

# Evaluate
df_all = evaluate("Davis ALL (442 prot)", davis, all_emb, excl_prots_davis)
df_novel = df_all[df_all.uniprot_id.isin(davis_novel_set)].copy()
log.info(f"\n{'='*60}\n  Davis NOVEL (<50% k-mer, {df_novel.uniprot_id.nunique()} prot)")
methods = [c for c in df_novel.columns if 'knn' in c]
log.info(f"  {'Method':<22s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s}")
log.info(f"  {'-'*42}")
for m in methods:
    ci = pp_ci(df_novel, m); auc = pp_auroc(df_novel, m); rmse = pp_rmse(df_novel, m)
    log.info(f"  {m:<22s} {ci.mean():6.3f} "
             f"{auc.mean() if len(auc) else 0:6.3f} "
             f"{rmse.mean() if len(rmse) else 0:6.3f}")
df_all.to_csv("results/knn_prot_davis.csv", index=False)

# ── GLASS2 ───────────────────────────────────────────────────────
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
g_excl_p = set(glass.uniprot_id) & set(bdb.uniprot_id)
log.info(f"\nGLASS2: {len(glass)} int, overlap: {len(g_excl_p)} prot")

glass_homologs = set()
if os.path.exists("results/glass_bdb_homologs_50.txt"):
    with open("results/glass_bdb_homologs_50.txt") as f:
        glass_homologs = {l.strip() for l in f if l.strip()}
glass_novel_set = set(glass.uniprot_id.unique()) - glass_homologs

df_glass = evaluate("GLASS2 (excl overlap prot)", glass, raygun_pooled, g_excl_p)
if df_glass is not None and len(df_glass) > 10:
    df_glass.to_csv("results/knn_prot_glass2.csv", index=False)
    df_gn = df_glass[df_glass.uniprot_id.isin(glass_novel_set)].copy()
    if len(df_gn) > 10:
        log.info(f"\n{'='*60}\n  GLASS2 NOVEL (<50% k-mer, {df_gn.uniprot_id.nunique()} prot)")
        methods = [c for c in df_gn.columns if 'knn' in c]
        log.info(f"  {'Method':<22s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s}")
        log.info(f"  {'-'*42}")
        for m in methods:
            ci = pp_ci(df_gn, m); auc = pp_auroc(df_gn, m); rmse = pp_rmse(df_gn, m)
            log.info(f"  {m:<22s} {ci.mean():6.3f} "
                     f"{auc.mean() if len(auc) else 0:6.3f} "
                     f"{rmse.mean() if len(rmse) else 0:6.3f}")

log.info(f"\n{'='*60}")
log.info("REFERENCE (ConciseAnchor vs CoNCISE):")
log.info("  Davis all:   CA=0.624  CoNCISE=0.527  Retrieval=0.520")
log.info("  Davis novel: CA=0.717  CoNCISE=0.589")
log.info("  GLASS2:      CA=0.598  CoNCISE=0.547")
log.info("DONE")
