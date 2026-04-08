"""kNN baselines for DTA prediction — lightweight, memory-efficient.

Compares Drug-kNN, Protein-kNN, Joint-kNN, and Anchor-kNN against
ConciseAnchor on Davis and GLASS2 cross-dataset benchmarks.
Uses mean-pooled Raygun (1280-dim) for protein similarity and
Morgan FP Tanimoto for drug similarity.
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


# ── load BDB training data ──────────────────────────────────────

log.info("Loading BDB data...")
bdb = pd.read_csv("data/processed/bindingdb_interactions.csv")
morgan = pickle.load(open("data/processed/concise_bdb_morgan_fp.pkl", "rb"))
raygun_raw = torch.load("data/processed/raygun_bdb_embeddings.pt",
                         map_location="cpu", weights_only=False)

# Mean-pool Raygun: (50,1280) → (1280,) per protein
raygun_pooled = {}
for k, v in raygun_raw.items():
    t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    raygun_pooled[k] = t.mean(dim=0).numpy()  # (1280,)
del raygun_raw

bdb = bdb[bdb.uniprot_id.isin(raygun_pooled) & bdb.ligand_smiles.isin(morgan)].copy()
log.info(f"BDB: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot, "
         f"{bdb.ligand_smiles.nunique()} drugs")

# Index structures
bdb_drugs = sorted(set(bdb.ligand_smiles))
bdb_drug_idx = {d: i for i, d in enumerate(bdb_drugs)}
bdb_fp_mat = np.array([np.array(morgan[d], dtype=np.float32) for d in bdb_drugs])
log.info(f"FP matrix: {bdb_fp_mat.shape} ({bdb_fp_mat.nbytes/1e9:.1f} GB)")

bdb_prots = sorted(set(bdb.uniprot_id))
bdb_prot_idx = {p: i for i, p in enumerate(bdb_prots)}
bdb_emb_mat = np.array([raygun_pooled[p] for p in bdb_prots])  # (N, 1280)
bdb_emb_norms = np.linalg.norm(bdb_emb_mat, axis=1, keepdims=True) + 1e-10
bdb_emb_normed = bdb_emb_mat / bdb_emb_norms
log.info(f"Emb matrix: {bdb_emb_mat.shape} ({bdb_emb_mat.nbytes/1e6:.0f} MB)")

# drug → [(prot_idx, pki)], prot → [(drug_idx, pki)]
d2i = defaultdict(list)
p2i = defaultdict(list)
for _, r in bdb.iterrows():
    di, pi = bdb_drug_idx[r.ligand_smiles], bdb_prot_idx[r.uniprot_id]
    d2i[di].append((pi, r.pki))
    p2i[pi].append((di, r.pki))
log.info("Lookups built")


# ── similarity helpers ───────────────────────────────────────────

def drug_sims(query_fps, exclude=None):
    """Tanimoto: (n_query, 2048) vs (n_bdb, 2048) → (n_query, n_bdb)."""
    inter = query_fps @ bdb_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = bdb_fp_mat.sum(1, keepdims=True).T
    sims = inter / np.maximum(q_bits + db_bits - inter, 1)
    if exclude:
        for d in exclude:
            if d in bdb_drug_idx: sims[:, bdb_drug_idx[d]] = -1
    return sims

def prot_sims(query_embs, exclude=None):
    """Cosine: (n_query, 1280) vs (n_bdb, 1280) → (n_query, n_bdb)."""
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    sims = qn @ bdb_emb_normed.T
    if exclude:
        for p in exclude:
            if p in bdb_prot_idx: sims[:, bdb_prot_idx[p]] = -1
    return sims


# ── kNN methods (multiprocessing) ────────────────────────────────

N_WORKERS = min(cpu_count(), 48)

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
                if ps[pi, bpi] > best_ps:
                    best_ps = ps[pi, bpi]; best_pki = pki
            if best_pki is not None:
                vals.append(best_pki); wts.append(float(ds[di, bdi]))
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
                if ds[di, bdi] > best_ds:
                    best_ds = ds[di, bdi]; best_pki = pki
            if best_pki is not None:
                vals.append(best_pki); wts.append(float(ps[pi, bpi]))
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
                if ps[pi, bpi] > best_ps:
                    best_ps = ps[pi, bpi]; best_pki = pki
            if best_pki is not None:
                preds.append(best_pki); found = True; break
        if not found: preds.append(np.nan)
    return preds

def _par(fn, rows, ds, ps, dm, pm, lookup, extra=None):
    n = len(rows)
    sz = max(1, n // N_WORKERS)
    chunks = []
    for i in range(0, n, sz):
        c = rows[i:i+sz]
        if extra is not None:
            chunks.append((c, ds, ps, dm, pm, lookup, extra))
        else:
            chunks.append((c, ds, ps, dm, pm, lookup))
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


# ── evaluate ─────────────────────────────────────────────────────

def evaluate(name, bench, emb_dict, fp_dict, excl_drugs=None, excl_prots=None):
    log.info(f"\n{'='*60}\n  {name}")
    df = bench[bench.uniprot_id.isin(emb_dict) & bench.ligand_smiles.isin(fp_dict)].copy()
    if excl_drugs: df = df[~df.ligand_smiles.isin(excl_drugs)].copy()
    if excl_prots: df = df[~df.uniprot_id.isin(excl_prots)].copy()
    log.info(f"  {len(df)} int, {df.uniprot_id.nunique()} prot, "
             f"{df.ligand_smiles.nunique()} drugs")
    if len(df) < 10: log.info("  Too few, skip"); return

    ud = sorted(df.ligand_smiles.unique())
    up = sorted(df.uniprot_id.unique())
    dm = {d: i for i, d in enumerate(ud)}
    pm = {p: i for i, p in enumerate(up)}

    qfp = np.array([np.array(fp_dict[d], dtype=np.float32) for d in ud])
    qemb = np.array([emb_dict[p] for p in up])

    t0 = time.time()
    ds = drug_sims(qfp, excl_drugs)
    ps = prot_sims(qemb, excl_prots)
    log.info(f"  Sims: {time.time()-t0:.1f}s  (ds {ds.shape}, ps {ps.shape})")

    rows = list(zip(df.ligand_smiles.values, df.uniprot_id.values))
    methods = {}

    for k in [1, 5, 10]:
        log.info(f"  drug_knn k={k}...")
        methods[f'drug_knn_k{k}'] = run_drug_knn(rows, ds, ps, dm, pm, k)
    for k in [1, 5, 10]:
        log.info(f"  prot_knn k={k}...")
        methods[f'prot_knn_k{k}'] = run_prot_knn(rows, ds, ps, dm, pm, k)
    for k in [10, 20]:
        log.info(f"  joint_knn k={k}...")
        methods[f'joint_knn_k{k}'] = run_joint_knn(rows, ds, ps, dm, pm, k)
    log.info("  anchor_knn...")
    methods['anchor_knn'] = run_anchor_knn(rows, ds, ps, dm, pm)

    log.info(f"  Done: {time.time()-t0:.0f}s")

    for m, p in methods.items(): df[m] = p

    log.info(f"\n  {'Method':<18s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s} {'NaN%':>5s}")
    log.info(f"  {'-'*42}")
    for m in methods:
        p = np.array(methods[m], dtype=float)
        ci = pp_ci(df, m); auc = pp_auroc(df, m); rmse = pp_rmse(df, m)
        log.info(f"  {m:<18s} {ci.mean():6.3f} "
                 f"{auc.mean() if len(auc) else 0:6.3f} "
                 f"{rmse.mean() if len(rmse) else 0:6.3f} "
                 f"{np.isnan(p).mean()*100:5.1f}")

    df.to_csv(f"results/knn_{name.lower().replace(' ','_')}.csv", index=False)
    log.info(f"  Saved results/knn_{name.lower().replace(' ','_')}.csv")


# ── main ─────────────────────────────────────────────────────────

# Canonical SMILES overlap
try:
    from rdkit import Chem
    def canon(s):
        try: m = Chem.MolFromSmiles(s); return Chem.MolToSmiles(m) if m else s
        except: return s
except ImportError:
    def canon(s): return s

# ── DAVIS ────────────────────────────────────────────────────────
davis_raw = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis_raw.rename(columns={"protein_name": "uniprot_id",
                                   "drug_smiles": "ligand_smiles"})
log.info(f"Davis: {len(davis)} int, {davis.uniprot_id.nunique()} prot")

# Overlap
c_davis = {d: canon(d) for d in davis.ligand_smiles.unique()}
c_bdb = {d: canon(d) for d in bdb.ligand_smiles.unique()}
bdb_cs = set(c_bdb.values())
ov_d = {d for d, c in c_davis.items() if c in bdb_cs}
ov_b = {d for d, c in c_bdb.items() if c in {c_davis[x] for x in ov_d}}
excl_drugs_davis = ov_d | ov_b
excl_prots_davis = set(davis.uniprot_id) & set(bdb.uniprot_id)
log.info(f"Davis overlap: {len(ov_d)} drugs, {len(excl_prots_davis)} prots")

# Raygun for Davis — compute or load cache
CACHE = "results/raygun_davis_pooled.pt"
if os.path.exists(CACHE):
    davis_emb = torch.load(CACHE, map_location="cpu", weights_only=False)
    log.info(f"Loaded Davis Raygun cache: {len(davis_emb)} proteins")
else:
    log.info("Computing Raygun for Davis proteins...")
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
        if (i+8) % 50 < 8:
            log.info(f"  ESM-2: {min(i+8, len(items))}/{len(items)}")

    del esm_model; torch.cuda.empty_cache()

    raygun_model, _, _ = torch.hub.load(
        "rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)

    davis_emb = {}
    for uid, emb in esm_embs.items():
        with torch.no_grad():
            out = raygun_model(emb.unsqueeze(0).to(DEVICE))
            proj = out[0] if isinstance(out, tuple) else out
            davis_emb[uid] = proj.squeeze(0).mean(dim=0).cpu().numpy()  # (1280,)

    del raygun_model; torch.cuda.empty_cache()
    torch.save(davis_emb, CACHE)
    log.info(f"Cached {len(davis_emb)} Davis Raygun embeddings")

# Merge embeddings
all_emb = {**raygun_pooled, **davis_emb}

# Davis FPs
all_fp = dict(morgan)
if os.path.exists("results/concise_davis_fp.pkl"):
    all_fp.update(pickle.load(open("results/concise_davis_fp.pkl", "rb")))
# Compute missing
missing = set(davis.ligand_smiles.unique()) - set(all_fp.keys())
if missing:
    log.info(f"Computing {len(missing)} missing Davis Morgan FPs...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    for s in missing:
        m = Chem.MolFromSmiles(s)
        if m: all_fp[s] = np.array(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)

# Davis with drug exclusion (strict — only 10 novel drugs remain)
evaluate("Davis_strict", davis, all_emb, all_fp, excl_drugs_davis, excl_prots_davis)

# Davis with protein-only exclusion (all 68 drugs, 442 novel proteins)
# This matches the paper's main evaluation protocol
evaluate("Davis_prot_novel", davis, all_emb, all_fp,
         excl_drugs=None, excl_prots=excl_prots_davis)

# ── GLASS2 ───────────────────────────────────────────────────────
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
log.info(f"\nGLASS2: {len(glass)} int, {glass.uniprot_id.nunique()} prot")

# Use canonical SMILES for drug overlap (not raw string matching)
c_glass = {d: canon(d) for d in glass.ligand_smiles.unique()}
ov_glass_d = {d for d, c in c_glass.items() if c in bdb_cs}
ov_glass_b = {d for d, c in c_bdb.items() if c in {c_glass[x] for x in ov_glass_d}}
g_excl_d = ov_glass_d | ov_glass_b
g_excl_p = set(glass.uniprot_id.unique()) & set(bdb.uniprot_id.unique())
log.info(f"GLASS2 overlap (canonical): {len(ov_glass_d)}/{glass.ligand_smiles.nunique()} drugs, "
         f"{len(g_excl_p)}/{glass.uniprot_id.nunique()} prots")

all_fp_g = dict(morgan)
if os.path.exists("results/concise_glass_fp.pkl"):
    all_fp_g.update(pickle.load(open("results/concise_glass_fp.pkl", "rb")))

evaluate("GLASS2", glass, raygun_pooled, all_fp_g, g_excl_d, g_excl_p)

# ── Summary ──────────────────────────────────────────────────────
log.info(f"\n{'='*60}")
log.info("REFERENCE (from paper, same eval protocol):")
log.info("  Davis:  ConciseAnchor CI=0.624  CoNCISE CI=0.527  Retrieval-only CI=0.520")
log.info("  GLASS2: ConciseAnchor CI=0.598  CoNCISE CI=0.547  Retrieval-only CI=0.575")
log.info("="*60 + "\nDONE")
