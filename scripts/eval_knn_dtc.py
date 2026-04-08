"""kNN baselines using DTC as retrieval pool — paper-fair protocol (FAST).

Matches the paper's evaluation:
- Training/retrieval pool: DTC 80/10/10 train split (seed 42)
- Test set: ALL Davis drugs + proteins NOT in DTC train
- Exact-match drugs masked in similarity matrix (can't trivially look up)
- kNN must predict from SIMILAR but not IDENTICAL drugs

Vectorized with scipy.sparse interaction matrix — no Python loops over neighbors.
"""
import os, sys, pickle, logging, time, random
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

# ── metrics (per-protein) ───────────────────────────────────────

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


# ── canonical SMILES ────────────────────────────────────────────

from rdkit import Chem
from rdkit.Chem import AllChem

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except:
        return s

def _compute_fp(s):
    m = Chem.MolFromSmiles(s)
    if m:
        return s, np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)
    return s, None


# ── load DTC training split (paper protocol) ───────────────────

log.info("Loading DTC data...")
dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")

dtc_prots = sorted(set(dtc.uniprot_id))
random.seed(42); random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1))
nv = max(1, int(len(dtc_prots) * 0.1))
train_prots = set(dtc_prots[nt + nv:])

dtc_train = dtc[dtc.uniprot_id.isin(train_prots)].copy()
log.info(f"DTC train split: {len(dtc_train)} int, {dtc_train.uniprot_id.nunique()} prot, "
         f"{dtc_train.ligand_smiles.nunique()} drugs")

# ── ESM2-650M embeddings for DTC ───────────────────────────────

log.info("Loading ESM2-650M embeddings...")
esm2_dtc = torch.load("data/processed/esm2_650m_dtc.pt",
                       map_location="cpu", weights_only=False)
esm2_dict = {}
for k, v in esm2_dtc.items():
    t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    esm2_dict[k] = t.numpy() if t.dim() == 1 else t.mean(dim=0).numpy()
del esm2_dtc
log.info(f"ESM2 embeddings: {len(esm2_dict)} proteins")

dtc_train = dtc_train[dtc_train.uniprot_id.isin(esm2_dict)].copy()
log.info(f"DTC train (with embeddings): {len(dtc_train)} int, "
         f"{dtc_train.uniprot_id.nunique()} prot, {dtc_train.ligand_smiles.nunique()} drugs")

# ── Morgan FPs for DTC drugs (parallel) ────────────────────────

FP_CACHE = "results/dtc_train_morgan_fp.pkl"
if os.path.exists(FP_CACHE):
    log.info("Loading cached DTC Morgan FPs...")
    morgan = pickle.load(open(FP_CACHE, "rb"))
else:
    drugs = sorted(dtc_train.ligand_smiles.unique())
    log.info(f"Computing Morgan FPs for {len(drugs)} DTC drugs ({cpu_count()} cores)...")
    with Pool(min(cpu_count(), 32)) as pool:
        results = pool.map(_compute_fp, drugs, chunksize=500)
    morgan = {s: fp for s, fp in results if fp is not None}
    pickle.dump(morgan, open(FP_CACHE, "wb"))
    log.info(f"Computed and cached {len(morgan)} Morgan FPs")

dtc_train = dtc_train[dtc_train.ligand_smiles.isin(morgan)].copy()
log.info(f"DTC train (with FPs): {len(dtc_train)} int, "
         f"{dtc_train.uniprot_id.nunique()} prot, {dtc_train.ligand_smiles.nunique()} drugs")

# ── Index structures + sparse interaction matrix ────────────────

dtc_drugs = sorted(set(dtc_train.ligand_smiles))
dtc_drug_idx = {d: i for i, d in enumerate(dtc_drugs)}
dtc_fp_mat = np.array([morgan[d] for d in dtc_drugs])
log.info(f"FP matrix: {dtc_fp_mat.shape} ({dtc_fp_mat.nbytes/1e9:.1f} GB)")

dtc_prots_list = sorted(set(dtc_train.uniprot_id))
dtc_prot_idx = {p: i for i, p in enumerate(dtc_prots_list)}
dtc_emb_mat = np.array([esm2_dict[p] for p in dtc_prots_list])
dtc_emb_norms = np.linalg.norm(dtc_emb_mat, axis=1, keepdims=True) + 1e-10
dtc_emb_normed = dtc_emb_mat / dtc_emb_norms
log.info(f"Emb matrix: {dtc_emb_mat.shape} ({dtc_emb_mat.nbytes/1e6:.0f} MB)")

n_drugs = len(dtc_drugs)
n_prots = len(dtc_prots_list)

# Sparse interaction matrix: (n_drugs, n_prots) with pKi values
# For duplicate (drug, prot) pairs, keep the max pKi
log.info("Building sparse interaction matrix...")
rows_d, cols_p, vals = [], [], []
for _, r in dtc_train.iterrows():
    rows_d.append(dtc_drug_idx[r.ligand_smiles])
    cols_p.append(dtc_prot_idx[r.uniprot_id])
    vals.append(r.pki)
# Use max for duplicates via a dict
pair_max = {}
for d, p, v in zip(rows_d, cols_p, vals):
    key = (d, p)
    if key not in pair_max or v > pair_max[key]:
        pair_max[key] = v
rows_d = [k[0] for k in pair_max]
cols_p = [k[1] for k in pair_max]
vals = list(pair_max.values())
INT_MAT = sp.csr_matrix((vals, (rows_d, cols_p)), shape=(n_drugs, n_prots))
# Dense version for fast indexing (drugs x prots) — uses ~1GB for 130K x 2K
# Only feasible because n_prots is small
INT_DENSE = INT_MAT.toarray()  # (n_drugs, n_prots), 0 where no interaction
log.info(f"Interaction matrix: {INT_DENSE.shape}, "
         f"nnz={INT_MAT.nnz} ({INT_DENSE.nbytes/1e9:.1f} GB)")

# Also build: for each drug, strongest binder pKi and which protein
drug_max_pki = np.max(INT_DENSE, axis=1)  # (n_drugs,)
drug_max_prot = np.argmax(INT_DENSE, axis=1)  # (n_drugs,)
# Strong binder mask (pKi >= 7)
drug_strong_mask = drug_max_pki >= 7.0

# Canonical SMILES for DTC drugs
log.info("Canonicalizing DTC drug SMILES...")
c_dtc = {d: canon(d) for d in dtc_drugs}
dtc_canonical_set = set(c_dtc.values())
log.info("Setup done")


# ── similarity helpers ──────────────────────────────────────────

def drug_sims(query_fps, mask_canonical=None, query_smiles=None):
    inter = query_fps @ dtc_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = dtc_fp_mat.sum(1, keepdims=True).T
    sims = inter / np.maximum(q_bits + db_bits - inter, 1)
    if mask_canonical and query_smiles:
        q_canon = [canon(s) for s in query_smiles]
        # Build mask matrix: which (query, dtc) pairs are exact matches
        dtc_canon_arr = [c_dtc.get(d, "") for d in dtc_drugs]
        for qi, qc in enumerate(q_canon):
            for di, dc in enumerate(dtc_canon_arr):
                if qc == dc:
                    sims[qi, di] = -1
    return sims

def prot_sims(query_embs, exclude=None):
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    sims = qn @ dtc_emb_normed.T
    if exclude:
        for p in exclude:
            if p in dtc_prot_idx:
                sims[:, dtc_prot_idx[p]] = -1
    return sims


# ── VECTORIZED kNN methods ─────────────────────────────────────

def run_drug_knn_vec(ds, ps, di_arr, pi_arr, k):
    """Vectorized drug-kNN: for each test pair, find k nearest drugs,
    then pick interaction with protein closest to query protein."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    # For each unique query drug, find top-k training drugs
    unique_di = np.unique(di_arr)
    for qdi in unique_di:
        mask_rows = di_arr == qdi
        top_k = np.argsort(ds[qdi])[-k:][::-1]
        top_sims = ds[qdi, top_k]
        valid = top_sims > 0
        top_k = top_k[valid]
        top_sims = top_sims[valid]
        if len(top_k) == 0: continue
        # INT_DENSE[top_k] → (k', n_prots): pKi values for neighbor drugs
        neighbor_pkis = INT_DENSE[top_k]  # (k', n_prots)
        # For each test row with this drug, find best protein match
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qpi = pi_arr[ri]
            if qpi < 0: continue
            # For each neighbor drug, pick protein closest to query
            prot_s = ps[qpi]  # (n_dtc_prots,)
            best_pkis = []
            best_wts = []
            for ki, bdi in enumerate(top_k):
                # Interactions of this neighbor drug
                int_row = neighbor_pkis[ki]  # (n_prots,)
                has_int = int_row > 0
                if not has_int.any(): continue
                # Among proteins this drug interacts with, pick closest to query
                candidate_ps = prot_s.copy()
                candidate_ps[~has_int] = -2
                best_pi = np.argmax(candidate_ps)
                if candidate_ps[best_pi] <= 0: continue
                best_pkis.append(int_row[best_pi])
                best_wts.append(top_sims[ki])
            if best_pkis:
                preds[ri] = np.average(best_pkis, weights=best_wts)
    return preds

def run_prot_knn_vec(ds, ps, di_arr, pi_arr, k):
    """Vectorized protein-kNN — memory efficient (no INT_DENSE.T copy)."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    unique_pi = np.unique(pi_arr[pi_arr >= 0])
    for qpi in unique_pi:
        mask_rows = pi_arr == qpi
        top_k = np.argsort(ps[qpi])[-k:][::-1]
        top_sims = ps[qpi, top_k]
        valid = top_sims > 0
        top_k = top_k[valid]
        top_sims = top_sims[valid]
        if len(top_k) == 0: continue
        # Index INT_DENSE by column (protein) — no full transpose
        neighbor_pkis = [INT_DENSE[:, bpi] for bpi in top_k]  # list of (n_drugs,)
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qdi = di_arr[ri]
            if qdi < 0: continue
            drug_s = ds[qdi]  # (n_dtc_drugs,)
            best_pkis = []
            best_wts = []
            for ki, bpi in enumerate(top_k):
                int_col = neighbor_pkis[ki]  # (n_drugs,) — interactions of this protein
                has_int = int_col > 0
                if not has_int.any(): continue
                # Among drugs this protein interacts with, pick closest to query
                best_di_score = -1
                best_pki = None
                # Vectorized: mask and argmax
                masked = np.where(has_int, drug_s, -2)
                best_di = np.argmax(masked)
                if masked[best_di] <= 0: continue
                best_pkis.append(int_col[best_di])
                best_wts.append(top_sims[ki])
            if best_pkis:
                preds[ri] = np.average(best_pkis, weights=best_wts)
    return preds

def run_joint_knn_vec(ds, ps, di_arr, pi_arr, k, alpha=0.5):
    """Vectorized joint-kNN using sparse interaction matrix."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    # Precompute interaction indices
    int_rows, int_cols = INT_MAT.nonzero()
    int_vals = np.array(INT_MAT[int_rows, int_cols]).flatten()
    n_int = len(int_rows)
    for i in range(n):
        qdi, qpi = di_arr[i], pi_arr[i]
        if qdi < 0 or qpi < 0: continue
        scores = alpha * ds[qdi][int_rows] + (1-alpha) * ps[qpi][int_cols]
        topk = np.argpartition(scores, -min(k, n_int))[-k:]
        s, p = scores[topk], int_vals[topk]
        m = s > 0
        if m.sum():
            preds[i] = np.average(p[m], weights=s[m])
    return preds

def run_anchor_knn_vec(ds, ps, di_arr, pi_arr):
    """Vectorized anchor-kNN: nearest drug with strong binder."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    for i in range(n):
        qdi, qpi = di_arr[i], pi_arr[i]
        if qdi < 0 or qpi < 0: continue
        top10 = np.argsort(ds[qdi])[-10:][::-1]
        for bdi in top10:
            if ds[qdi, bdi] <= 0: break
            int_row = INT_DENSE[bdi]
            strong = int_row >= 7.0
            if not strong.any(): continue
            # Among strong binders, pick protein closest to query
            candidate_ps = ps[qpi].copy()
            candidate_ps[~strong] = -2
            best_pi = np.argmax(candidate_ps)
            if candidate_ps[best_pi] > 0:
                preds[i] = int_row[best_pi]
                break
    return preds


# ── evaluate ────────────────────────────────────────────────────

def evaluate(name, df, emb_dict, fp_dict, mask_canonical_drugs=True,
             exclude_prots=None):
    log.info(f"\n{'='*60}\n  {name}")

    df = df[df.uniprot_id.isin(emb_dict) & df.ligand_smiles.isin(fp_dict)].copy()
    if exclude_prots:
        df = df[~df.uniprot_id.isin(exclude_prots)].copy()
    log.info(f"  {len(df)} int, {df.uniprot_id.nunique()} prot, "
             f"{df.ligand_smiles.nunique()} drugs")
    if len(df) < 10:
        log.info("  Too few, skip"); return

    ud = sorted(df.ligand_smiles.unique())
    up = sorted(df.uniprot_id.unique())
    dm = {d: i for i, d in enumerate(ud)}
    pm = {p: i for i, p in enumerate(up)}

    qfp = np.array([np.array(fp_dict[d], dtype=np.float32) for d in ud])
    qemb = np.array([emb_dict[p] for p in up])

    t0 = time.time()
    ds = drug_sims(qfp,
                   mask_canonical=dtc_canonical_set if mask_canonical_drugs else None,
                   query_smiles=ud if mask_canonical_drugs else None)
    ps = prot_sims(qemb, exclude=exclude_prots)
    log.info(f"  Sims: {time.time()-t0:.1f}s  (ds {ds.shape}, ps {ps.shape})")

    if mask_canonical_drugs:
        n_masked = sum(1 for d in ud if canon(d) in dtc_canonical_set)
        log.info(f"  Exact-match drugs masked in retrieval: {n_masked}/{len(ud)}")

    # Map test rows to query indices
    di_arr = np.array([dm.get(s, -1) for s in df.ligand_smiles.values])
    pi_arr = np.array([pm.get(p, -1) for p in df.uniprot_id.values])

    log.info(f"\n  {'Method':<18s} {'CI':>6s} {'AUROC':>6s} {'RMSE':>6s} {'NaN%':>5s}  {'t':>5s}")
    log.info(f"  {'-'*52}")

    def run_and_report(method_name, preds):
        df[method_name] = preds
        p = np.array(preds, dtype=float)
        ci = pp_ci(df, method_name)
        auc = pp_auroc(df, method_name)
        rmse = pp_rmse(df, method_name)
        log.info(f"  {method_name:<18s} {ci.mean():6.3f} "
                 f"{auc.mean() if len(auc) else 0:6.3f} "
                 f"{rmse.mean() if len(rmse) else 0:6.3f} "
                 f"{np.isnan(p).mean()*100:5.1f}  "
                 f"{time.time()-t0:5.0f}s")
        sys.stdout.flush()

    for k in [1, 5, 10]:
        run_and_report(f'drug_knn_k{k}', run_drug_knn_vec(ds, ps, di_arr, pi_arr, k))
    for k in [1, 5, 10]:
        run_and_report(f'prot_knn_k{k}', run_prot_knn_vec(ds, ps, di_arr, pi_arr, k))
    for k in [10, 20]:
        run_and_report(f'joint_knn_k{k}', run_joint_knn_vec(ds, ps, di_arr, pi_arr, k))
    run_and_report('anchor_knn', run_anchor_knn_vec(ds, ps, di_arr, pi_arr))

    log.info(f"  {'-'*52}")
    log.info(f"  All done: {time.time()-t0:.0f}s")

    df.to_csv(f"results/knn_dtc_{name.lower().replace(' ','_')}.csv", index=False)
    log.info(f"  Saved results/knn_dtc_{name.lower().replace(' ','_')}.csv")


# ── DAVIS ───────────────────────────────────────────────────────

log.info("\n" + "="*60 + "\nLoading Davis...")
davis_raw = pd.read_csv("data/raw/davis/davis_benchmark.csv")
davis = davis_raw.rename(columns={"protein_name": "uniprot_id",
                                   "drug_smiles": "ligand_smiles"})
log.info(f"Davis: {len(davis)} int, {davis.uniprot_id.nunique()} prot, "
         f"{davis.ligand_smiles.nunique()} drugs")

excl_prots = set(davis.uniprot_id.unique()) & set(dtc_prots_list)
log.info(f"Davis protein overlap with DTC train: {len(excl_prots)}/{davis.uniprot_id.nunique()}")

# ESM2 embeddings for Davis proteins
DAVIS_EMB_CACHE = "results/esm2_davis_pooled.pt"
if os.path.exists(DAVIS_EMB_CACHE):
    davis_emb = torch.load(DAVIS_EMB_CACHE, map_location="cpu", weights_only=False)
    davis_emb = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in davis_emb.items()}
    log.info(f"Loaded Davis ESM2 cache: {len(davis_emb)} proteins")
else:
    log.info("Computing ESM2-650M for Davis proteins...")
    import esm
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()

    davis_seqs = {r.protein_name: r.protein_sequence
                  for _, r in davis_raw.drop_duplicates("protein_name").iterrows()}

    davis_emb = {}
    items = list(davis_seqs.items())
    for i in range(0, len(items), 8):
        batch = [(u, s[:1022]) for u, s in items[i:i+8]]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
        for j, (u, s) in enumerate(batch):
            davis_emb[u] = out["representations"][33][j, 1:len(s)+1, :].mean(dim=0).cpu().numpy()
        if (i+8) % 50 < 8:
            log.info(f"  ESM2: {min(i+8, len(items))}/{len(items)}")

    del esm_model; torch.cuda.empty_cache()
    torch.save(davis_emb, DAVIS_EMB_CACHE)
    log.info(f"Cached {len(davis_emb)} Davis ESM2 embeddings")

all_emb = {**esm2_dict, **davis_emb}

all_fp = dict(morgan)
missing = set(davis.ligand_smiles.unique()) - set(all_fp.keys())
if missing:
    log.info(f"Computing {len(missing)} missing Davis Morgan FPs...")
    for s in missing:
        m = Chem.MolFromSmiles(s)
        if m:
            all_fp[s] = np.array(
                AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048),
                dtype=np.float32)

# Paper-fair: all drugs in test, exact matches masked in retrieval
evaluate("Davis_paper_fair", davis, all_emb, all_fp,
         mask_canonical_drugs=True, exclude_prots=excl_prots)

# Novel-protein variant
HOMOLOG_FILE = "results/davis_bdb_homologs_50.txt"
if os.path.exists(HOMOLOG_FILE):
    homolog_prots = set(open(HOMOLOG_FILE).read().strip().split("\n"))
    log.info(f"\nHomolog exclusion list: {len(homolog_prots)} proteins")
    evaluate("Davis_novel_prot", davis, all_emb, all_fp,
             mask_canonical_drugs=True,
             exclude_prots=excl_prots | homolog_prots)

log.info(f"\n{'='*60}")
log.info("REFERENCE (from paper, same eval protocol):")
log.info("  Davis:  ConciseAnchor CI=0.624  CoNCISE CI=0.527")
log.info("="*60 + "\nDONE")
