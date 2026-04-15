"""Lean prot_knn k=1,5 on DTC — vectorized numpy, no per-sample loops."""
import os, sys, pickle, logging, time, random
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_WORKERS = cpu_count()
log.info(f"Using {N_WORKERS} CPU workers, device={DEVICE}")


# ── per-protein metrics ──────────────────────────────────────────

def _ci_one(yt, yp):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 3 or yp.std() < 1e-8:
        return 0.5 if len(yt) >= 3 else np.nan
    dt = yt[:, None] - yt[None, :]
    dp = yp[:, None] - yp[None, :]
    idx = np.triu_indices(len(yt), k=1)
    dt_f, dp_f = dt[idx], dp[idx]
    nz = dt_f != 0
    if nz.sum() == 0:
        return 0.5
    dt_nz, dp_nz = dt_f[nz], dp_f[nz]
    return float(((dt_nz * dp_nz > 0).sum() + 0.5 * (dp_nz == 0).sum()) / len(dt_nz))


def _auroc_one(yt, yp, hi=7.0, lo=5.0):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    mask = (yt >= hi) | (yt <= lo)
    if mask.sum() < 2:
        return np.nan
    labels = (yt[mask] >= hi).astype(int)
    if len(set(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, yp[mask])


def _rmse_one(yt, yp):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    return np.sqrt(mean_squared_error(yt, yp)) if len(yt) >= 3 else np.nan


def _metrics_worker(args):
    uid, yt, yp = args
    return uid, _ci_one(yt, yp), _auroc_one(yt, yp), _rmse_one(yt, yp)


def pp_metrics(df, col):
    tasks = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        tasks.append((uid, s.pki.values, np.array(s[col].values, dtype=float)))
    with Pool(N_WORKERS) as pool:
        results = pool.map(_metrics_worker, tasks)
    cis = [r[1] for r in results if not np.isnan(r[1])]
    aucs = [r[2] for r in results if not np.isnan(r[2])]
    rmses = [r[3] for r in results if not np.isnan(r[3])]
    return np.array(cis), np.array(aucs), np.array(rmses)


# ── prot_knn — fully vectorized, no multiprocessing needed ───────

def run_prot_knn_vectorized(ds, ps, di_arr, pi_arr, k, int_dense_t):
    """
    For each query protein, find k nearest train proteins.
    For each (query_drug, neighbor_protein) pair, find the best matching
    train drug by drug similarity, weighted by protein similarity.

    Only 268 unique query proteins — loop is trivial.
    Inner loops vectorized with numpy.
    """
    n = len(di_arr)
    preds = np.full(n, np.nan)
    unique_pi = np.unique(pi_arr[pi_arr >= 0])

    # Precompute top-k for all unique proteins
    ps_sub = ps[unique_pi]
    topk_all = np.argsort(ps_sub, axis=1)[:, -k:][:, ::-1]

    log.info(f"  prot_knn k={k}: processing {len(unique_pi)} proteins...")

    for li, qpi in enumerate(tqdm(unique_pi, desc=f"prot_knn_k{k}")):
        top_k = topk_all[li]
        top_sims = ps[qpi, top_k]
        valid = top_sims > 0
        top_k = top_k[valid]
        top_sims = top_sims[valid]
        if len(top_k) == 0:
            continue

        # neighbor_pkis: (k', n_drugs) — pKi values for each neighbor protein
        neighbor_pkis = int_dense_t[top_k]  # shape: (k', n_train_drugs)

        # All rows for this query protein
        mask_rows = pi_arr == qpi
        row_indices = np.where(mask_rows)[0]
        di_sub = di_arr[row_indices]

        # Get unique query drugs for this protein
        unique_di = np.unique(di_sub[di_sub >= 0])
        if len(unique_di) == 0:
            continue

        # Drug similarities for all unique query drugs: (n_unique_drugs, n_train_drugs)
        ds_unique = ds[unique_di]

        # For each unique query drug, compute weighted prediction
        drug_preds = {}
        for dli, qdi in enumerate(unique_di):
            drug_s = ds_unique[dli]  # (n_train_drugs,)
            # For each neighbor protein, find best matching train drug
            has_int = neighbor_pkis > 0  # (k', n_train_drugs)
            # Mask drug similarities where no interaction exists
            masked_ds = np.where(has_int, drug_s[None, :], -2)  # (k', n_train_drugs)
            best_dis = np.argmax(masked_ds, axis=1)  # (k',)
            valid_k = masked_ds[np.arange(len(top_k)), best_dis] > 0
            if valid_k.any():
                pkis = neighbor_pkis[np.arange(len(top_k)), best_dis][valid_k]
                wts = top_sims[valid_k]
                drug_preds[qdi] = np.average(pkis, weights=wts)

        # Assign predictions to rows
        for ri, qdi in zip(row_indices, di_sub):
            if qdi in drug_preds:
                preds[ri] = drug_preds[qdi]

    nan_pct = np.isnan(preds).mean() * 100
    log.info(f"  prot_knn k={k}: NaN={nan_pct:.1f}%")
    return preds


# ── FP helpers ───────────────────────────────────────────────────

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


# ── main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()
    log.info(f"prot_knn k=1,5 — DTC In-Domain — {N_WORKERS} cores, GPU={DEVICE}")

    # Load DTC
    dtc_path = PROJECT / "embeddings_model_files" / "dtc_training_interactions.csv"
    if not dtc_path.exists():
        dtc_path = PROJECT / "data" / "processed" / "dtc_training_interactions.csv"
    dtc = pd.read_csv(dtc_path)
    log.info(f"DTC: {len(dtc)} int, {dtc.uniprot_id.nunique()} prot, {dtc.ligand_smiles.nunique()} drugs")

    # ESM2
    esm2_path = PROJECT / "embeddings_model_files" / "esm2_650m_dtc.pt"
    esm2_raw = torch.load(esm2_path, map_location="cpu", weights_only=False)
    esm2_dict = {}
    for k, v in esm2_raw.items():
        t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
        esm2_dict[k] = t.numpy() if t.dim() == 1 else t.mean(dim=0).numpy()
    del esm2_raw
    log.info(f"ESM2: {len(esm2_dict)} proteins")

    # Split (same as original)
    random.seed(42)
    dtc_prots = sorted(set(dtc.uniprot_id) & set(esm2_dict.keys()))
    random.shuffle(dtc_prots)
    nt = max(1, int(len(dtc_prots) * 0.1))
    nv = max(1, int(len(dtc_prots) * 0.1))
    test_prots = set(dtc_prots[:nt])
    val_prots = set(dtc_prots[nt:nt + nv])
    train_prots = set(dtc_prots[nt + nv:])

    dtc_filt = dtc[dtc.uniprot_id.isin(esm2_dict)].copy()
    train_df = dtc_filt[dtc_filt.uniprot_id.isin(train_prots)].copy()
    test_df = dtc_filt[dtc_filt.uniprot_id.isin(test_prots)].copy()
    log.info(f"Train: {len(train_df)} ({train_df.uniprot_id.nunique()} prot), Test: {len(test_df)} ({test_df.uniprot_id.nunique()} prot)")

    # Morgan FPs
    FP_CACHE = RESULTS_DIR / "dtc_indomain_morgan_fp.pkl"
    if FP_CACHE.exists():
        log.info("Loading cached Morgan FPs...")
        with open(FP_CACHE, "rb") as f:
            morgan = pickle.load(f)
    else:
        drugs = sorted(set(train_df.ligand_smiles) | set(test_df.ligand_smiles))
        log.info(f"Computing Morgan FPs for {len(drugs)} drugs ({N_WORKERS} workers)...")
        with Pool(N_WORKERS) as pool:
            results = list(tqdm(pool.imap(_compute_fp, drugs, chunksize=500),
                               total=len(drugs), desc="Morgan FP"))
        morgan = {s: fp for s, fp in results if fp is not None}
        with open(FP_CACHE, "wb") as f:
            pickle.dump(morgan, f)
        log.info(f"Cached {len(morgan)} Morgan FPs")

    train_df = train_df[train_df.ligand_smiles.isin(morgan)].copy()
    test_df = test_df[test_df.ligand_smiles.isin(morgan)].copy()
    log.info(f"After FP filter — Train: {len(train_df)}, Test: {len(test_df)}")

    # Index structures
    train_drugs = sorted(set(train_df.ligand_smiles))
    train_drug_idx = {d: i for i, d in enumerate(train_drugs)}
    train_fp_mat = np.array([morgan[d] for d in train_drugs])

    train_prots_list = sorted(set(train_df.uniprot_id))
    train_prot_idx = {p: i for i, p in enumerate(train_prots_list)}
    train_emb_mat = np.array([esm2_dict[p] for p in train_prots_list])
    train_emb_normed = train_emb_mat / (np.linalg.norm(train_emb_mat, axis=1, keepdims=True) + 1e-10)

    # Interaction matrix
    di_col = train_df.ligand_smiles.map(train_drug_idx).values
    pi_col = train_df.uniprot_id.map(train_prot_idx).values
    pki_col = train_df.pki.values
    tmp = pd.DataFrame({"di": di_col, "pi": pi_col, "pki": pki_col})
    tmp = tmp.groupby(["di", "pi"]).pki.max().reset_index()
    INT_MAT = sp.csr_matrix((tmp.pki.values, (tmp.di.values, tmp.pi.values)),
                             shape=(len(train_drugs), len(train_prots_list)))
    INT_DENSE = INT_MAT.toarray()
    INT_DENSE_T = INT_DENSE.T  # (n_prots, n_drugs)
    log.info(f"Interaction matrix: {INT_DENSE.shape}, nnz={INT_MAT.nnz}")

    # ── Evaluate test set ──
    log.info(f"\nEvaluating test: {len(test_df)} int, {test_df.uniprot_id.nunique()} prot, {test_df.ligand_smiles.nunique()} drugs")

    eval_df = test_df[test_df.uniprot_id.isin(esm2_dict) & test_df.ligand_smiles.isin(morgan)].copy()

    ud = sorted(eval_df.ligand_smiles.unique())
    up = sorted(eval_df.uniprot_id.unique())
    dm = {d: i for i, d in enumerate(ud)}
    pm = {p: i for i, p in enumerate(up)}

    qfp = np.array([morgan[d] for d in ud])
    qemb = np.array([esm2_dict[p] for p in up])

    # Drug Tanimoto on GPU
    log.info(f"Computing Tanimoto ({len(ud)} x {len(train_drugs)}) on {DEVICE}...")
    t_sim = time.time()
    train_fp_t = torch.tensor(train_fp_mat, dtype=torch.float16, device=DEVICE)
    db_bits = train_fp_t.sum(1)
    CHUNK = 2000
    ds = np.empty((len(ud), len(train_drugs)), dtype=np.float32)
    for ci in range(0, len(ud), CHUNK):
        ce = min(ci + CHUNK, len(ud))
        q_t = torch.tensor(qfp[ci:ce], dtype=torch.float16, device=DEVICE)
        q_bits = q_t.sum(1, keepdim=True)
        inter = q_t @ train_fp_t.T
        sims = inter / torch.clamp(q_bits + db_bits.unsqueeze(0) - inter, min=1)
        ds[ci:ce] = sims.float().cpu().numpy()
    del train_fp_t, db_bits
    torch.cuda.empty_cache()
    log.info(f"Tanimoto: {time.time() - t_sim:.1f}s")

    # Mask exact matches
    with Pool(N_WORKERS) as pool:
        canon_results = list(pool.imap(canon, train_drugs, chunksize=1000))
    c_train = dict(zip(train_drugs, canon_results))
    canon_to_idx = {}
    for di, d in enumerate(train_drugs):
        dc = c_train.get(d, "")
        if dc:
            canon_to_idx.setdefault(dc, []).append(di)
    q_canon = [canon(s) for s in ud]
    n_masked = 0
    for qi, qc in enumerate(q_canon):
        if qc in canon_to_idx:
            for di in canon_to_idx[qc]:
                ds[qi, di] = -1
            n_masked += 1
    log.info(f"Exact-match drugs masked: {n_masked}/{len(ud)}")

    # Protein cosine similarity
    qn = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_emb_normed.T
    log.info(f"Similarities ready: ds {ds.shape}, ps {ps.shape}")

    di_arr = np.array([dm.get(s, -1) for s in eval_df.ligand_smiles.values])
    pi_arr = np.array([pm.get(p, -1) for p in eval_df.uniprot_id.values])

    # ── Run prot_knn k=1,5 ──
    results_all = []
    for k in [1, 5]:
        t1 = time.time()
        preds = run_prot_knn_vectorized(ds, ps, di_arr, pi_arr, k, INT_DENSE_T)
        elapsed = time.time() - t1
        col = f"prot_knn_k{k}"
        eval_df[col] = preds
        ci, auc, rmse = pp_metrics(eval_df, col)
        nan_pct = np.isnan(preds).mean() * 100
        log.info(f"\n  {col:<20s} CI={ci.mean():.4f}  AUROC={auc.mean() if len(auc) else 0:.4f}  "
                 f"RMSE={rmse.mean() if len(rmse) else 0:.4f}  NaN={nan_pct:.1f}%  t={elapsed:.0f}s")
        results_all.append({"method": col, "ci": ci.mean(),
                           "auroc": auc.mean() if len(auc) else np.nan,
                           "rmse": rmse.mean() if len(rmse) else np.nan,
                           "nan_pct": nan_pct, "n_proteins": eval_df.uniprot_id.nunique(),
                           "n_interactions": len(eval_df)})

        # Q1-Q4 by pKi quartile
        valid_preds = eval_df[~eval_df[col].isna()].copy()
        if len(valid_preds) > 20:
            q25, q50, q75 = np.percentile(valid_preds.pki.values, [25, 50, 75])
            valid_preds["pki_quartile"] = pd.cut(valid_preds.pki, bins=[-np.inf, q25, q50, q75, np.inf],
                                                  labels=["Q1", "Q2", "Q3", "Q4"])
            log.info(f"  pKi quartiles: Q1<={q25:.2f}, Q2<={q50:.2f}, Q3<={q75:.2f}, Q4>{q75:.2f}")
            for q in ["Q1", "Q2", "Q3", "Q4"]:
                sub = valid_preds[valid_preds.pki_quartile == q]
                if len(sub) >= 10:
                    ci_q, _, rmse_q = pp_metrics(sub, col)
                    log.info(f"    {q}: CI={ci_q.mean():.4f}, RMSE={rmse_q.mean() if len(rmse_q) else 0:.4f}, n={len(sub)}")

    # Save
    summary = pd.DataFrame(results_all)
    summary.to_csv(RESULTS_DIR / "prot_knn_dtc_summary.csv", index=False)
    eval_df.to_csv(RESULTS_DIR / "prot_knn_dtc_test.csv", index=False)

    log.info(f"\n{'='*70}")
    log.info(f"FINAL — prot_knn DTC In-Domain (test, {eval_df.uniprot_id.nunique()} proteins)")
    log.info(f"{'='*70}")
    for _, r in summary.iterrows():
        log.info(f"  {r['method']:<20s} CI={r['ci']:.4f}  AUROC={r['auroc']:.4f}  RMSE={r['rmse']:.4f}  NaN={r['nan_pct']:.1f}%")
    log.info(f"Total time: {time.time()-t_start:.0f}s")
    log.info("DONE")
