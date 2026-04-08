"""kNN-DTA vs ConciseAnchor on DTC data — in-domain evaluation.

Same DTC train/val/test split (80/10/10 by protein, seed 42) used for
ConciseAnchor training. kNN uses train split as retrieval pool.

Methods:
  - drug_knn (k=1,5,10): Tanimoto similarity on Morgan FP
  - prot_knn (k=1,5,10): cosine similarity on ESM2-650M
  - joint_knn (k=10,20): weighted drug + protein similarity
  - anchor_knn: nearest drug with strong binder (pKi >= 7)
  - ConciseAnchor-Bilinear: model inference (requires Raygun embeddings)

Metrics: per-protein CI, AUROC (hi=7, lo=5), RMSE, with Q1-Q4 anchor quartile breakdown.
Uses ALL CPU cores for FP computation and metrics, GPU for model inference.
"""
import os, sys, pickle, logging, time, random, json
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from pathlib import Path
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm
from functools import partial

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_WORKERS = min(cpu_count(), 64)
log.info(f"Using {N_WORKERS} CPU workers, device={DEVICE}")


# ── per-protein metrics (vectorized) ──────────────────────────────

def _ci_one_protein(yt, yp):
    """Vectorized CI for one protein using numpy broadcasting."""
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 3 or yp.std() < 1e-8:
        return 0.5 if len(yt) >= 3 else np.nan
    # Vectorized pairwise comparison
    dt = yt[:, None] - yt[None, :]
    dp = yp[:, None] - yp[None, :]
    # Upper triangle only
    idx = np.triu_indices(len(yt), k=1)
    dt_flat = dt[idx]
    dp_flat = dp[idx]
    nonzero = dt_flat != 0
    if nonzero.sum() == 0:
        return 0.5
    dt_nz = dt_flat[nonzero]
    dp_nz = dp_flat[nonzero]
    concordant = ((dt_nz * dp_nz) > 0).sum()
    tied = (dp_nz == 0).sum()
    total = len(dt_nz)
    return float((concordant + 0.5 * tied) / total)


def _auroc_one_protein(yt, yp, hi=7.0, lo=5.0):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    mask = (yt >= hi) | (yt <= lo)
    if mask.sum() < 2:
        return np.nan
    labels = (yt[mask] >= hi).astype(int)
    if len(set(labels)) < 2:
        return np.nan
    return roc_auc_score(labels, yp[mask])


def _rmse_one_protein(yt, yp):
    m = ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    if len(yt) < 3:
        return np.nan
    return np.sqrt(mean_squared_error(yt, yp))


def _compute_metrics_for_protein(args):
    """Worker function for parallel per-protein metrics."""
    uid, yt, yp = args
    return uid, _ci_one_protein(yt, yp), _auroc_one_protein(yt, yp), _rmse_one_protein(yt, yp)


def pp_metrics_parallel(df, col):
    """Compute per-protein CI, AUROC, RMSE in parallel."""
    tasks = []
    for uid in df.uniprot_id.unique():
        s = df[df.uniprot_id == uid]
        yt = s.pki.values
        yp = np.array(s[col].values, dtype=float)
        tasks.append((uid, yt, yp))

    with Pool(N_WORKERS) as pool:
        results = pool.map(_compute_metrics_for_protein, tasks)

    cis = [r[1] for r in results if not np.isnan(r[1])]
    aurocs = [r[2] for r in results if not np.isnan(r[2])]
    rmses = [r[3] for r in results if not np.isnan(r[3])]
    return np.array(cis), np.array(aurocs), np.array(rmses)


# ── canonical SMILES + Morgan FP ────────────────────────────────

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


# ── kNN methods (vectorized, no per-sample loops where possible) ─

def run_drug_knn(ds, ps, di_arr, pi_arr, k):
    """Vectorized drug-kNN: precompute top-k per unique drug, then broadcast."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    unique_di = np.unique(di_arr[di_arr >= 0])
    # Precompute top-k for all unique query drugs at once
    ds_sub = ds[unique_di]  # (n_unique, n_train)
    topk_all = np.argsort(ds_sub, axis=1)[:, -k:][:, ::-1]  # (n_unique, k)
    di_to_local = {d: i for i, d in enumerate(unique_di)}

    for qdi in unique_di:
        li = di_to_local[qdi]
        top_k = topk_all[li]
        top_sims = ds[qdi, top_k]
        valid = top_sims > 0
        top_k, top_sims = top_k[valid], top_sims[valid]
        if len(top_k) == 0:
            continue
        neighbor_pkis = INT_DENSE[top_k]
        mask_rows = di_arr == qdi
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qpi = pi_arr[ri]
            if qpi < 0:
                continue
            prot_s = ps[qpi]
            # Vectorized: for each neighbor, find best interacting protein
            has_int = neighbor_pkis > 0  # (k', n_prots)
            masked_ps = np.where(has_int, prot_s[None, :], -2)  # broadcast
            best_pis = np.argmax(masked_ps, axis=1)
            valid_k = masked_ps[np.arange(len(top_k)), best_pis] > 0
            if valid_k.any():
                pkis = neighbor_pkis[np.arange(len(top_k)), best_pis][valid_k]
                wts = top_sims[valid_k]
                preds[ri] = np.average(pkis, weights=wts)
    return preds


def run_prot_knn(ds, ps, di_arr, pi_arr, k):
    n = len(di_arr)
    preds = np.full(n, np.nan)
    INT_T = INT_DENSE.T  # (n_prots, n_drugs) — single copy
    unique_pi = np.unique(pi_arr[pi_arr >= 0])
    # Precompute top-k for all unique query proteins at once
    ps_sub = ps[unique_pi]
    topk_all = np.argsort(ps_sub, axis=1)[:, -k:][:, ::-1]
    pi_to_local = {p: i for i, p in enumerate(unique_pi)}

    for qpi in unique_pi:
        li = pi_to_local[qpi]
        top_k = topk_all[li]
        top_sims = ps[qpi, top_k]
        valid = top_sims > 0
        top_k, top_sims = top_k[valid], top_sims[valid]
        if len(top_k) == 0:
            continue
        neighbor_pkis = INT_T[top_k]  # (k', n_drugs)
        mask_rows = pi_arr == qpi
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qdi = di_arr[ri]
            if qdi < 0:
                continue
            drug_s = ds[qdi]
            has_int = neighbor_pkis > 0
            masked_ds = np.where(has_int, drug_s[None, :], -2)
            best_dis = np.argmax(masked_ds, axis=1)
            valid_k = masked_ds[np.arange(len(top_k)), best_dis] > 0
            if valid_k.any():
                pkis = neighbor_pkis[np.arange(len(top_k)), best_dis][valid_k]
                wts = top_sims[valid_k]
                preds[ri] = np.average(pkis, weights=wts)
    return preds


def run_joint_knn(ds, ps, di_arr, pi_arr, k, alpha=0.5):
    """Joint kNN — vectorized score computation."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    int_rows, int_cols = INT_MAT.nonzero()
    int_vals = np.array(INT_MAT[int_rows, int_cols]).flatten()
    n_int = len(int_rows)
    # Precompute drug sims for all interaction drugs: ds[:, int_rows] would be huge
    # Instead, process per-sample but use numpy vectorized ops
    for i in range(n):
        qdi, qpi = di_arr[i], pi_arr[i]
        if qdi < 0 or qpi < 0:
            continue
        scores = alpha * ds[qdi][int_rows] + (1 - alpha) * ps[qpi][int_cols]
        topk = np.argpartition(scores, -min(k, n_int))[-k:]
        s, p = scores[topk], int_vals[topk]
        m = s > 0
        if m.sum():
            preds[i] = np.average(p[m], weights=s[m])
    return preds


def run_anchor_knn(ds, ps, di_arr, pi_arr):
    """Anchor kNN — precompute top-10 per unique drug."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    tanimotos = np.full(n, np.nan)
    # Precompute top-10 for all unique query drugs
    unique_di = np.unique(di_arr[di_arr >= 0])
    ds_sub = ds[unique_di]
    topk_all = np.argsort(ds_sub, axis=1)[:, -10:][:, ::-1]
    di_to_local = {d: i for i, d in enumerate(unique_di)}

    for i in range(n):
        qdi, qpi = di_arr[i], pi_arr[i]
        if qdi < 0 or qpi < 0:
            continue
        li = di_to_local[qdi]
        top10 = topk_all[li]
        for bdi in top10:
            if ds[qdi, bdi] <= 0:
                break
            int_row = INT_DENSE[bdi]
            strong = int_row >= 7.0
            if not strong.any():
                continue
            candidate_ps = ps[qpi].copy()
            candidate_ps[~strong] = -2
            best_pi = np.argmax(candidate_ps)
            if candidate_ps[best_pi] > 0:
                preds[i] = int_row[best_pi]
                tanimotos[i] = ds[qdi, bdi]
                break
    return preds, tanimotos


# ── evaluate ────────────────────────────────────────────────────

def evaluate(name, eval_df, morgan, esm2_dict, train_canonical_set):
    log.info(f"\n{'=' * 70}")
    log.info(f"  {name}: {len(eval_df)} int, {eval_df.uniprot_id.nunique()} prot, "
             f"{eval_df.ligand_smiles.nunique()} drugs")

    eval_df = eval_df[eval_df.uniprot_id.isin(esm2_dict) & eval_df.ligand_smiles.isin(morgan)].copy()

    ud = sorted(eval_df.ligand_smiles.unique())
    up = sorted(eval_df.uniprot_id.unique())
    dm = {d: i for i, d in enumerate(ud)}
    pm = {p: i for i, p in enumerate(up)}

    qfp = np.array([morgan[d] for d in ud])
    qemb = np.array([esm2_dict[p] for p in up])

    t0 = time.time()
    # Drug similarities — GPU-accelerated Tanimoto in chunks
    log.info(f"  Computing Tanimoto similarity ({len(ud)} x {len(train_drugs)}) on {DEVICE}...")
    t_sim = time.time()
    train_fp_t = torch.tensor(train_fp_mat, dtype=torch.float16, device=DEVICE)
    db_bits_t = train_fp_t.sum(1)  # (n_train,)
    CHUNK = 2000  # chunk query drugs to fit GPU memory
    ds = np.empty((len(ud), len(train_drugs)), dtype=np.float32)
    for ci in range(0, len(ud), CHUNK):
        ce = min(ci + CHUNK, len(ud))
        q_t = torch.tensor(qfp[ci:ce], dtype=torch.float16, device=DEVICE)
        q_bits_t = q_t.sum(1, keepdim=True)  # (chunk, 1)
        inter_t = q_t @ train_fp_t.T  # (chunk, n_train) — GPU matmul
        sims_t = inter_t / torch.clamp(q_bits_t + db_bits_t.unsqueeze(0) - inter_t, min=1)
        ds[ci:ce] = sims_t.float().cpu().numpy()
    del train_fp_t, db_bits_t
    torch.cuda.empty_cache()
    log.info(f"  Tanimoto: {time.time() - t_sim:.1f}s")

    # Mask exact canonical matches — O(n) via reverse index
    canon_to_train_idx = {}
    for di, d in enumerate(train_drugs):
        dc = c_train.get(d, "")
        if dc:
            canon_to_train_idx.setdefault(dc, []).append(di)
    q_canon = [canon(s) for s in ud]
    n_masked = 0
    for qi, qc in enumerate(q_canon):
        if qc in canon_to_train_idx:
            for di in canon_to_train_idx[qc]:
                ds[qi, di] = -1
            n_masked += 1
    log.info(f"  Exact-match drugs masked: {n_masked}/{len(ud)}")

    # Protein similarities (cosine) — small, instant on CPU
    qn = qemb / (np.linalg.norm(qemb, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_emb_normed.T

    log.info(f"  All similarities: {time.time() - t0:.1f}s  (ds {ds.shape}, ps {ps.shape})")

    di_arr = np.array([dm.get(s, -1) for s in eval_df.ligand_smiles.values])
    pi_arr = np.array([pm.get(p, -1) for p in eval_df.uniprot_id.values])

    all_results = []
    header = f"  {'Method':<20s} {'CI':>7s} {'AUROC':>7s} {'RMSE':>7s} {'NaN%':>6s} {'t':>5s}"
    log.info(f"\n{header}")
    log.info(f"  {'-' * 60}")

    def run_and_report(method_name, preds, tanimotos=None):
        eval_df[method_name] = preds
        if tanimotos is not None:
            eval_df[f"{method_name}_tanimoto"] = tanimotos

        ci, auc, rmse = pp_metrics_parallel(eval_df, method_name)
        nan_pct = np.isnan(np.array(preds, dtype=float)).mean() * 100
        elapsed = time.time() - t0

        log.info(f"  {method_name:<20s} {ci.mean():7.4f} "
                 f"{auc.mean() if len(auc) else 0:7.4f} "
                 f"{rmse.mean() if len(rmse) else 0:7.4f} "
                 f"{nan_pct:6.1f} {elapsed:5.0f}s")

        all_results.append({"method": method_name, "ci": ci.mean(),
               "auroc": auc.mean() if len(auc) else np.nan,
               "rmse": rmse.mean() if len(rmse) else np.nan,
               "nan_pct": nan_pct, "n_proteins": eval_df.uniprot_id.nunique(),
               "n_interactions": len(eval_df)})
        sys.stdout.flush()

    # Drug kNN
    for k in [1, 5, 10]:
        t1 = time.time()
        preds = run_drug_knn(ds, ps, di_arr, pi_arr, k)
        log.info(f"  drug_knn_k{k} computed in {time.time()-t1:.1f}s")
        run_and_report(f"drug_knn_k{k}", preds)

    # Protein kNN
    for k in [1, 5, 10]:
        t1 = time.time()
        preds = run_prot_knn(ds, ps, di_arr, pi_arr, k)
        log.info(f"  prot_knn_k{k} computed in {time.time()-t1:.1f}s")
        run_and_report(f"prot_knn_k{k}", preds)

    # Joint kNN
    for k in [10, 20]:
        t1 = time.time()
        preds = run_joint_knn(ds, ps, di_arr, pi_arr, k)
        log.info(f"  joint_knn_k{k} computed in {time.time()-t1:.1f}s")
        run_and_report(f"joint_knn_k{k}", preds)

    # Anchor kNN
    t1 = time.time()
    preds, tanimotos = run_anchor_knn(ds, ps, di_arr, pi_arr)
    log.info(f"  anchor_knn computed in {time.time()-t1:.1f}s")
    run_and_report("anchor_knn", preds, tanimotos)

    # Q1-Q4 breakdown
    valid_tan = tanimotos[~np.isnan(tanimotos)]
    if len(valid_tan) > 10:
        q25, q50, q75 = np.percentile(valid_tan, [25, 50, 75])
        quartiles = np.where(np.isnan(tanimotos), "NA",
                    np.where(tanimotos <= q25, "Q1",
                    np.where(tanimotos <= q50, "Q2",
                    np.where(tanimotos <= q75, "Q3", "Q4"))))
        eval_df["anchor_tanimoto"] = tanimotos
        eval_df["pki_quartile"] = quartiles
        log.info(f"\n  Anchor Tanimoto quartiles: Q1<={q25:.3f}, Q2<={q50:.3f}, Q3<={q75:.3f}, Q4>{q75:.3f}")
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = eval_df[eval_df["pki_quartile"] == q]
            if len(sub) < 10:
                continue
            ci_q, _, rmse_q = pp_metrics_parallel(sub, "anchor_knn")
            log.info(f"    {q}: CI={ci_q.mean():.4f}, RMSE={rmse_q.mean() if len(rmse_q) else 0:.4f}, n={len(sub)}")

    log.info(f"  {'-' * 60}")
    out_path = RESULTS_DIR / f"knn_dtc_indomain_{name.lower().replace(' ', '_')}.csv"
    eval_df.to_csv(out_path, index=False)
    log.info(f"  Saved {out_path}")
    return pd.DataFrame(all_results), eval_df


# ── ConciseAnchor inference ─────────────────────────────────────

def compute_raygun_embeddings(proteins, seqs):
    """Compute ESM-2 → Raygun embeddings on GPU. Batch size 8 for max throughput."""
    import esm
    log.info(f"  Computing ESM-2 650M embeddings for {len(proteins)} proteins...")
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(DEVICE).eval()

    esm_embeddings = {}
    BS = 8  # batch size for GPU
    with torch.no_grad():
        for i in tqdm(range(0, len(proteins), BS), desc="ESM-2"):
            batch = [(u, seqs[u][:1022]) for u in proteins[i:i+BS] if u in seqs]
            if not batch:
                continue
            _, _, tokens = bc(batch)
            out = esm_model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            for j, (u, s) in enumerate(batch):
                esm_embeddings[u] = out["representations"][33][j:j+1, 1:len(s)+1, :].cpu()

    del esm_model
    torch.cuda.empty_cache()

    log.info(f"  Running Raygun encoder on {len(esm_embeddings)} proteins...")
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(DEVICE).eval()

    raygun_embs = {}
    with torch.no_grad():
        for uid, emb in tqdm(esm_embeddings.items(), desc="Raygun"):
            ray_enc = raymodel.encoder(emb.to(DEVICE)).squeeze().cpu()
            raygun_embs[uid] = ray_enc

    del raymodel, esm_embeddings
    torch.cuda.empty_cache()
    return raygun_embs


def run_concise_anchor_inference(eval_df, train_df_full, morgan):
    """Run ConciseAnchor-Bilinear on test set. Computes Raygun if needed."""
    raygun_cache = RESULTS_DIR / "raygun_embeddings.pt"
    if not raygun_cache.exists():
        if not torch.cuda.is_available():
            log.info("  No Raygun cache and no GPU — skipping ConciseAnchor")
            return None
        seq_path = PROJECT / "data" / "processed" / "merged_sequences.json"
        if not seq_path.exists():
            log.info(f"  {seq_path} missing — skipping")
            return None
        seqs = json.load(open(seq_path))
        all_proteins = sorted(set(eval_df.uniprot_id.unique()) |
                              set(train_df_full.uniprot_id.unique()))
        all_proteins = [u for u in all_proteins if u in seqs]
        log.info(f"  Computing Raygun for {len(all_proteins)} proteins on GPU...")
        raygun_embs = compute_raygun_embeddings(all_proteins, seqs)
        torch.save(raygun_embs, raygun_cache)
        log.info(f"  Saved {len(raygun_embs)} Raygun to {raygun_cache}")

    log.info("  Loading Raygun embeddings...")
    raygun_embs = torch.load(raygun_cache, map_location="cpu", weights_only=False)

    # Use morgan dict directly as FPs
    fp_dict = morgan

    # Load model
    model_path = PROJECT / "models" / "concise_anchor_bilinear_5ep.pt"
    if not model_path.exists():
        model_path = PROJECT / "models" / "concise_anchor_bilinear_dtc" / "best_model.pt"
    if not model_path.exists():
        log.info("  Model checkpoint not found — skipping")
        return None

    from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear
    model = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256,
                                   n_codes=3, dropout=0.2).to(DEVICE)
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()
    log.info(f"  Loaded ConciseAnchor from {model_path}")

    # Build anchors (strongest binder per drug, pKi >= 7)
    drug_to_anchor, drug_to_second = {}, {}
    for smi, grp in train_df_full.groupby("ligand_smiles"):
        s = grp.sort_values("pki", ascending=False)
        uids, pkis = s.uniprot_id.values, s.pki.values
        if pkis[0] >= 7.0 and uids[0] in raygun_embs:
            drug_to_anchor[smi] = (uids[0], pkis[0])
            if len(uids) > 1 and uids[1] in raygun_embs:
                drug_to_second[smi] = (uids[1], pkis[1])
    log.info(f"  Anchors: {len(drug_to_anchor)} drugs with pKi >= 7")

    # Batch inference
    preds = np.full(len(eval_df), np.nan)
    batch_fps, batch_anc, batch_qry, batch_idx = [], [], [], []
    BATCH = 1024  # large batches for GPU

    for idx, (_, row) in enumerate(eval_df.iterrows()):
        uid, smi = row["uniprot_id"], row["ligand_smiles"]
        if smi not in drug_to_anchor or smi not in fp_dict or uid not in raygun_embs:
            continue
        au, ap = drug_to_anchor[smi]
        if au == uid:
            if smi not in drug_to_second:
                continue
            au, ap = drug_to_second[smi]
        if au not in raygun_embs:
            continue
        batch_fps.append(np.array(fp_dict[smi], dtype=np.float32))
        batch_anc.append(raygun_embs[au])
        batch_qry.append(raygun_embs[uid])
        batch_idx.append(idx)

        if len(batch_fps) >= BATCH:
            with torch.no_grad():
                fp_t = torch.tensor(np.array(batch_fps)).to(DEVICE)
                anc_t = torch.stack(batch_anc).to(DEVICE)
                qry_t = torch.stack(batch_qry).to(DEVICE)
                pred = model(fp_t, anc_t, qry_t).cpu().numpy()
            for bi, pi in enumerate(batch_idx):
                preds[pi] = pred[bi]
            batch_fps, batch_anc, batch_qry, batch_idx = [], [], [], []

    if batch_fps:
        with torch.no_grad():
            fp_t = torch.tensor(np.array(batch_fps)).to(DEVICE)
            anc_t = torch.stack(batch_anc).to(DEVICE)
            qry_t = torch.stack(batch_qry).to(DEVICE)
            pred = model(fp_t, anc_t, qry_t).cpu().numpy()
        for bi, pi in enumerate(batch_idx):
            preds[pi] = pred[bi]

    n_pred = (~np.isnan(preds)).sum()
    log.info(f"  ConciseAnchor predicted: {n_pred}/{len(eval_df)} ({n_pred/len(eval_df)*100:.1f}%)")
    return preds


# ── main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info(f"\n{'=' * 70}")
    log.info(f"  kNN-DTA vs ConciseAnchor — DTC In-Domain Evaluation")
    log.info(f"  {N_WORKERS} CPU cores, GPU={DEVICE}")
    log.info(f"{'=' * 70}")

    # Load DTC
    log.info("Loading DTC data...")
    dtc_path = PROJECT / "embeddings_model_files" / "dtc_training_interactions.csv"
    if not dtc_path.exists():
        dtc_path = PROJECT / "data" / "processed" / "dtc_training_interactions.csv"
    dtc = pd.read_csv(dtc_path)
    log.info(f"DTC: {len(dtc)} int, {dtc.uniprot_id.nunique()} prot, {dtc.ligand_smiles.nunique()} drugs")

    # ESM2
    log.info("Loading ESM2-650M embeddings...")
    esm2_path = PROJECT / "embeddings_model_files" / "esm2_650m_dtc.pt"
    esm2_raw = torch.load(esm2_path, map_location="cpu", weights_only=False)
    esm2_dict = {}
    for k, v in esm2_raw.items():
        t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
        esm2_dict[k] = t.numpy() if t.dim() == 1 else t.mean(dim=0).numpy()
    del esm2_raw
    log.info(f"ESM2: {len(esm2_dict)} proteins")

    # Split
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
    val_df = dtc_filt[dtc_filt.uniprot_id.isin(val_prots)].copy()
    test_df = dtc_filt[dtc_filt.uniprot_id.isin(test_prots)].copy()
    log.info(f"Train: {len(train_df)} ({train_df.uniprot_id.nunique()} prot)")
    log.info(f"Val:   {len(val_df)} ({val_df.uniprot_id.nunique()} prot)")
    log.info(f"Test:  {len(test_df)} ({test_df.uniprot_id.nunique()} prot)")

    # Morgan FPs — parallel with all cores
    FP_CACHE = RESULTS_DIR / "dtc_indomain_morgan_fp.pkl"
    if FP_CACHE.exists():
        log.info("Loading cached Morgan FPs...")
        with open(FP_CACHE, "rb") as f:
            morgan = pickle.load(f)
    else:
        drugs = sorted(set(train_df.ligand_smiles) | set(test_df.ligand_smiles) | set(val_df.ligand_smiles))
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
    val_df = val_df[val_df.ligand_smiles.isin(morgan)].copy()
    log.info(f"After FP filter — Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Index structures
    log.info("Building index structures...")
    train_drugs = sorted(set(train_df.ligand_smiles))
    train_drug_idx = {d: i for i, d in enumerate(train_drugs)}
    train_fp_mat = np.array([morgan[d] for d in train_drugs])
    log.info(f"Train FP matrix: {train_fp_mat.shape}")

    train_prots_list = sorted(set(train_df.uniprot_id))
    train_prot_idx = {p: i for i, p in enumerate(train_prots_list)}
    train_emb_mat = np.array([esm2_dict[p] for p in train_prots_list])
    train_emb_norms = np.linalg.norm(train_emb_mat, axis=1, keepdims=True) + 1e-10
    train_emb_normed = train_emb_mat / train_emb_norms
    log.info(f"Train Emb matrix: {train_emb_mat.shape}")

    n_drugs = len(train_drugs)
    n_prots = len(train_prots_list)

    # Interaction matrix (vectorized build)
    log.info("Building interaction matrix...")
    di_col = train_df.ligand_smiles.map(train_drug_idx).values
    pi_col = train_df.uniprot_id.map(train_prot_idx).values
    pki_col = train_df.pki.values
    # Use pandas groupby for max dedup
    tmp = pd.DataFrame({"di": di_col, "pi": pi_col, "pki": pki_col})
    tmp = tmp.groupby(["di", "pi"]).pki.max().reset_index()
    INT_MAT = sp.csr_matrix((tmp.pki.values, (tmp.di.values, tmp.pi.values)),
                             shape=(n_drugs, n_prots))
    INT_DENSE = INT_MAT.toarray()
    log.info(f"Interaction matrix: {INT_DENSE.shape}, nnz={INT_MAT.nnz}")

    # Canonical SMILES — parallel
    log.info("Canonicalizing train drug SMILES...")
    with Pool(N_WORKERS) as pool:
        canon_results = list(tqdm(pool.imap(canon, train_drugs, chunksize=1000),
                                  total=len(train_drugs), desc="Canon"))
    c_train = dict(zip(train_drugs, canon_results))
    train_canonical_set = set(c_train.values())

    # ── Run evaluation ──
    summary_df, test_results = evaluate("test", test_df, morgan, esm2_dict, train_canonical_set)

    # ConciseAnchor inference
    ca_preds = run_concise_anchor_inference(test_results, train_df, morgan)
    if ca_preds is not None:
        test_results["concise_anchor"] = ca_preds
        ci, auc, rmse = pp_metrics_parallel(test_results, "concise_anchor")
        nan_pct = np.isnan(ca_preds).mean() * 100
        log.info(f"\n  {'concise_anchor':<20s} {ci.mean():7.4f} "
                 f"{auc.mean() if len(auc) else 0:7.4f} "
                 f"{rmse.mean() if len(rmse) else 0:7.4f} {nan_pct:6.1f}")
        summary_df = pd.concat([summary_df, pd.DataFrame([{
            "method": "concise_anchor", "ci": ci.mean(),
            "auroc": auc.mean() if len(auc) else np.nan,
            "rmse": rmse.mean() if len(rmse) else np.nan,
            "nan_pct": nan_pct, "n_proteins": test_results.uniprot_id.nunique(),
            "n_interactions": len(test_results)}])], ignore_index=True)

        # Save with ConciseAnchor predictions
        test_results.to_csv(RESULTS_DIR / "knn_dtc_indomain_test.csv", index=False)

    # Val set
    log.info("\n\n")
    val_summary, _ = evaluate("val", val_df, morgan, esm2_dict, train_canonical_set)

    # Save summary
    summary_df.to_csv(RESULTS_DIR / "knn_vs_concise_dtc_summary.csv", index=False)

    # Final table
    log.info(f"\n{'=' * 70}")
    log.info(f"  FINAL COMPARISON — DTC Test Set ({test_df.uniprot_id.nunique()} proteins)")
    log.info(f"{'=' * 70}")
    log.info(f"  {'Method':<20s} {'CI':>7s} {'AUROC':>7s} {'RMSE':>7s} {'NaN%':>6s}")
    log.info(f"  {'-' * 55}")
    for _, r in summary_df.iterrows():
        log.info(f"  {r['method']:<20s} {r['ci']:7.4f} "
                 f"{r['auroc']:7.4f} {r['rmse']:7.4f} {r['nan_pct']:6.1f}")
    log.info(f"{'=' * 70}")
    log.info("DONE")
