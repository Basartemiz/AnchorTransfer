"""kNN baselines on GLASS2 novel proteins — BDB retrieval pool.

Paper-fair: all drugs in test, exact-match drugs masked in retrieval,
proteins with >50% identity to BDB excluded from test set.
Vectorized with dense interaction matrix.
"""
import os, sys, pickle, logging, time
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from itertools import combinations
from sklearn.metrics import roc_auc_score, mean_squared_error
from collections import defaultdict
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── metrics ─────────────────────────────────────────────────────

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

from rdkit import Chem
from rdkit.Chem import AllChem

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except: return s

def _compute_fp(s):
    m = Chem.MolFromSmiles(s)
    if m:
        return s, np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)
    return s, None

# ── Load BDB ────────────────────────────────────────────────────

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
log.info(f"FP matrix: {bdb_fp_mat.shape} ({bdb_fp_mat.nbytes/1e9:.1f} GB)")

bdb_prots = sorted(set(bdb.uniprot_id))
bdb_prot_idx = {p: i for i, p in enumerate(bdb_prots)}
bdb_emb_mat = np.array([raygun_pooled[p] for p in bdb_prots])
bdb_emb_norms = np.linalg.norm(bdb_emb_mat, axis=1, keepdims=True) + 1e-10
bdb_emb_normed = bdb_emb_mat / bdb_emb_norms
log.info(f"Emb matrix: {bdb_emb_mat.shape}")

n_drugs = len(bdb_drugs)
n_prots = len(bdb_prots)

# Sparse interaction matrix
log.info("Building interaction matrix...")
pair_max = {}
for _, r in bdb.iterrows():
    di, pi = bdb_drug_idx[r.ligand_smiles], bdb_prot_idx[r.uniprot_id]
    key = (di, pi)
    if key not in pair_max or r.pki > pair_max[key]:
        pair_max[key] = r.pki
rows_d = [k[0] for k in pair_max]
cols_p = [k[1] for k in pair_max]
vals = list(pair_max.values())
INT_MAT = sp.csr_matrix((vals, (rows_d, cols_p)), shape=(n_drugs, n_prots))
INT_DENSE = INT_MAT.toarray()
log.info(f"Interaction matrix: {INT_DENSE.shape}, nnz={INT_MAT.nnz} ({INT_DENSE.nbytes/1e9:.1f} GB)")

# Canonical SMILES
log.info("Canonicalizing BDB drug SMILES...")
c_bdb = {d: canon(d) for d in bdb_drugs}
bdb_canonical_set = set(c_bdb.values())
log.info("BDB setup done")

# ── similarity helpers ──────────────────────────────────────────

def drug_sims(query_fps, mask_canonical=None, query_smiles=None):
    inter = query_fps @ bdb_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = bdb_fp_mat.sum(1, keepdims=True).T
    sims = inter / np.maximum(q_bits + db_bits - inter, 1)
    if mask_canonical and query_smiles:
        q_canon = [canon(s) for s in query_smiles]
        for qi, qc in enumerate(q_canon):
            for di, d in enumerate(bdb_drugs):
                if c_bdb.get(d) == qc:
                    sims[qi, di] = -1
    return sims

def prot_sims(query_embs, exclude=None):
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    sims = qn @ bdb_emb_normed.T
    if exclude:
        for p in exclude:
            if p in bdb_prot_idx:
                sims[:, bdb_prot_idx[p]] = -1
    return sims

# ── Vectorized kNN ──────────────────────────────────────────────

def run_drug_knn_vec(ds, ps, di_arr, pi_arr, k):
    n = len(di_arr)
    preds = np.full(n, np.nan)
    unique_di = np.unique(di_arr[di_arr >= 0])
    for qdi in unique_di:
        mask_rows = di_arr == qdi
        top_k = np.argsort(ds[qdi])[-k:][::-1]
        top_sims = ds[qdi, top_k]
        valid = top_sims > 0
        top_k, top_sims = top_k[valid], top_sims[valid]
        if len(top_k) == 0: continue
        neighbor_pkis = INT_DENSE[top_k]
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qpi = pi_arr[ri]
            if qpi < 0: continue
            prot_s = ps[qpi]
            best_pkis, best_wts = [], []
            for ki, bdi in enumerate(top_k):
                int_row = neighbor_pkis[ki]
                has_int = int_row > 0
                if not has_int.any(): continue
                masked = np.where(has_int, prot_s, -2)
                best_pi = np.argmax(masked)
                if masked[best_pi] <= 0: continue
                best_pkis.append(int_row[best_pi])
                best_wts.append(top_sims[ki])
            if best_pkis:
                preds[ri] = np.average(best_pkis, weights=best_wts)
    return preds

def run_prot_knn_vec(ds, ps, di_arr, pi_arr, k):
    n = len(di_arr)
    preds = np.full(n, np.nan)
    unique_pi = np.unique(pi_arr[pi_arr >= 0])
    for qpi in unique_pi:
        mask_rows = pi_arr == qpi
        top_k = np.argsort(ps[qpi])[-k:][::-1]
        top_sims = ps[qpi, top_k]
        valid = top_sims > 0
        top_k, top_sims = top_k[valid], top_sims[valid]
        if len(top_k) == 0: continue
        neighbor_pkis = [INT_DENSE[:, bpi] for bpi in top_k]
        row_indices = np.where(mask_rows)[0]
        for ri in row_indices:
            qdi = di_arr[ri]
            if qdi < 0: continue
            drug_s = ds[qdi]
            best_pkis, best_wts = [], []
            for ki, bpi in enumerate(top_k):
                int_col = neighbor_pkis[ki]
                has_int = int_col > 0
                if not has_int.any(): continue
                masked = np.where(has_int, drug_s, -2)
                best_di = np.argmax(masked)
                if masked[best_di] <= 0: continue
                best_pkis.append(int_col[best_di])
                best_wts.append(top_sims[ki])
            if best_pkis:
                preds[ri] = np.average(best_pkis, weights=best_wts)
    return preds

def run_joint_knn_vec(ds, ps, di_arr, pi_arr, k, alpha=0.5):
    n = len(di_arr)
    preds = np.full(n, np.nan)
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
                   mask_canonical=bdb_canonical_set if mask_canonical_drugs else None,
                   query_smiles=ud if mask_canonical_drugs else None)
    ps = prot_sims(qemb, exclude=exclude_prots)
    log.info(f"  Sims: {time.time()-t0:.1f}s  (ds {ds.shape}, ps {ps.shape})")

    if mask_canonical_drugs:
        n_masked = sum(1 for d in ud if canon(d) in bdb_canonical_set)
        log.info(f"  Exact-match drugs masked in retrieval: {n_masked}/{len(ud)}")

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
        run_and_report(f'prot_knn_k{k}', run_prot_knn_vec(ds, ps, di_arr, pi_arr, k))
    for k in [10]:
        run_and_report(f'joint_knn_k{k}', run_joint_knn_vec(ds, ps, di_arr, pi_arr, k))
    run_and_report('anchor_knn', run_anchor_knn_vec(ds, ps, di_arr, pi_arr))

    log.info(f"  {'-'*52}")
    log.info(f"  All done: {time.time()-t0:.0f}s")

    # ── Quartile analysis (by pKi range) ──
    methods_list = [c for c in df.columns if 'knn' in c or 'anchor' in c]
    if methods_list:
        pki_vals = df.pki.values
        q_edges = np.quantile(pki_vals, [0, 0.25, 0.5, 0.75, 1.0])
        q_labels = ['Q1 weakest', 'Q2', 'Q3', 'Q4 strongest']
        df['pki_quartile'] = pd.cut(pki_vals, bins=q_edges, labels=q_labels, include_lowest=True)
        log.info(f"\n  ── pKi Quartile breakdown ──")
        for m in methods_list:
            log.info(f"  {m}:")
            for ql in q_labels:
                qdf = df[df.pki_quartile == ql]
                if len(qdf) < 5: continue
                ci = pp_ci(qdf, m)
                log.info(f"    {ql:<15s}  CI={ci.mean():.3f}  (n={len(qdf)})")

    # ── Family analysis ──
    if family_map:
        df['pfam'] = df.uniprot_id.map(lambda u: family_map.get(u, 'Unknown'))
        df['gpcr_class'] = df.uniprot_id.map(lambda u: gpcr_map.get(u, 'Unknown'))
        # Top families by interaction count
        fam_counts = df.pfam.value_counts()
        top_fams = [f for f in fam_counts.index if f != 'Unknown'][:10]
        if top_fams:
            log.info(f"\n  ── Family breakdown (top {len(top_fams)} Pfam) ──")
            for m in methods_list[:3]:  # top 3 methods only
                log.info(f"  {m}:")
                for fam in top_fams:
                    fdf = df[df.pfam == fam]
                    if fdf.uniprot_id.nunique() < 3: continue
                    ci = pp_ci(fdf, m)
                    log.info(f"    {fam:<20s}  CI={ci.mean():.3f}  (n={len(fdf)}, {fdf.uniprot_id.nunique()} prot)")

        # GPCR class breakdown
        gpcr_classes = [c for c in df.gpcr_class.unique() if c != 'Unknown']
        if gpcr_classes:
            log.info(f"\n  ── GPCR class breakdown ──")
            for m in methods_list[:3]:
                log.info(f"  {m}:")
                for gc in sorted(gpcr_classes):
                    gdf = df[df.gpcr_class == gc]
                    if gdf.uniprot_id.nunique() < 3: continue
                    ci = pp_ci(gdf, m)
                    log.info(f"    Class {gc:<10s}  CI={ci.mean():.3f}  (n={len(gdf)}, {gdf.uniprot_id.nunique()} prot)")

    df.to_csv(f"results/knn_glass_{name.lower().replace(' ','_')}.csv", index=False)
    log.info(f"  Saved results/knn_glass_{name.lower().replace(' ','_')}.csv")


# ── GLASS2 ──────────────────────────────────────────────────────

log.info("\n" + "="*60 + "\nLoading GLASS2...")
glass = pd.read_csv("data/raw/glass/glass2_ki_interactions.csv")
log.info(f"GLASS2: {len(glass)} int, {glass.uniprot_id.nunique()} prot, "
         f"{glass.ligand_smiles.nunique()} drugs")

# Protein overlap: direct UniProt + homologs
excl_prots_direct = set(glass.uniprot_id) & set(bdb_prots)

# Load both 50% and 30% homolog lists
HOMOLOG_50 = "results/glass_bdb_homologs_50.txt"
HOMOLOG_30 = "results/glass_bdb_homologs_30.txt"
homologs_50 = set()
if os.path.exists(HOMOLOG_50):
    homologs_50 = set(open(HOMOLOG_50).read().strip().split("\n"))

# Compute 30% homologs if not cached
if os.path.exists(HOMOLOG_30):
    homologs_30 = set(open(HOMOLOG_30).read().strip().split("\n"))
else:
    log.info("Computing 30% identity homologs (GLASS2 vs BDB)...")
    import json
    seq_path = "data/raw/glass/glass2_sequences.json"
    glass_seqs = json.load(open(seq_path)) if os.path.exists(seq_path) else {}
    # BDB sequences
    bdb_seq_path = "data/processed/bdb_sequences.json"
    bdb_seqs = json.load(open(bdb_seq_path)) if os.path.exists(bdb_seq_path) else {}
    if not bdb_seqs:
        # Try loading from protein file
        bdb_prot_path = "data/processed/bdb_proteins.json"
        if os.path.exists(bdb_prot_path):
            bdb_seqs = json.load(open(bdb_prot_path))

    if glass_seqs and bdb_seqs:
        from difflib import SequenceMatcher
        homologs_30 = set()
        bdb_seq_list = [(uid, seq) for uid, seq in bdb_seqs.items() if uid in set(bdb_prots)]
        glass_test = [(uid, seq) for uid, seq in glass_seqs.items() if uid not in excl_prots_direct]
        log.info(f"  Comparing {len(glass_test)} GLASS2 vs {len(bdb_seq_list)} BDB proteins...")
        for gi, (g_uid, g_seq) in enumerate(glass_test):
            for b_uid, b_seq in bdb_seq_list:
                # Quick length filter
                ratio = len(g_seq) / len(b_seq) if len(b_seq) > 0 else 0
                if ratio < 0.5 or ratio > 2.0: continue
                identity = SequenceMatcher(None, g_seq, b_seq).ratio()
                if identity >= 0.30:
                    homologs_30.add(g_uid)
                    break
            if (gi+1) % 50 == 0:
                log.info(f"    {gi+1}/{len(glass_test)}, found {len(homologs_30)} homologs")
        with open(HOMOLOG_30, "w") as f:
            f.write("\n".join(sorted(homologs_30)))
        log.info(f"  Saved {len(homologs_30)} homologs at 30% to {HOMOLOG_30}")
    else:
        log.info("  No sequence files found, using 50% homologs only")
        homologs_30 = homologs_50

excl_prots_30 = excl_prots_direct | homologs_30
excl_prots_50 = excl_prots_direct | homologs_50
log.info(f"GLASS2 protein exclusion: {len(excl_prots_direct)} direct, "
         f"{len(homologs_50)} homologs@50%, {len(homologs_30)} homologs@30%")

# Raygun for GLASS2 proteins
GLASS_EMB_CACHE = "results/raygun_glass_pooled.pt"
if os.path.exists(GLASS_EMB_CACHE):
    glass_emb = torch.load(GLASS_EMB_CACHE, map_location="cpu", weights_only=False)
    glass_emb = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in glass_emb.items()}
    log.info(f"Loaded GLASS2 Raygun cache: {len(glass_emb)} proteins")
else:
    log.info("Computing Raygun for GLASS2 proteins...")
    # Load sequences
    import json
    seq_path = "data/raw/glass/glass2_sequences.json"
    if os.path.exists(seq_path):
        glass_seqs = json.load(open(seq_path))
    else:
        glass_seqs = {}
        for _, r in glass.drop_duplicates("uniprot_id").iterrows():
            if hasattr(r, "protein_sequence") and pd.notna(r.protein_sequence):
                glass_seqs[r.uniprot_id] = r.protein_sequence

    # ESM2
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()

    esm_embs = {}
    items = [(u, s) for u, s in glass_seqs.items() if u not in excl_prots_30]
    log.info(f"  Computing ESM2 for {len(items)} novel GLASS2 proteins (excl 30% homologs)...")
    for i in range(0, len(items), 8):
        batch = [(u, s[:1022]) for u, s in items[i:i+8]]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
        for j, (u, s) in enumerate(batch):
            esm_embs[u] = out["representations"][33][j, 1:len(s)+1, :].cpu()
        if (i+8) % 100 < 8:
            log.info(f"    ESM2: {min(i+8, len(items))}/{len(items)}")

    del esm_model; torch.cuda.empty_cache()

    raygun_model, _, _ = torch.hub.load(
        "rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)

    glass_emb = {}
    for uid, emb in esm_embs.items():
        with torch.no_grad():
            out = raygun_model(emb.unsqueeze(0).to(DEVICE))
            proj = out[0] if isinstance(out, tuple) else out
            glass_emb[uid] = proj.squeeze(0).mean(dim=0).cpu().numpy()

    del raygun_model; torch.cuda.empty_cache()
    torch.save(glass_emb, GLASS_EMB_CACHE)
    log.info(f"Cached {len(glass_emb)} GLASS2 Raygun embeddings")

all_emb = {**raygun_pooled, **glass_emb}

# GLASS2 FPs
all_fp = dict(morgan)
GLASS_FP_CACHE = "results/concise_glass_fp.pkl"
if os.path.exists(GLASS_FP_CACHE):
    all_fp.update(pickle.load(open(GLASS_FP_CACHE, "rb")))

missing = set(glass.ligand_smiles.unique()) - set(all_fp.keys())
if missing:
    log.info(f"Computing {len(missing)} missing GLASS2 Morgan FPs ({cpu_count()} cores)...")
    with Pool(min(cpu_count(), 32)) as pool:
        results = pool.map(_compute_fp, list(missing), chunksize=500)
    for s, fp in results:
        if fp is not None: all_fp[s] = fp
    log.info(f"  Computed {sum(1 for _,fp in results if fp is not None)} FPs")

# ── Load family annotations ─────────────────────────────────────
import json as _json
family_map = {}
gpcr_map = {}
PROT_META = "data/raw/glass/protein.json"
if os.path.exists(PROT_META):
    pmeta = _json.load(open(PROT_META))
    for uid, info in pmeta.items():
        if not isinstance(info, dict): continue
        pfam = info.get('pfam_ids')
        if isinstance(pfam, list) and pfam:
            family_map[uid] = pfam[0]
        elif pfam:
            family_map[uid] = str(pfam)
        gc = info.get('gpcr_class')
        if gc:
            gpcr_map[uid] = str(gc).strip()
    log.info(f"Family annotations: {len(family_map)} pfam, {len(gpcr_map)} gpcr_class")
else:
    log.info("No protein.json found, skipping family analysis")

# Novel proteins — 30% identity threshold (strictest)
evaluate("GLASS2_novel_30", glass, all_emb, all_fp,
         mask_canonical_drugs=True, exclude_prots=excl_prots_30)

# Novel proteins — 50% identity threshold
evaluate("GLASS2_novel_50", glass, all_emb, all_fp,
         mask_canonical_drugs=True, exclude_prots=excl_prots_50)

# ═══════════════════════════════════════════════════════════════
# DTC → GLASS2 evaluation (paper's training DB)
# ═══════════════════════════════════════════════════════════════

log.info("\n" + "="*60 + "\nSwitching to DTC retrieval pool...")
import random as _random

dtc_data = pd.read_csv("data/processed/dtc_training_interactions.csv")
dtc_all_prots = sorted(set(dtc_data.uniprot_id))
_random.seed(42); _random.shuffle(dtc_all_prots)
_nt = max(1, int(len(dtc_all_prots) * 0.1))
_nv = max(1, int(len(dtc_all_prots) * 0.1))
dtc_train_prots = set(dtc_all_prots[_nt + _nv:])
dtc_train = dtc_data[dtc_data.uniprot_id.isin(dtc_train_prots)].copy()

# ESM2 embeddings for DTC
esm2_dtc = torch.load("data/processed/esm2_650m_dtc.pt",
                       map_location="cpu", weights_only=False)
dtc_esm = {}
for k, v in esm2_dtc.items():
    t = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    dtc_esm[k] = t.numpy() if t.dim() == 1 else t.mean(dim=0).numpy()
del esm2_dtc

dtc_train = dtc_train[dtc_train.uniprot_id.isin(dtc_esm)].copy()
log.info(f"DTC train: {len(dtc_train)} int, {dtc_train.uniprot_id.nunique()} prot, "
         f"{dtc_train.ligand_smiles.nunique()} drugs")

# DTC Morgan FPs
DTC_FP_CACHE = "results/dtc_train_morgan_fp.pkl"
if os.path.exists(DTC_FP_CACHE):
    dtc_morgan = pickle.load(open(DTC_FP_CACHE, "rb"))
else:
    dtc_drug_list = sorted(dtc_train.ligand_smiles.unique())
    log.info(f"Computing DTC Morgan FPs ({len(dtc_drug_list)} drugs)...")
    with Pool(min(cpu_count(), 32)) as pool:
        res = pool.map(_compute_fp, dtc_drug_list, chunksize=500)
    dtc_morgan = {s: fp for s, fp in res if fp is not None}
    pickle.dump(dtc_morgan, open(DTC_FP_CACHE, "wb"))

dtc_train = dtc_train[dtc_train.ligand_smiles.isin(dtc_morgan)].copy()

# Rebuild index structures for DTC
bdb_drugs = sorted(set(dtc_train.ligand_smiles))
bdb_drug_idx = {d: i for i, d in enumerate(bdb_drugs)}
bdb_fp_mat = np.array([dtc_morgan[d] for d in bdb_drugs])

bdb_prots = sorted(set(dtc_train.uniprot_id))
bdb_prot_idx = {p: i for i, p in enumerate(bdb_prots)}
bdb_emb_mat = np.array([dtc_esm[p] for p in bdb_prots])
bdb_emb_norms = np.linalg.norm(bdb_emb_mat, axis=1, keepdims=True) + 1e-10
bdb_emb_normed = bdb_emb_mat / bdb_emb_norms

# Rebuild interaction matrix
pair_max2 = {}
for _, r in dtc_train.iterrows():
    di, pi = bdb_drug_idx[r.ligand_smiles], bdb_prot_idx[r.uniprot_id]
    key = (di, pi)
    if key not in pair_max2 or r.pki > pair_max2[key]:
        pair_max2[key] = r.pki
rd = [k[0] for k in pair_max2]; cp = [k[1] for k in pair_max2]; vv = list(pair_max2.values())
INT_MAT = sp.csr_matrix((vv, (rd, cp)), shape=(len(bdb_drugs), len(bdb_prots)))
INT_DENSE = INT_MAT.toarray()
log.info(f"DTC interaction matrix: {INT_DENSE.shape}, nnz={INT_MAT.nnz}")

# Canonical
c_bdb = {d: canon(d) for d in bdb_drugs}
bdb_canonical_set = set(c_bdb.values())

# ESM2 for GLASS2 proteins (for DTC comparison — need ESM2 not Raygun)
GLASS_ESM_CACHE = "results/esm2_glass_pooled.pt"
if os.path.exists(GLASS_ESM_CACHE):
    glass_esm = torch.load(GLASS_ESM_CACHE, map_location="cpu", weights_only=False)
    glass_esm = {k: (v.numpy() if isinstance(v, torch.Tensor) else v) for k, v in glass_esm.items()}
    log.info(f"Loaded GLASS2 ESM2 cache: {len(glass_esm)} proteins")
else:
    log.info("Computing ESM2-650M for GLASS2 proteins...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(DEVICE)
    bc = alphabet.get_batch_converter()

    glass_seqs_j = _json.load(open("data/raw/glass/glass2_sequences.json"))
    items = [(u, s) for u, s in glass_seqs_j.items() if u not in excl_prots_30]
    log.info(f"  {len(items)} novel proteins to embed...")
    glass_esm = {}
    for i in range(0, len(items), 8):
        batch = [(u, s[:1022]) for u, s in items[i:i+8]]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
        for j, (u, s) in enumerate(batch):
            glass_esm[u] = out["representations"][33][j, 1:len(s)+1, :].mean(dim=0).cpu().numpy()
        if (i+8) % 100 < 8:
            log.info(f"    ESM2: {min(i+8, len(items))}/{len(items)}")
    del esm_model; torch.cuda.empty_cache()
    torch.save(glass_esm, GLASS_ESM_CACHE)
    log.info(f"Cached {len(glass_esm)} GLASS2 ESM2 embeddings")

dtc_all_emb = {**dtc_esm, **glass_esm}

# DTC protein overlap with GLASS2
dtc_excl_p = set(glass.uniprot_id) & set(bdb_prots)
log.info(f"DTC-GLASS2 protein overlap: {len(dtc_excl_p)}")

# FPs for GLASS2 drugs
dtc_all_fp = dict(dtc_morgan)
for s in set(glass.ligand_smiles.unique()) - set(dtc_all_fp.keys()):
    if s in all_fp:
        dtc_all_fp[s] = all_fp[s]

evaluate("GLASS2_DTC_novel_30", glass, dtc_all_emb, dtc_all_fp,
         mask_canonical_drugs=True, exclude_prots=excl_prots_30 | dtc_excl_p)

evaluate("GLASS2_DTC_novel_50", glass, dtc_all_emb, dtc_all_fp,
         mask_canonical_drugs=True, exclude_prots=excl_prots_50 | dtc_excl_p)

log.info(f"\n{'='*60}")
log.info("REFERENCE (from paper):")
log.info("  GLASS2: ConciseAnchor CI=0.598  CoNCISE CI=0.547")
log.info("="*60 + "\nDONE")
