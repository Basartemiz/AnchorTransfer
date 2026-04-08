"""BDB identity-partitioned split at 30% + training + BDB & GLASS2 evaluation.

Split BindingDB at 30% k-mer identity, train CoNCISE + ConciseAnchor,
evaluate with Prot-kNN k=1,5 on both BDB test set and GLASS2 (homolog-filtered).
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from itertools import combinations
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src/src")
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

IDENTITY_THRESHOLD = 0.30

# ================================================================
# PHASE 0: Load data
# ================================================================
log.info("Loading BDB data...")
bdb = pd.read_csv(DATA_DIR / "processed/bindingdb_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed/merged_sequences.json"))

raygun_path = Path("data/processed/raygun_bdb_embeddings.pt")
if not raygun_path.exists():
    for p in ["results/raygun_bdb_embeddings.pt", "embeddings_model_files/raygun_bdb_embeddings.pt"]:
        if Path(p).exists(): raygun_path = Path(p); break
raygun_embs = torch.load(raygun_path, map_location="cpu", weights_only=False)
log.info(f"Raygun: {len(raygun_embs)} proteins")

fp_path = Path("data/processed/concise_bdb_morgan_fp.pkl")
if not fp_path.exists():
    fp_path = Path("results/concise_bdb_morgan_fp.pkl")
fp_dict = pickle.load(open(fp_path, "rb"))
log.info(f"Morgan FPs: {len(fp_dict)} drugs")

bdb = bdb[bdb.uniprot_id.isin(raygun_embs) & bdb.ligand_smiles.isin(fp_dict)].copy()
log.info(f"BDB filtered: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot, "
         f"{bdb.ligand_smiles.nunique()} drugs")

# ================================================================
# PHASE 1: Identity-based split at 30%
# ================================================================
SPLIT_CACHE = RESULTS / "bdb_identity_split_30.json"

if SPLIT_CACHE.exists():
    log.info(f"Loading cached split from {SPLIT_CACHE}")
    split_info = json.load(open(SPLIT_CACHE))
    train_prots = set(split_info["train"])
    val_prots = set(split_info["val"])
    test_prots = set(split_info["test"])
else:
    log.info("Computing all-vs-all k-mer identity (vectorized)...")
    prot_list = sorted(set(bdb.uniprot_id) & set(seqs.keys()) & set(raygun_embs.keys()))
    log.info(f"  {len(prot_list)} proteins with sequences + Raygun")

    t0 = time.time()
    kmer_vocab = {}
    for uid in prot_list:
        s = seqs[uid]
        for i in range(len(s) - 2):
            km = s[i:i+3]
            if km not in kmer_vocab:
                kmer_vocab[km] = len(kmer_vocab)
    log.info(f"  3-mer vocabulary: {len(kmer_vocab)} unique k-mers")

    mat = np.zeros((len(prot_list), len(kmer_vocab)), dtype=np.float32)
    for idx, uid in enumerate(prot_list):
        s = seqs[uid]
        for i in range(len(s) - 2):
            mat[idx, kmer_vocab[s[i:i+3]]] = 1

    log.info("  Computing pairwise Jaccard (matrix multiply)...")
    intersection = mat @ mat.T
    sums = mat.sum(axis=1)
    union = sums[:, None] + sums[None, :] - intersection
    jaccard = intersection / np.maximum(union, 1)
    np.fill_diagonal(jaccard, 0)
    log.info(f"  Identity matrix: {jaccard.shape}, {time.time()-t0:.1f}s")

    upper = jaccard[np.triu_indices_from(jaccard, k=1)]
    n_above = (upper >= IDENTITY_THRESHOLD).sum()
    log.info(f"  Pairs >= {IDENTITY_THRESHOLD:.0%} identity: {n_above} / {len(upper)} "
             f"({100*n_above/len(upper):.2f}%)")

    log.info(f"  Finding connected components at {IDENTITY_THRESHOLD:.0%}...")
    adj = jaccard >= IDENTITY_THRESHOLD
    visited = np.zeros(len(prot_list), dtype=bool)
    components = []
    for i in range(len(prot_list)):
        if visited[i]: continue
        comp = []
        queue = [i]
        while queue:
            node = queue.pop()
            if visited[node]: continue
            visited[node] = True
            comp.append(node)
            queue.extend(np.where(adj[node] & ~visited)[0].tolist())
        components.append(comp)

    prot_int_counts = bdb.groupby("uniprot_id").size().to_dict()
    comp_sizes = []
    for comp in components:
        total_int = sum(prot_int_counts.get(prot_list[i], 0) for i in comp)
        comp_sizes.append((total_int, len(comp), comp))
    comp_sizes.sort(key=lambda x: x[0], reverse=True)

    log.info(f"  {len(components)} connected components")
    log.info(f"  Largest 5: {[(s[0], s[1]) for s in comp_sizes[:5]]}")
    log.info(f"  Singletons: {sum(1 for s in comp_sizes if s[1] == 1)}")

    total_interactions = sum(s[0] for s in comp_sizes)
    target_test = 0.10 * total_interactions
    target_val = 0.10 * total_interactions
    random.shuffle(comp_sizes)

    test_indices, val_indices, train_indices = [], [], []
    test_int, val_int = 0, 0
    for int_count, n_prots, comp in comp_sizes:
        if test_int < target_test:
            test_indices.extend(comp); test_int += int_count
        elif val_int < target_val:
            val_indices.extend(comp); val_int += int_count
        else:
            train_indices.extend(comp)

    train_prots = set(prot_list[i] for i in train_indices)
    val_prots = set(prot_list[i] for i in val_indices)
    test_prots = set(prot_list[i] for i in test_indices)

    log.info("  Verifying identity separation...")
    violations = 0
    for ti in test_indices:
        for tri in train_indices:
            if jaccard[ti, tri] >= IDENTITY_THRESHOLD:
                violations += 1; break
    log.info(f"  Violations: {violations}")

    split_info = {"train": sorted(train_prots), "val": sorted(val_prots),
                  "test": sorted(test_prots), "threshold": IDENTITY_THRESHOLD,
                  "n_components": len(components)}
    json.dump(split_info, open(SPLIT_CACHE, "w"), indent=2)
    log.info(f"  Saved split to {SPLIT_CACHE}")

train_df = bdb[bdb.uniprot_id.isin(train_prots)]
val_df = bdb[bdb.uniprot_id.isin(val_prots)]
test_df = bdb[bdb.uniprot_id.isin(test_prots)]
log.info(f"\n{'='*60}")
log.info(f"  SPLIT SUMMARY (30% identity)")
log.info(f"  Train: {len(train_prots)} prot, {len(train_df)} int ({100*len(train_df)/len(bdb):.1f}%)")
log.info(f"  Val:   {len(val_prots)} prot, {len(val_df)} int ({100*len(val_df)/len(bdb):.1f}%)")
log.info(f"  Test:  {len(test_prots)} prot, {len(test_df)} int ({100*len(test_df)/len(bdb):.1f}%)")
log.info(f"{'='*60}\n")

# ================================================================
# PHASE 2: Build anchors + fair filtering
# ================================================================
log.info("Building anchor pool from training proteins...")
train_sub = bdb[bdb.uniprot_id.isin(train_prots)]
drug_to_anchors = {}
for smi, grp in train_sub.groupby("ligand_smiles"):
    best = grp.sort_values("pki", ascending=False)
    candidates = [(u, p) for u, p in zip(best.uniprot_id.values, best.pki.values)
                   if p >= 7.0 and u in raygun_embs]
    if candidates:
        drug_to_anchors[smi] = candidates
anchor_drugs = set(drug_to_anchors.keys())
log.info(f"Anchors: {len(anchor_drugs)} drugs with pKi>=7 binders")

train_df = train_df[train_df.ligand_smiles.isin(anchor_drugs)].copy()
val_df = val_df[val_df.ligand_smiles.isin(anchor_drugs)].copy()
test_df = test_df[test_df.ligand_smiles.isin(anchor_drugs)].copy()
log.info(f"After anchor-drug filter: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# Fair comparison: filter train to same interactions ConciseAnchor can use
log.info("Filtering to interactions with non-self anchors (fair comparison)...")
keep_mask = []
for _, row in train_df.iterrows():
    uid, smi = row.uniprot_id, row.ligand_smiles
    has_nonself = False
    if smi in drug_to_anchors and uid in raygun_embs:
        for au, ap in drug_to_anchors[smi]:
            if au != uid: has_nonself = True; break
    keep_mask.append(has_nonself)
train_df = train_df[keep_mask].copy()
log.info(f"After non-self anchor filter: train={len(train_df)} (same for both models)")

if len(test_df) < 10:
    log.error("Too few test interactions. Aborting."); sys.exit(1)

# ================================================================
# Metrics
# ================================================================
from sklearn.metrics import roc_auc_score, mean_squared_error

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

def bootstrap_mean_ci(values, n_boot=1000, ci=0.95):
    """Bootstrap 95% CI for the mean of per-protein values."""
    if len(values) < 2: return np.mean(values), np.nan, np.nan
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2 * 100)
    hi = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(values), lo, hi

# ================================================================
# PHASE 3: Train CoNCISE
# ================================================================
from concise.model.concise import Concise

class DS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, normalize_pki=False, pki_min=3.0, pki_max=12.0):
        fps, uids, pkis = [], [], []
        for _, r in df.iterrows():
            if r.uniprot_id in raygun_embs and r.ligand_smiles in fp_dict:
                fps.append(np.array(fp_dict[r.ligand_smiles], dtype=np.float32))
                uids.append(r.uniprot_id); pkis.append(r.pki)
        self.fps = torch.tensor(np.array(fps))
        self.uids = uids
        pkis_arr = np.array(pkis, dtype=np.float32)
        if normalize_pki:
            pkis_arr = (pkis_arr - pki_min) / (pki_max - pki_min)  # map to [0, 1]
        self.pkis = torch.tensor(pkis_arr, dtype=torch.float32)
        log.info(f"  DS: {len(self.pkis)} interactions")
    def __len__(self): return len(self.pkis)
    def __getitem__(self, i): return self.fps[i], raygun_embs[self.uids[i]], self.pkis[i]

class ConciseOriginal(nn.Module):
    """Original CoNCISE architecture: drug_dim=256, gelu, cosine_prediction=True."""
    def __init__(self):
        super().__init__()
        drug_layers = [[32], [32], [32]]
        self.backbone = Concise(drug_layers=drug_layers, ligand_dim=2048, residue_dim=1280,
            drug_dim=256, proj_dim=256, nheads=32, activation="gelu", cosine_prediction=True)
    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]

# Compute pKi range for normalization (cosine output is [-1,1])
PKI_MIN = float(train_df.pki.min())
PKI_MAX = float(train_df.pki.max())
log.info(f"pKi range: [{PKI_MIN:.2f}, {PKI_MAX:.2f}]")

CONCISE_DIR = Path("models/concise_bdb_id30"); CONCISE_DIR.mkdir(parents=True, exist_ok=True)
CONCISE_BEST = CONCISE_DIR / "best_model.pt"

if CONCISE_BEST.exists():
    log.info(f"CoNCISE model exists at {CONCISE_BEST}, skipping training")
else:
    log.info("Training CoNCISE (original architecture: cosine_prediction=True)...")
    model_c = ConciseOriginal().to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_c.parameters() if p.requires_grad):,}")
    # Normalize pKi to [0,1] for cosine output training
    train_ds = DS(train_df, fp_dict, raygun_embs, normalize_pki=True, pki_min=PKI_MIN, pki_max=PKI_MAX)
    val_ds = DS(val_df, fp_dict, raygun_embs, normalize_pki=True, pki_min=PKI_MIN, pki_max=PKI_MAX)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)
    optimizer = torch.optim.AdamW(model_c.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_r = -1.0  # select by correlation, not loss (cosine output makes MSE misleading)
    for ep in range(1, 11):
        t0 = time.time(); model_c.train(); total_loss, nb = 0, 0
        for drug, prot, pki in tqdm(train_loader, desc=f"CoNCISE Ep {ep}", leave=False):
            drug, prot, pki = drug.to(DEVICE), prot.to(DEVICE), pki.to(DEVICE)
            pred = model_c(drug, prot); loss = F.mse_loss(pred, pki)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(pki); nb += len(pki)
        scheduler.step()
        model_c.eval(); vp, vt = [], []
        with torch.no_grad():
            for d, p, pk in val_loader:
                vp.extend(model_c(d.to(DEVICE), p.to(DEVICE)).cpu().tolist()); vt.extend(pk.tolist())
        val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0
        val_loss = np.mean((np.array(vt) - np.array(vp))**2)
        tag = "*BEST*" if val_r > best_r else ""
        if val_r > best_r:
            best_r = val_r
            torch.save({"model_state_dict": model_c.state_dict(), "epoch": ep,
                        "pki_min": PKI_MIN, "pki_max": PKI_MAX}, str(CONCISE_BEST))
        log.info(f"  CoNCISE Ep {ep} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {tag}")
    del model_c, train_ds, val_ds, train_loader, val_loader; torch.cuda.empty_cache()

# ================================================================
# PHASE 4: Train ConciseAnchor
# ================================================================
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class AnchorDS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, drug_to_anchors):
        fps, anchor_uids, query_uids, pkis = [], [], [], []
        for _, row in df.iterrows():
            uid, smi, pki = row.uniprot_id, row.ligand_smiles, row.pki
            if smi not in drug_to_anchors or smi not in fp_dict or uid not in raygun_embs: continue
            anchor = None
            for au, ap in drug_to_anchors[smi]:
                if au != uid: anchor = au; break
            if anchor is None: continue
            fps.append(np.array(fp_dict[smi], dtype=np.float32))
            anchor_uids.append(anchor); query_uids.append(uid); pkis.append(pki)
        self.drug_fps = torch.tensor(np.array(fps))
        self.anchor_uids = anchor_uids; self.query_uids = query_uids
        self.pkis = torch.tensor(pkis, dtype=torch.float32)
        log.info(f"  AnchorDS: {len(self.pkis)} interactions")
    def __len__(self): return len(self.pkis)
    def __getitem__(self, i):
        return (self.drug_fps[i], raygun_embs[self.anchor_uids[i]],
                raygun_embs[self.query_uids[i]], self.pkis[i])

ANCHOR_DIR = Path("models/concise_anchor_bdb_id30"); ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_BEST = ANCHOR_DIR / "best_model.pt"

if ANCHOR_BEST.exists():
    log.info(f"ConciseAnchor model exists at {ANCHOR_BEST}, skipping training")
else:
    log.info("Training ConciseAnchor...")
    model_a = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2).to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad):,}")
    train_ds = AnchorDS(train_df, fp_dict, raygun_embs, drug_to_anchors)
    val_ds = AnchorDS(val_df, fp_dict, raygun_embs, drug_to_anchors)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=4e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val = float("inf")
    for ep in range(1, 11):
        t0 = time.time(); model_a.train(); total_loss, nb = 0, 0
        for drug, anchor, query, pki in tqdm(train_loader, desc=f"Anchor Ep {ep}", leave=False):
            drug, anchor, query, pki = drug.to(DEVICE), anchor.to(DEVICE), query.to(DEVICE), pki.to(DEVICE)
            pred = model_a(drug, anchor, query); loss = F.mse_loss(pred, pki)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(pki); nb += len(pki)
        scheduler.step()
        model_a.eval(); vp, vt = [], []
        with torch.no_grad():
            for d, a, q, pk in val_loader:
                vp.extend(model_a(d.to(DEVICE), a.to(DEVICE), q.to(DEVICE)).cpu().tolist()); vt.extend(pk.tolist())
        val_loss = np.mean((np.array(vt) - np.array(vp))**2)
        val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0
        tag = "*BEST*" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model_a.state_dict(), "epoch": ep}, str(ANCHOR_BEST))
        log.info(f"  Anchor Ep {ep} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {tag}")
    del model_a, train_ds, val_ds, train_loader, val_loader; torch.cuda.empty_cache()


# ================================================================
# Helper: predict + kNN for any test set
# ================================================================
from rdkit import Chem
from rdkit.Chem import AllChem

def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except: return s

# Build training retrieval pool (shared across BDB test + GLASS2 eval)
log.info("Building training retrieval pool...")
train_drugs = sorted(train_df.ligand_smiles.unique())
train_prots_list = sorted(train_prots & set(raygun_embs.keys()))
train_drug_idx = {d: i for i, d in enumerate(train_drugs)}
train_prot_idx = {p: i for i, p in enumerate(train_prots_list)}

pair_max = {}
for _, r in train_df.iterrows():
    di, pi = train_drug_idx.get(r.ligand_smiles, -1), train_prot_idx.get(r.uniprot_id, -1)
    if di < 0 or pi < 0: continue
    key = (di, pi)
    if key not in pair_max or r.pki > pair_max[key]: pair_max[key] = r.pki

INT_DENSE = np.zeros((len(train_drugs), len(train_prots_list)), dtype=np.float32)
for (di, pi), val in pair_max.items(): INT_DENSE[di, pi] = val
log.info(f"  Interaction matrix: {INT_DENSE.shape}, nnz={np.count_nonzero(INT_DENSE)}")

train_emb_mat = np.array([raygun_embs[p].mean(dim=0).numpy() for p in train_prots_list])
train_emb_norms = np.linalg.norm(train_emb_mat, axis=1, keepdims=True) + 1e-10
train_emb_normed = train_emb_mat / train_emb_norms
train_fp_mat = np.array([np.array(fp_dict[d], dtype=np.float32) for d in train_drugs])

log.info("  Canonicalizing training drugs...")
train_canon = {d: canon(d) for d in train_drugs}
train_canonical_set = set(train_canon.values())


def evaluate_dataset(name, eval_df, eval_raygun, eval_fp_dict):
    """Full evaluation: CoNCISE + ConciseAnchor + Prot-kNN k=1,5."""
    log.info(f"\n{'='*60}")
    log.info(f"  {name}: {len(eval_df)} int, {eval_df.uniprot_id.nunique()} proteins")
    log.info(f"{'='*60}")

    # ── Model predictions ──
    log.info("  CoNCISE predictions...")
    model_c = ConciseOriginal().to(DEVICE)
    ckpt = torch.load(str(CONCISE_BEST), map_location=DEVICE, weights_only=False)
    model_c.load_state_dict(ckpt["model_state_dict"]); model_c.eval()
    concise_preds = {}
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            uid, smi = row.uniprot_id, row.ligand_smiles
            if uid not in eval_raygun or smi not in eval_fp_dict: continue
            fp = torch.tensor(np.array(eval_fp_dict[smi], dtype=np.float32)).unsqueeze(0).to(DEVICE)
            emb = eval_raygun[uid].unsqueeze(0).to(DEVICE)
            concise_preds[(uid, smi)] = model_c(fp, emb).item()
    eval_df = eval_df.copy()
    eval_df["concise_pred"] = [concise_preds.get((r.uniprot_id, r.ligand_smiles), np.nan) for _, r in eval_df.iterrows()]
    log.info(f"    {sum(~np.isnan(eval_df.concise_pred))}/{len(eval_df)} predictions")
    del model_c; torch.cuda.empty_cache()

    log.info("  ConciseAnchor predictions...")
    model_a = ConciseAnchorBilinear(ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2).to(DEVICE)
    ckpt = torch.load(str(ANCHOR_BEST), map_location=DEVICE, weights_only=False)
    model_a.load_state_dict(ckpt["model_state_dict"]); model_a.eval()
    anchor_preds, anchor_pkis_map = {}, {}
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            uid, smi = row.uniprot_id, row.ligand_smiles
            if smi not in drug_to_anchors or smi not in eval_fp_dict or uid not in eval_raygun: continue
            anchor_uid, anchor_pki_val = None, None
            for au, ap in drug_to_anchors[smi]:
                if au != uid: anchor_uid, anchor_pki_val = au, ap; break
            if anchor_uid is None: continue
            fp = torch.tensor(np.array(eval_fp_dict[smi], dtype=np.float32)).unsqueeze(0).to(DEVICE)
            anc_emb = raygun_embs[anchor_uid].unsqueeze(0).to(DEVICE)
            qry_emb = eval_raygun[uid].unsqueeze(0).to(DEVICE)
            anchor_preds[(uid, smi)] = model_a(fp, anc_emb, qry_emb).item()
            anchor_pkis_map[(uid, smi)] = anchor_pki_val
    eval_df["anchor_pred"] = [anchor_preds.get((r.uniprot_id, r.ligand_smiles), np.nan) for _, r in eval_df.iterrows()]
    eval_df["anchor_pki"] = [anchor_pkis_map.get((r.uniprot_id, r.ligand_smiles), np.nan) for _, r in eval_df.iterrows()]
    log.info(f"    {sum(~np.isnan(eval_df.anchor_pred))}/{len(eval_df)} predictions")
    del model_a; torch.cuda.empty_cache()

    # ── Prot-kNN ──
    log.info("  Prot-kNN baselines...")
    eval_drugs_uniq = sorted(eval_df.ligand_smiles.unique())
    eval_prots_uniq = sorted(eval_df.uniprot_id.unique())
    edm = {d: i for i, d in enumerate(eval_drugs_uniq)}
    epm = {p: i for i, p in enumerate(eval_prots_uniq)}

    # Drug sims with masking
    eval_fps = np.array([np.array(eval_fp_dict[d], dtype=np.float32) for d in eval_drugs_uniq])
    inter = eval_fps @ train_fp_mat.T
    q_bits = eval_fps.sum(1, keepdims=True)
    db_bits = train_fp_mat.sum(1, keepdims=True).T
    ds = inter / np.maximum(q_bits + db_bits - inter, 1)
    q_canon = [canon(s) for s in eval_drugs_uniq]
    for qi, qc in enumerate(q_canon):
        for di, d in enumerate(train_drugs):
            if train_canon.get(d) == qc: ds[qi, di] = -1

    # Prot sims
    eval_embs = np.array([eval_raygun[p].mean(dim=0).numpy() if isinstance(eval_raygun[p], torch.Tensor)
                           else np.mean(eval_raygun[p], axis=0) for p in eval_prots_uniq])
    qn = eval_embs / (np.linalg.norm(eval_embs, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_emb_normed.T
    log.info(f"    Drug sims: {ds.shape}, Prot sims: {ps.shape}")

    di_arr = np.array([edm.get(s, -1) for s in eval_df.ligand_smiles.values])
    pi_arr = np.array([epm.get(p, -1) for p in eval_df.uniprot_id.values])

    def run_prot_knn(k):
        n = len(di_arr); preds = np.full(n, np.nan)
        for qpi_idx in np.unique(pi_arr[pi_arr >= 0]):
            mask_rows = pi_arr == qpi_idx
            top_k = np.argsort(ps[qpi_idx])[-k:][::-1]
            top_sims = ps[qpi_idx, top_k]; valid = top_sims > 0
            top_k, top_sims = top_k[valid], top_sims[valid]
            if len(top_k) == 0: continue
            for ri in np.where(mask_rows)[0]:
                qdi = di_arr[ri]
                if qdi < 0: continue
                drug_s = ds[qdi]; best_pkis, best_wts = [], []
                for ki, bpi in enumerate(top_k):
                    int_col = INT_DENSE[:, bpi]; has_int = int_col > 0
                    if not has_int.any(): continue
                    masked = np.where(has_int, drug_s, -2)
                    best_di = np.argmax(masked)
                    if masked[best_di] <= 0: continue
                    best_pkis.append(int_col[best_di]); best_wts.append(top_sims[ki])
                if best_pkis: preds[ri] = np.average(best_pkis, weights=best_wts)
        return preds

    for k in [1, 5]:
        t0 = time.time()
        eval_df[f"prot_knn_k{k}"] = run_prot_knn(k)
        log.info(f"    Prot-kNN k={k}: {sum(~np.isnan(eval_df[f'prot_knn_k{k}']))}/{len(eval_df)} preds, {time.time()-t0:.0f}s")

    # ── Report with bootstrap 95% CI ──
    methods = ["concise_pred", "anchor_pred", "prot_knn_k1", "prot_knn_k5"]
    method_names = ["CoNCISE", "ConciseAnchor", "Prot-kNN k=1", "Prot-kNN k=5"]
    log.info(f"\n  {'Method':<18s} {'CI (95% boot)':>22s} {'AUROC':>7s} {'RMSE':>7s}")
    log.info(f"  {'-'*60}")
    for col, mname in zip(methods, method_names):
        ci_vals = pp_ci(eval_df, col)
        ci_mean, ci_lo, ci_hi = bootstrap_mean_ci(ci_vals) if len(ci_vals) > 1 else (ci_vals.mean(), 0, 0)
        auc = pp_auroc(eval_df, col); rmse = pp_rmse(eval_df, col)
        log.info(f"  {mname:<18s} {ci_mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] "
                 f"{auc.mean() if len(auc) else 0:7.3f} "
                 f"{rmse.mean() if len(rmse) else 0:7.3f}")
    log.info(f"  {'-'*60}")

    # Quartile breakdown
    valid_anchor = eval_df.dropna(subset=["anchor_pki"])
    if len(valid_anchor) > 20:
        q_edges = np.quantile(valid_anchor.anchor_pki.values, [0, 0.25, 0.5, 0.75, 1.0])
        q_labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
        valid_anchor = valid_anchor.copy()
        valid_anchor["anchor_q"] = pd.cut(valid_anchor.anchor_pki, bins=q_edges, labels=q_labels, include_lowest=True)
        log.info(f"\n  ── Anchor pKi Quartile Breakdown ──")
        for col, mname in zip(methods, method_names):
            log.info(f"\n  {mname}:")
            for ql in q_labels:
                qdf = valid_anchor[valid_anchor.anchor_q == ql]
                if len(qdf) < 5: continue
                ci = pp_ci(qdf, col); rmse = pp_rmse(qdf, col); auc = pp_auroc(qdf, col)
                log.info(f"    {ql:<16s}  CI={ci.mean():.3f}  RMSE={rmse.mean() if len(rmse) else 0:.3f}  "
                         f"AUROC={auc.mean() if len(auc) else 0:.3f}  (n={len(qdf)}, {qdf.uniprot_id.nunique()} prot)")

    out_csv = RESULTS / f"{name.lower().replace(' ','_')}_results.csv"
    eval_df.to_csv(out_csv, index=False)
    log.info(f"\n  Saved {out_csv}")
    return eval_df


# ================================================================
# PHASE 5: Evaluate on BDB test set
# ================================================================
evaluate_dataset("bdb_id30_test", test_df, raygun_embs, fp_dict)


# ================================================================
# PHASE 6: Compute Raygun embeddings for GLASS2 proteins
# ================================================================
log.info(f"\n{'='*60}")
log.info("GLASS2 CROSS-DATASET EVALUATION")
log.info(f"{'='*60}")

glass = pd.read_csv(DATA_DIR / "raw/glass/glass2_ki_interactions.csv")
glass_seqs = json.load(open(DATA_DIR / "raw/glass/glass2_sequences.json"))
log.info(f"GLASS2: {len(glass)} int, {glass.uniprot_id.nunique()} prot, "
         f"{glass.ligand_smiles.nunique()} drugs")

# Compute Raygun embeddings for GLASS2 proteins
GLASS_RAYGUN_CACHE = RESULTS / "raygun_glass_embeddings.pt"

if GLASS_RAYGUN_CACHE.exists():
    glass_raygun = torch.load(GLASS_RAYGUN_CACHE, map_location="cpu", weights_only=False)
    log.info(f"Loaded GLASS2 Raygun: {len(glass_raygun)} proteins")
else:
    log.info("Computing Raygun embeddings for GLASS2 proteins...")
    import esm

    # Step 1: ESM2 embeddings
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(DEVICE)

    glass_prots = sorted(set(glass.uniprot_id) & set(glass_seqs.keys()))
    log.info(f"  {len(glass_prots)} GLASS2 proteins with sequences")
    esm_embs = {}
    items = [(u, glass_seqs[u][:1022]) for u in glass_prots if len(glass_seqs[u]) >= 25]
    log.info(f"  Computing ESM2 for {len(items)} proteins...")
    for i in range(0, len(items), 8):
        batch = items[i:i+8]
        _, _, toks = bc(batch)
        with torch.no_grad():
            out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
        for j, (u, s) in enumerate(batch):
            esm_embs[u] = out["representations"][33][j, 1:len(s)+1, :].cpu()
        if (i+8) % 100 < 8:
            log.info(f"    ESM2: {min(i+8, len(items))}/{len(items)}")
    del esm_model; torch.cuda.empty_cache()

    # Step 2: Raygun encoder (full 50x1280 embeddings)
    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)
    glass_raygun = {}
    for idx, (uid, emb) in enumerate(esm_embs.items()):
        try:
            with torch.no_grad():
                out = raygun_model(emb.unsqueeze(0).to(DEVICE))
                proj = out[0] if isinstance(out, tuple) else out
                glass_raygun[uid] = proj.squeeze(0).cpu()  # (50, 1280)
        except Exception as e:
            log.info(f"  Error on {uid}: {e}")
        if (idx+1) % 100 == 0:
            log.info(f"    Raygun: {idx+1}/{len(esm_embs)}")

    del raygun_model; torch.cuda.empty_cache()
    torch.save(glass_raygun, str(GLASS_RAYGUN_CACHE))
    log.info(f"  Saved {len(glass_raygun)} GLASS2 Raygun embeddings to {GLASS_RAYGUN_CACHE}")

# Compute Morgan FPs for GLASS2 drugs
log.info("Computing GLASS2 drug fingerprints...")
glass_fp = {}
glass_drug_list = glass.ligand_smiles.unique()
for smi in glass_drug_list:
    if smi in fp_dict:
        glass_fp[smi] = fp_dict[smi]
    else:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                glass_fp[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
log.info(f"  GLASS2 FPs: {len(glass_fp)}/{len(glass_drug_list)} drugs")

# Filter GLASS2 to proteins with Raygun embeddings + drugs with FPs
glass = glass[glass.uniprot_id.isin(glass_raygun) & glass.ligand_smiles.isin(glass_fp)].copy()
log.info(f"GLASS2 filtered: {len(glass)} int, {glass.uniprot_id.nunique()} prot")

# Homolog filtering: exclude GLASS2 proteins with >=30% identity to BDB training
log.info(f"Computing 30% identity homologs (GLASS2 vs BDB training)...")
bdb_train_seqs = {p: seqs[p] for p in train_prots if p in seqs}

# Build k-mer sets
glass_kmer_sets = {}
for uid in glass.uniprot_id.unique():
    if uid in glass_seqs:
        s = glass_seqs[uid]
        glass_kmer_sets[uid] = set(s[i:i+3] for i in range(len(s)-2))

bdb_kmer_sets = {}
for uid, s in bdb_train_seqs.items():
    bdb_kmer_sets[uid] = set(s[i:i+3] for i in range(len(s)-2))

homologs = set()
exact_overlap = set(glass.uniprot_id) & train_prots
log.info(f"  Exact protein overlap: {len(exact_overlap)}")

from multiprocessing import Pool, cpu_count
def _check_glass_homolog(g_uid):
    if g_uid not in glass_kmer_sets: return None
    gk = glass_kmer_sets[g_uid]
    if not gk: return None
    for bk in bdb_kmer_sets.values():
        if not bk: continue
        jacc = len(gk & bk) / len(gk | bk)
        if jacc >= 0.30: return g_uid
    return None

glass_prots_to_check = sorted(set(glass.uniprot_id) - exact_overlap)
log.info(f"  Checking {len(glass_prots_to_check)} GLASS2 proteins for 30% homologs...")
with Pool(min(cpu_count(), 32)) as pool:
    results = pool.map(_check_glass_homolog, glass_prots_to_check)
homologs = {r for r in results if r is not None}
log.info(f"  Found {len(homologs)} homologs (>= 30% identity)")

all_excl = exact_overlap | homologs
glass_novel = glass[~glass.uniprot_id.isin(all_excl)].copy()
# Also filter to anchor-eligible drugs
glass_novel = glass_novel[glass_novel.ligand_smiles.isin(anchor_drugs)].copy()
log.info(f"GLASS2 novel (<30% identity, anchor-eligible): {len(glass_novel)} int, "
         f"{glass_novel.uniprot_id.nunique()} proteins")

if len(glass_novel) >= 10:
    evaluate_dataset("glass2_novel_30pct", glass_novel, glass_raygun, glass_fp)
else:
    log.info("Too few GLASS2 novel interactions for evaluation.")

log.info("\nAll done!")
