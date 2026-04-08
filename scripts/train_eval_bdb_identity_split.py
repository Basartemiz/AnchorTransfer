"""BDB identity-partitioned train/val/test + training + kNN evaluation.

Split BindingDB proteins at 50% k-mer identity:
  - Connected components at 50% → clusters
  - Assign clusters to train/val/test targeting 80/10/10 by interactions
  - Guarantees: every test protein < 50% identity to every train protein

Then trains CoNCISE and ConciseAnchor, and evaluates Prot-kNN k=1,5.
Reports CI, RMSE, AUROC, and per-anchor-quartile breakdown.
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
from collections import defaultdict
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IDENTITY_THRESHOLD = 0.50  # 50% k-mer identity

# ================================================================
# PHASE 0: Load data
# ================================================================
log.info("Loading BDB data...")
bdb = pd.read_csv(DATA_DIR / "processed/bindingdb_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed/merged_sequences.json"))

# Raygun embeddings
raygun_path = Path("data/processed/raygun_bdb_embeddings.pt")
if not raygun_path.exists():
    raygun_path = Path("results/raygun_bdb_embeddings.pt")
if not raygun_path.exists():
    raygun_path = Path("embeddings_model_files/raygun_bdb_embeddings.pt")
raygun_embs = torch.load(raygun_path, map_location="cpu", weights_only=False)
log.info(f"Raygun: {len(raygun_embs)} proteins")

# Morgan fingerprints
fp_path = Path("data/processed/concise_bdb_morgan_fp.pkl")
if not fp_path.exists():
    fp_path = Path("results/concise_bdb_morgan_fp.pkl")
fp_dict = pickle.load(open(fp_path, "rb"))
log.info(f"Morgan FPs: {len(fp_dict)} drugs")

# Filter BDB to proteins/drugs with embeddings
bdb = bdb[bdb.uniprot_id.isin(raygun_embs) & bdb.ligand_smiles.isin(fp_dict)].copy()
log.info(f"BDB filtered: {len(bdb)} int, {bdb.uniprot_id.nunique()} prot, "
         f"{bdb.ligand_smiles.nunique()} drugs")

# ================================================================
# PHASE 1: Identity-based split
# ================================================================
SPLIT_CACHE = RESULTS / "bdb_identity_split_50.json"

if SPLIT_CACHE.exists():
    log.info(f"Loading cached split from {SPLIT_CACHE}")
    split_info = json.load(open(SPLIT_CACHE))
    train_prots = set(split_info["train"])
    val_prots = set(split_info["val"])
    test_prots = set(split_info["test"])
else:
    log.info("Computing all-vs-all k-mer identity (vectorized)...")
    # Get proteins with both sequence and Raygun embeddings
    prot_list = sorted(set(bdb.uniprot_id) & set(seqs.keys()) & set(raygun_embs.keys()))
    log.info(f"  {len(prot_list)} proteins with sequences + Raygun")

    # Build 3-mer binary matrix
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
            km = s[i:i+3]
            mat[idx, kmer_vocab[km]] = 1

    # Pairwise Jaccard via matrix multiplication
    log.info("  Computing pairwise Jaccard (matrix multiply)...")
    intersection = mat @ mat.T
    sums = mat.sum(axis=1)
    union = sums[:, None] + sums[None, :] - intersection
    jaccard = intersection / np.maximum(union, 1)
    np.fill_diagonal(jaccard, 0)  # no self-edges
    log.info(f"  Identity matrix: {jaccard.shape}, {time.time()-t0:.1f}s")

    # Stats
    upper = jaccard[np.triu_indices_from(jaccard, k=1)]
    n_above = (upper >= IDENTITY_THRESHOLD).sum()
    log.info(f"  Pairs ≥ {IDENTITY_THRESHOLD:.0%} identity: {n_above} / {len(upper)} "
             f"({100*n_above/len(upper):.2f}%)")

    # Connected components at threshold
    log.info(f"  Finding connected components at {IDENTITY_THRESHOLD:.0%}...")
    adj = jaccard >= IDENTITY_THRESHOLD
    visited = np.zeros(len(prot_list), dtype=bool)
    components = []
    for i in range(len(prot_list)):
        if visited[i]:
            continue
        # BFS
        comp = []
        queue = [i]
        while queue:
            node = queue.pop()
            if visited[node]:
                continue
            visited[node] = True
            comp.append(node)
            neighbors = np.where(adj[node] & ~visited)[0]
            queue.extend(neighbors.tolist())
        components.append(comp)

    # Sort components by total interactions (largest first)
    prot_int_counts = bdb.groupby("uniprot_id").size().to_dict()
    comp_sizes = []
    for comp in components:
        total_int = sum(prot_int_counts.get(prot_list[i], 0) for i in comp)
        comp_sizes.append((total_int, len(comp), comp))
    comp_sizes.sort(key=lambda x: x[0], reverse=True)

    log.info(f"  {len(components)} connected components")
    log.info(f"  Largest 5: {[(s[0], s[1]) for s in comp_sizes[:5]]} (interactions, proteins)")
    log.info(f"  Singletons: {sum(1 for s in comp_sizes if s[1] == 1)}")

    # Assign clusters to train/val/test targeting 80/10/10 by interactions
    total_interactions = sum(s[0] for s in comp_sizes)
    target_test = 0.10 * total_interactions
    target_val = 0.10 * total_interactions

    # Shuffle for randomness (keep seed)
    random.shuffle(comp_sizes)

    test_indices, val_indices, train_indices = [], [], []
    test_int, val_int = 0, 0

    for int_count, n_prots, comp in comp_sizes:
        if test_int < target_test:
            test_indices.extend(comp)
            test_int += int_count
        elif val_int < target_val:
            val_indices.extend(comp)
            val_int += int_count
        else:
            train_indices.extend(comp)

    train_prots = set(prot_list[i] for i in train_indices)
    val_prots = set(prot_list[i] for i in val_indices)
    test_prots = set(prot_list[i] for i in test_indices)

    # Verify: no test protein ≥50% identity to any train protein
    log.info("  Verifying identity separation...")
    violations = 0
    for ti in test_indices:
        for tri in train_indices:
            if jaccard[ti, tri] >= IDENTITY_THRESHOLD:
                violations += 1
                break
    log.info(f"  Violations: {violations} test proteins with ≥{IDENTITY_THRESHOLD:.0%} "
             f"identity to train (should be 0)")

    # Save split
    split_info = {
        "train": sorted(train_prots),
        "val": sorted(val_prots),
        "test": sorted(test_prots),
        "threshold": IDENTITY_THRESHOLD,
        "n_components": len(components),
    }
    json.dump(split_info, open(SPLIT_CACHE, "w"), indent=2)
    log.info(f"  Saved split to {SPLIT_CACHE}")

# Report split stats
train_df = bdb[bdb.uniprot_id.isin(train_prots)]
val_df = bdb[bdb.uniprot_id.isin(val_prots)]
test_df = bdb[bdb.uniprot_id.isin(test_prots)]
log.info(f"\n{'='*60}")
log.info(f"  SPLIT SUMMARY (50% identity)")
log.info(f"  Train: {len(train_prots)} prot, {len(train_df)} int "
         f"({100*len(train_df)/len(bdb):.1f}%)")
log.info(f"  Val:   {len(val_prots)} prot, {len(val_df)} int "
         f"({100*len(val_df)/len(bdb):.1f}%)")
log.info(f"  Test:  {len(test_prots)} prot, {len(test_df)} int "
         f"({100*len(test_df)/len(bdb):.1f}%)")
log.info(f"{'='*60}\n")


# ================================================================
# PHASE 2: Build anchors from training set
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
log.info(f"Anchors: {len(anchor_drugs)} drugs with pKi≥7 binders in training set")

# Filter all splits to anchor-eligible drugs (apple-to-apple comparison)
train_df = train_df[train_df.ligand_smiles.isin(anchor_drugs)].copy()
val_df = val_df[val_df.ligand_smiles.isin(anchor_drugs)].copy()
test_df = test_df[test_df.ligand_smiles.isin(anchor_drugs)].copy()
log.info(f"After anchor-drug filter: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

# CRITICAL: Filter train to SAME interactions ConciseAnchor can use
# ConciseAnchor requires a non-self anchor: for (protein p, drug d), there must be
# a training protein != p that binds drug d with pKi >= 7.
# Both models must train on identical interactions for fair comparison.
log.info("Filtering to interactions with non-self anchors (fair comparison)...")
keep_mask = []
for _, row in train_df.iterrows():
    uid, smi = row.uniprot_id, row.ligand_smiles
    has_nonself = False
    if smi in drug_to_anchors and uid in raygun_embs:
        for au, ap in drug_to_anchors[smi]:
            if au != uid:
                has_nonself = True
                break
    keep_mask.append(has_nonself)
train_df = train_df[keep_mask].copy()
log.info(f"After non-self anchor filter: train={len(train_df)} (same for both models)")
log.info(f"Final: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

if len(test_df) < 10:
    log.error("Too few test interactions after anchor filtering. Aborting.")
    sys.exit(1)


# ================================================================
# PHASE 3: Train CoNCISE
# ================================================================
CONCISE_DIR = Path("models/concise_bdb_id50")
CONCISE_DIR.mkdir(parents=True, exist_ok=True)
CONCISE_BEST = CONCISE_DIR / "best_model.pt"

from concise.model.concise import Concise


class DS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs):
        fps, uids, pkis = [], [], []
        for _, r in df.iterrows():
            if r.uniprot_id in raygun_embs and r.ligand_smiles in fp_dict:
                fps.append(np.array(fp_dict[r.ligand_smiles], dtype=np.float32))
                uids.append(r.uniprot_id)
                pkis.append(r.pki)
        self.fps = torch.tensor(np.array(fps))
        self.uids = uids
        self.pkis = torch.tensor(pkis, dtype=torch.float32)
        log.info(f"  DS: {len(self.pkis)} interactions")

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, i):
        return self.fps[i], raygun_embs[self.uids[i]], self.pkis[i]


class ConciseFixed(nn.Module):
    def __init__(self):
        super().__init__()
        drug_layers = [[32], [32], [32]]
        self.backbone = Concise(
            drug_layers=drug_layers, ligand_dim=2048, residue_dim=1280,
            drug_dim=128, proj_dim=256, nheads=32, activation="tanh",
            cosine_prediction=False,
        )
        fused_dim = len(drug_layers) * 256 + 256
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 1),
        )
        nn.init.constant_(self.backbone.final[-1].bias, 6.5)

    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]


if CONCISE_BEST.exists():
    log.info(f"CoNCISE model already exists at {CONCISE_BEST}, skipping training")
else:
    log.info("Training CoNCISE...")
    model_c = ConciseFixed().to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_c.parameters() if p.requires_grad):,}")

    train_ds = DS(train_df, fp_dict, raygun_embs)
    val_ds = DS(val_df, fp_dict, raygun_embs)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model_c.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val = float("inf")

    for ep in range(1, 11):
        t0 = time.time()
        model_c.train()
        total_loss, nb = 0, 0
        for drug, prot, pki in tqdm(train_loader, desc=f"CoNCISE Ep {ep}", leave=False):
            drug, prot, pki = drug.to(DEVICE), prot.to(DEVICE), pki.to(DEVICE)
            pred = model_c(drug, prot)
            loss = F.mse_loss(pred, pki)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(pki); nb += len(pki)
        scheduler.step()

        model_c.eval()
        vp, vt = [], []
        with torch.no_grad():
            for d, p, pk in val_loader:
                vp.extend(model_c(d.to(DEVICE), p.to(DEVICE)).cpu().tolist())
                vt.extend(pk.tolist())
        val_loss = np.mean((np.array(vt) - np.array(vp))**2)
        val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0
        tag = "*BEST*" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model_c.state_dict(), "epoch": ep}, str(CONCISE_BEST))
        log.info(f"  CoNCISE Ep {ep} [{time.time()-t0:.0f}s] "
                 f"Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {tag}")

    del model_c, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()


# ================================================================
# PHASE 4: Train ConciseAnchor
# ================================================================
ANCHOR_DIR = Path("models/concise_anchor_bdb_id50")
ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_BEST = ANCHOR_DIR / "best_model.pt"

from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear


class AnchorDS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, drug_to_anchors):
        fps, anchor_uids, query_uids, pkis = [], [], [], []
        for _, row in df.iterrows():
            uid, smi, pki = row.uniprot_id, row.ligand_smiles, row.pki
            if smi not in drug_to_anchors or smi not in fp_dict or uid not in raygun_embs:
                continue
            anchor = None
            for au, ap in drug_to_anchors[smi]:
                if au != uid:
                    anchor = au; break
            if anchor is None:
                continue
            fps.append(np.array(fp_dict[smi], dtype=np.float32))
            anchor_uids.append(anchor)
            query_uids.append(uid)
            pkis.append(pki)
        self.drug_fps = torch.tensor(np.array(fps))
        self.anchor_uids = anchor_uids
        self.query_uids = query_uids
        self.pkis = torch.tensor(pkis, dtype=torch.float32)
        log.info(f"  AnchorDS: {len(self.pkis)} interactions")

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, i):
        return (self.drug_fps[i], raygun_embs[self.anchor_uids[i]],
                raygun_embs[self.query_uids[i]], self.pkis[i])


if ANCHOR_BEST.exists():
    log.info(f"ConciseAnchor model exists at {ANCHOR_BEST}, skipping training")
else:
    log.info("Training ConciseAnchor...")
    model_a = ConciseAnchorBilinear(
        ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2,
    ).to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad):,}")

    train_ds = AnchorDS(train_df, fp_dict, raygun_embs, drug_to_anchors)
    val_ds = AnchorDS(val_df, fp_dict, raygun_embs, drug_to_anchors)
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4)

    optimizer = torch.optim.AdamW(model_a.parameters(), lr=4e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    best_val = float("inf")

    for ep in range(1, 11):
        t0 = time.time()
        model_a.train()
        total_loss, nb = 0, 0
        for drug, anchor, query, pki in tqdm(train_loader, desc=f"Anchor Ep {ep}", leave=False):
            drug, anchor, query, pki = (drug.to(DEVICE), anchor.to(DEVICE),
                                        query.to(DEVICE), pki.to(DEVICE))
            pred = model_a(drug, anchor, query)
            loss = F.mse_loss(pred, pki)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(pki); nb += len(pki)
        scheduler.step()

        model_a.eval()
        vp, vt = [], []
        with torch.no_grad():
            for d, a, q, pk in val_loader:
                vp.extend(model_a(d.to(DEVICE), a.to(DEVICE), q.to(DEVICE)).cpu().tolist())
                vt.extend(pk.tolist())
        val_loss = np.mean((np.array(vt) - np.array(vp))**2)
        val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0
        tag = "*BEST*" if val_loss < best_val else ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model_state_dict": model_a.state_dict(), "epoch": ep}, str(ANCHOR_BEST))
        log.info(f"  Anchor Ep {ep} [{time.time()-t0:.0f}s] "
                 f"Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {tag}")

    del model_a, train_ds, val_ds, train_loader, val_loader
    torch.cuda.empty_cache()


# ================================================================
# PHASE 5: Evaluate all methods on test set
# ================================================================
log.info(f"\n{'='*60}")
log.info("EVALUATION ON TEST SET")
log.info(f"{'='*60}")

# ── Metrics ────────────────────────────────────────────────────
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


# ── 5a: CoNCISE predictions on test ────────────────────────────
log.info("\nCoNCISE predictions on test set...")
model_c = ConciseFixed().to(DEVICE)
ckpt = torch.load(str(CONCISE_BEST), map_location=DEVICE, weights_only=False)
model_c.load_state_dict(ckpt["model_state_dict"])
model_c.eval()

concise_preds = {}
test_uids = sorted(test_df.uniprot_id.unique())
test_drugs_list = sorted(test_df.ligand_smiles.unique())

with torch.no_grad():
    for _, row in test_df.iterrows():
        uid, smi = row.uniprot_id, row.ligand_smiles
        if uid not in raygun_embs or smi not in fp_dict:
            continue
        fp = torch.tensor(np.array(fp_dict[smi], dtype=np.float32)).unsqueeze(0).to(DEVICE)
        emb = raygun_embs[uid].unsqueeze(0).to(DEVICE)
        pred = model_c(fp, emb).item()
        concise_preds[(uid, smi)] = pred

test_df["concise_pred"] = [concise_preds.get((r.uniprot_id, r.ligand_smiles), np.nan)
                            for _, r in test_df.iterrows()]
log.info(f"  CoNCISE: {sum(~np.isnan(test_df.concise_pred))}/{len(test_df)} predictions")
del model_c; torch.cuda.empty_cache()


# ── 5b: ConciseAnchor predictions on test ──────────────────────
log.info("ConciseAnchor predictions on test set...")
model_a = ConciseAnchorBilinear(
    ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2,
).to(DEVICE)
ckpt = torch.load(str(ANCHOR_BEST), map_location=DEVICE, weights_only=False)
model_a.load_state_dict(ckpt["model_state_dict"])
model_a.eval()

anchor_preds = {}
anchor_pkis = {}
with torch.no_grad():
    for _, row in test_df.iterrows():
        uid, smi = row.uniprot_id, row.ligand_smiles
        if smi not in drug_to_anchors or smi not in fp_dict or uid not in raygun_embs:
            continue
        # Find anchor (from training set, exclude self)
        anchor_uid = None
        anchor_pki_val = None
        for au, ap in drug_to_anchors[smi]:
            if au != uid:
                anchor_uid = au
                anchor_pki_val = ap
                break
        if anchor_uid is None:
            continue
        fp = torch.tensor(np.array(fp_dict[smi], dtype=np.float32)).unsqueeze(0).to(DEVICE)
        anc_emb = raygun_embs[anchor_uid].unsqueeze(0).to(DEVICE)
        qry_emb = raygun_embs[uid].unsqueeze(0).to(DEVICE)
        pred = model_a(fp, anc_emb, qry_emb).item()
        anchor_preds[(uid, smi)] = pred
        anchor_pkis[(uid, smi)] = anchor_pki_val

test_df["anchor_pred"] = [anchor_preds.get((r.uniprot_id, r.ligand_smiles), np.nan)
                           for _, r in test_df.iterrows()]
test_df["anchor_pki"] = [anchor_pkis.get((r.uniprot_id, r.ligand_smiles), np.nan)
                          for _, r in test_df.iterrows()]
log.info(f"  ConciseAnchor: {sum(~np.isnan(test_df.anchor_pred))}/{len(test_df)} predictions")
del model_a; torch.cuda.empty_cache()


# ── 5c: Prot-kNN baselines ────────────────────────────────────
log.info("Prot-kNN baselines (retrieval pool = training set)...")

# Build training interaction matrix (drugs × train_proteins)
train_drugs = sorted(train_df.ligand_smiles.unique())
train_prots_list = sorted(train_prots & set(raygun_embs.keys()))
train_drug_idx = {d: i for i, d in enumerate(train_drugs)}
train_prot_idx = {p: i for i, p in enumerate(train_prots_list)}

# Interaction matrix: drugs × proteins (max pKi per pair)
log.info("  Building training interaction matrix...")
pair_max = {}
for _, r in train_df.iterrows():
    di = train_drug_idx.get(r.ligand_smiles, -1)
    pi = train_prot_idx.get(r.uniprot_id, -1)
    if di < 0 or pi < 0:
        continue
    key = (di, pi)
    if key not in pair_max or r.pki > pair_max[key]:
        pair_max[key] = r.pki

n_train_drugs = len(train_drugs)
n_train_prots = len(train_prots_list)
INT_DENSE = np.zeros((n_train_drugs, n_train_prots), dtype=np.float32)
for (di, pi), val in pair_max.items():
    INT_DENSE[di, pi] = val
log.info(f"  Interaction matrix: {INT_DENSE.shape}, nnz={np.count_nonzero(INT_DENSE)}")

# Training protein embedding matrix (pooled Raygun)
train_emb_mat = np.array([raygun_embs[p].mean(dim=0).numpy() for p in train_prots_list])
train_emb_norms = np.linalg.norm(train_emb_mat, axis=1, keepdims=True) + 1e-10
train_emb_normed = train_emb_mat / train_emb_norms

# Training drug FP matrix
train_fp_mat = np.array([np.array(fp_dict[d], dtype=np.float32) for d in train_drugs])

# Canonicalize training drugs for masking
from rdkit import Chem
def canon(s):
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else s
    except: return s

log.info("  Canonicalizing training drugs...")
train_canon = {d: canon(d) for d in train_drugs}
train_canonical_set = set(train_canon.values())


def drug_sims(query_fps, query_smiles):
    """Tanimoto similarity, mask exact-match drugs."""
    inter = query_fps @ train_fp_mat.T
    q_bits = query_fps.sum(1, keepdims=True)
    db_bits = train_fp_mat.sum(1, keepdims=True).T
    sims = inter / np.maximum(q_bits + db_bits - inter, 1)
    # Mask exact-match drugs
    q_canon = [canon(s) for s in query_smiles]
    for qi, qc in enumerate(q_canon):
        for di, d in enumerate(train_drugs):
            if train_canon.get(d) == qc:
                sims[qi, di] = -1
    return sims


def prot_sims(query_embs):
    """Cosine similarity to training proteins."""
    qn = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-10)
    return qn @ train_emb_normed.T


# Compute similarities for test set
test_drugs_uniq = sorted(test_df.ligand_smiles.unique())
test_prots_uniq = sorted(test_df.uniprot_id.unique())
test_drug_map = {d: i for i, d in enumerate(test_drugs_uniq)}
test_prot_map = {p: i for i, p in enumerate(test_prots_uniq)}

test_fps = np.array([np.array(fp_dict[d], dtype=np.float32) for d in test_drugs_uniq])
test_embs = np.array([raygun_embs[p].mean(dim=0).numpy() for p in test_prots_uniq])

log.info("  Computing similarities...")
ds = drug_sims(test_fps, test_drugs_uniq)
ps = prot_sims(test_embs)
log.info(f"  Drug sims: {ds.shape}, Prot sims: {ps.shape}")

# Map test interactions to indices
di_arr = np.array([test_drug_map.get(s, -1) for s in test_df.ligand_smiles.values])
pi_arr = np.array([test_prot_map.get(p, -1) for p in test_df.uniprot_id.values])


def run_prot_knn(k):
    """Prot-kNN: for each test protein, find k nearest train proteins."""
    n = len(di_arr)
    preds = np.full(n, np.nan)
    for qpi_idx in np.unique(pi_arr[pi_arr >= 0]):
        mask_rows = pi_arr == qpi_idx
        top_k = np.argsort(ps[qpi_idx])[-k:][::-1]
        top_sims = ps[qpi_idx, top_k]
        valid = top_sims > 0
        top_k, top_sims = top_k[valid], top_sims[valid]
        if len(top_k) == 0:
            continue
        # For each test interaction with this protein
        for ri in np.where(mask_rows)[0]:
            qdi = di_arr[ri]
            if qdi < 0:
                continue
            drug_s = ds[qdi]
            best_pkis, best_wts = [], []
            for ki, bpi in enumerate(top_k):
                int_col = INT_DENSE[:, bpi]
                has_int = int_col > 0
                if not has_int.any():
                    continue
                masked = np.where(has_int, drug_s, -2)
                best_di = np.argmax(masked)
                if masked[best_di] <= 0:
                    continue
                best_pkis.append(int_col[best_di])
                best_wts.append(top_sims[ki])
            if best_pkis:
                preds[ri] = np.average(best_pkis, weights=best_wts)
    return preds


log.info("  Running Prot-kNN k=1...")
t0 = time.time()
test_df["prot_knn_k1"] = run_prot_knn(1)
log.info(f"    Done in {time.time()-t0:.0f}s, "
         f"{sum(~np.isnan(test_df.prot_knn_k1))}/{len(test_df)} predictions")

log.info("  Running Prot-kNN k=5...")
t0 = time.time()
test_df["prot_knn_k5"] = run_prot_knn(5)
log.info(f"    Done in {time.time()-t0:.0f}s, "
         f"{sum(~np.isnan(test_df.prot_knn_k5))}/{len(test_df)} predictions")


# ── 5d: Report results ────────────────────────────────────────
methods = ["concise_pred", "anchor_pred", "prot_knn_k1", "prot_knn_k5"]
method_names = ["CoNCISE", "ConciseAnchor", "Prot-kNN k=1", "Prot-kNN k=5"]

log.info(f"\n{'='*60}")
log.info(f"  TEST SET RESULTS ({len(test_df)} int, {test_df.uniprot_id.nunique()} proteins)")
log.info(f"  {'Method':<18s} {'CI':>7s} {'AUROC':>7s} {'RMSE':>7s} {'NaN%':>6s}")
log.info(f"  {'-'*50}")

for col, name in zip(methods, method_names):
    ci = pp_ci(test_df, col)
    auc = pp_auroc(test_df, col)
    rmse = pp_rmse(test_df, col)
    nan_pct = 100 * np.isnan(test_df[col].values.astype(float)).mean()
    log.info(f"  {name:<18s} {ci.mean():7.3f} "
             f"{auc.mean() if len(auc) else 0:7.3f} "
             f"{rmse.mean() if len(rmse) else 0:7.3f} "
             f"{nan_pct:6.1f}%")

log.info(f"  {'-'*50}")


# ── 5e: Per-anchor-quartile analysis ──────────────────────────
log.info(f"\n  ── Anchor pKi Quartile Breakdown ──")
valid_anchor = test_df.dropna(subset=["anchor_pki"])
if len(valid_anchor) > 20:
    q_edges = np.quantile(valid_anchor.anchor_pki.values, [0, 0.25, 0.5, 0.75, 1.0])
    q_labels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
    valid_anchor = valid_anchor.copy()
    valid_anchor["anchor_q"] = pd.cut(valid_anchor.anchor_pki, bins=q_edges,
                                       labels=q_labels, include_lowest=True)

    for col, name in zip(methods, method_names):
        log.info(f"\n  {name}:")
        for ql in q_labels:
            qdf = valid_anchor[valid_anchor.anchor_q == ql]
            if len(qdf) < 5:
                continue
            ci = pp_ci(qdf, col)
            rmse = pp_rmse(qdf, col)
            auc = pp_auroc(qdf, col)
            log.info(f"    {ql:<16s}  CI={ci.mean():.3f}  RMSE={rmse.mean() if len(rmse) else 0:.3f}  "
                     f"AUROC={auc.mean() if len(auc) else 0:.3f}  (n={len(qdf)}, {qdf.uniprot_id.nunique()} prot)")


# ── Save results ──────────────────────────────────────────────
out_csv = RESULTS / "bdb_identity_split_50_results.csv"
test_df.to_csv(out_csv, index=False)
log.info(f"\nSaved results to {out_csv}")

log.info("\nDone!")
