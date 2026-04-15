"""Train ConciseAnchor-Bilinear on DTC (10 epochs) + eval on same test set as kNN.

Steps:
  1. Compute & cache Morgan FPs (all cores)
  2. Compute & cache ESM2→Raygun embeddings (GPU, batch=8)
  3. Train ConciseAnchor-Bilinear for 10 epochs
  4. Eval on cold-protein test set with per-protein CI/AUROC/RMSE + Q1-Q4
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from multiprocessing import Pool, cpu_count
from sklearn.metrics import roc_auc_score, mean_squared_error
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = PROJECT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
N_WORKERS = cpu_count()

log.info(f"ConciseAnchor train+eval — {N_WORKERS} cores, GPU={DEVICE}")


# ══════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════
dtc_path = PROJECT / "embeddings_model_files" / "dtc_training_interactions.csv"
if not dtc_path.exists():
    dtc_path = PROJECT / "data" / "processed" / "dtc_training_interactions.csv"
dtc = pd.read_csv(dtc_path)
seqs = json.load(open(PROJECT / "data" / "processed" / "merged_sequences.json"))
log.info(f"DTC: {len(dtc)} int, {dtc.uniprot_id.nunique()} prot, {dtc.ligand_smiles.nunique()} drugs")
log.info(f"Sequences: {len(seqs)} proteins")


# ══════════════════════════════════════════════════════════════════
# 2. Compute & cache Morgan FPs
# ══════════════════════════════════════════════════════════════════
from rdkit import Chem
from rdkit.Chem import AllChem


def _compute_fp(s):
    m = Chem.MolFromSmiles(s)
    if m:
        return s, np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048), dtype=np.float32)
    return s, None


FP_CACHE = RESULTS_DIR / "concise_morgan_fp.pkl"
if FP_CACHE.exists():
    log.info("Loading cached Morgan FPs...")
    with open(FP_CACHE, "rb") as f:
        fp_dict = pickle.load(f)
else:
    all_drugs = sorted(dtc.ligand_smiles.unique())
    log.info(f"Computing Morgan FPs for {len(all_drugs)} drugs ({N_WORKERS} workers)...")
    with Pool(N_WORKERS) as pool:
        results = list(tqdm(pool.imap(_compute_fp, all_drugs, chunksize=500),
                           total=len(all_drugs), desc="Morgan FP"))
    fp_dict = {s: fp for s, fp in results if fp is not None}
    with open(FP_CACHE, "wb") as f:
        pickle.dump(fp_dict, f)
log.info(f"Morgan FPs: {len(fp_dict)} drugs")


# ══════════════════════════════════════════════════════════════════
# 3. Compute & cache Raygun embeddings
# ══════════════════════════════════════════════════════════════════
RAYGUN_CACHE = RESULTS_DIR / "raygun_embeddings.pt"
if RAYGUN_CACHE.exists():
    log.info("Loading cached Raygun embeddings...")
    raygun_embs = torch.load(RAYGUN_CACHE, map_location="cpu", weights_only=False)
else:
    log.info("Computing ESM-2 650M → Raygun embeddings on GPU...")
    import esm

    # All proteins that have sequences
    all_prots = sorted(set(dtc.uniprot_id) & set(seqs.keys()))
    log.info(f"  {len(all_prots)} proteins with sequences")

    # ESM-2
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(DEVICE).eval()

    esm_embeddings = {}
    BS = 8
    with torch.no_grad():
        for i in tqdm(range(0, len(all_prots), BS), desc="ESM-2"):
            batch = [(u, seqs[u][:1022]) for u in all_prots[i:i+BS] if u in seqs]
            if not batch:
                continue
            _, _, tokens = bc(batch)
            out = esm_model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            for j, (u, s) in enumerate(batch):
                esm_embeddings[u] = out["representations"][33][j:j+1, 1:len(s)+1, :].cpu()

    del esm_model
    torch.cuda.empty_cache()
    log.info(f"  ESM-2 done: {len(esm_embeddings)} proteins")

    # Raygun
    log.info("  Running Raygun encoder...")
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M",
                                     trust_repo=True)
    raymodel = raymodel.to(DEVICE).eval()

    raygun_embs = {}
    with torch.no_grad():
        for uid, emb in tqdm(esm_embeddings.items(), desc="Raygun"):
            ray_enc = raymodel.encoder(emb.to(DEVICE)).squeeze(0).cpu()  # (50, 1280)
            raygun_embs[uid] = ray_enc

    del raymodel, esm_embeddings
    torch.cuda.empty_cache()
    torch.save(raygun_embs, RAYGUN_CACHE)
    log.info(f"  Raygun saved: {len(raygun_embs)} proteins, shape={next(iter(raygun_embs.values())).shape}")

log.info(f"Raygun: {len(raygun_embs)} proteins")


# ══════════════════════════════════════════════════════════════════
# 4. Split — SAME as kNN (seed 42, 10/10/80 by protein)
# ══════════════════════════════════════════════════════════════════
# Use ESM2 protein set for split (same as kNN script)
esm2_path = PROJECT / "embeddings_model_files" / "esm2_650m_dtc.pt"
esm2_raw = torch.load(esm2_path, map_location="cpu", weights_only=False)
esm2_keys = set(esm2_raw.keys())
del esm2_raw

random.seed(42)
dtc_prots = sorted(set(dtc.uniprot_id) & esm2_keys)
random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1))
nv = max(1, int(len(dtc_prots) * 0.1))
test_prots = set(dtc_prots[:nt])
val_prots = set(dtc_prots[nt:nt + nv])
train_prots = set(dtc_prots[nt + nv:])

# Filter by Raygun + FP availability
dtc_filt = dtc[dtc.uniprot_id.isin(raygun_embs.keys()) & dtc.ligand_smiles.isin(fp_dict)].copy()
train_df = dtc_filt[dtc_filt.uniprot_id.isin(train_prots)]
val_df = dtc_filt[dtc_filt.uniprot_id.isin(val_prots)]
test_df = dtc_filt[dtc_filt.uniprot_id.isin(test_prots)]
log.info(f"Train: {len(train_df)} ({train_df.uniprot_id.nunique()} prot)")
log.info(f"Val:   {len(val_df)} ({val_df.uniprot_id.nunique()} prot)")
log.info(f"Test:  {len(test_df)} ({test_df.uniprot_id.nunique()} prot)")


# ══════════════════════════════════════════════════════════════════
# 5. Build anchors: strongest binder per drug (pKi >= 7)
# ══════════════════════════════════════════════════════════════════
drug_to_anchor = {}
drug_to_second = {}
for smi, grp in train_df.groupby("ligand_smiles"):
    s = grp.sort_values("pki", ascending=False)
    uids, pkis = s.uniprot_id.values, s.pki.values
    if pkis[0] >= 7.0 and uids[0] in raygun_embs:
        drug_to_anchor[smi] = (uids[0], pkis[0])
        if len(uids) > 1 and uids[1] in raygun_embs:
            drug_to_second[smi] = (uids[1], pkis[1])
log.info(f"Anchors: {len(drug_to_anchor)} drugs with pKi >= 7")


# ══════════════════════════════════════════════════════════════════
# 6. Dataset
# ══════════════════════════════════════════════════════════════════
class ConciseAnchorDataset(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, drug_to_anchor, drug_to_second):
        self.raygun_embs = raygun_embs
        self.rows = []
        fps = []
        for _, row in df.iterrows():
            uid, smi, pki = row["uniprot_id"], row["ligand_smiles"], row["pki"]
            if smi not in drug_to_anchor or smi not in fp_dict or uid not in raygun_embs:
                continue
            au, ap = drug_to_anchor[smi]
            if au == uid:
                if smi not in drug_to_second:
                    continue
                au, ap = drug_to_second[smi]
            if au not in raygun_embs:
                continue
            self.rows.append((uid, smi, pki, au))
            fps.append(fp_dict[smi])

        self.drug_fps = torch.tensor(np.array(fps))
        self.pkis = torch.tensor([r[2] for r in self.rows], dtype=torch.float32)
        log.info(f"  Dataset: {len(self.rows)} interactions")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, smi, pki, au = self.rows[i]
        return self.drug_fps[i], self.raygun_embs[au], self.raygun_embs[uid], self.pkis[i]


log.info("Building datasets...")
train_ds = ConciseAnchorDataset(train_df, fp_dict, raygun_embs, drug_to_anchor, drug_to_second)
val_ds = ConciseAnchorDataset(val_df, fp_dict, raygun_embs, drug_to_anchor, drug_to_second)
test_ds = ConciseAnchorDataset(test_df, fp_dict, raygun_embs, drug_to_anchor, drug_to_second)

BATCH_SIZE = 512
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True, persistent_workers=True)


# ══════════════════════════════════════════════════════════════════
# 7. Model
# ══════════════════════════════════════════════════════════════════
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

model = ConciseAnchorBilinear(
    ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2
).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"ConciseAnchor-Bilinear: {n_params:,} parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

MODEL_DIR = PROJECT / "models" / "concise_anchor_bilinear_dtc_10ep"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
best_val = float("inf")


# ══════════════════════════════════════════════════════════════════
# 8. Train 10 epochs
# ══════════════════════════════════════════════════════════════════
log.info(f"\n{'='*70}")
log.info("TRAINING — 10 epochs")
log.info(f"{'='*70}")

for ep in range(1, 11):
    t0 = time.time()
    model.train()
    total_loss, nb = 0, 0
    for drug, anchor, query, pki in tqdm(train_loader, desc=f"Ep {ep:2d}", leave=False):
        drug = drug.to(DEVICE, non_blocking=True)
        anchor = anchor.to(DEVICE, non_blocking=True)
        query = query.to(DEVICE, non_blocking=True)
        pki = pki.to(DEVICE, non_blocking=True)

        pred = model(drug, anchor, query)
        loss = F.mse_loss(pred, pki)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(pki)
        nb += len(pki)
    scheduler.step()

    # Validation
    model.eval()
    val_preds, val_true = [], []
    with torch.no_grad():
        for drug, anchor, query, pki in val_loader:
            pred = model(drug.to(DEVICE), anchor.to(DEVICE), query.to(DEVICE))
            val_preds.extend(pred.cpu().tolist())
            val_true.extend(pki.tolist())

    val_true = np.array(val_true)
    val_preds_arr = np.array(val_preds)
    val_loss = np.mean((val_true - val_preds_arr) ** 2)
    val_r = np.corrcoef(val_true, val_preds_arr)[0, 1] if len(val_true) > 1 else 0

    improved = val_loss < best_val
    if improved:
        best_val = val_loss
        torch.save({"model_state_dict": model.state_dict(), "epoch": ep},
                    MODEL_DIR / "best_model.pt")

    log.info(f"Ep {ep:2d} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} "
             f"r={val_r:.4f} {'*BEST*' if improved else ''}")

log.info("Training complete. Loading best model...")
ckpt = torch.load(MODEL_DIR / "best_model.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
log.info(f"Loaded best model from epoch {ckpt['epoch']}")


# ══════════════════════════════════════════════════════════════════
# 9. Evaluate on test set
# ══════════════════════════════════════════════════════════════════
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


log.info(f"\n{'='*70}")
log.info(f"EVALUATION — Test set ({test_df.uniprot_id.nunique()} proteins)")
log.info(f"{'='*70}")

model.eval()
test_preds, test_true, test_uids, test_smiles = [], [], [], []
with torch.no_grad():
    for drug, anchor, query, pki in test_loader:
        pred = model(drug.to(DEVICE), anchor.to(DEVICE), query.to(DEVICE))
        test_preds.extend(pred.cpu().tolist())
        test_true.extend(pki.tolist())

# Map predictions back to test dataset rows
eval_rows = []
for i, (uid, smi, pki, au) in enumerate(test_ds.rows):
    eval_rows.append({"uniprot_id": uid, "ligand_smiles": smi, "pki": pki,
                       "anchor_uid": au, "concise_anchor": test_preds[i]})
eval_df = pd.DataFrame(eval_rows)

# Overall metrics
ci, auc, rmse = pp_metrics(eval_df, "concise_anchor")
nan_pct = eval_df.concise_anchor.isna().mean() * 100
coverage = len(eval_df) / len(test_df) * 100

log.info(f"\n  ConciseAnchor-Bilinear (10ep)")
log.info(f"  CI={ci.mean():.4f}  AUROC={auc.mean() if len(auc) else 0:.4f}  "
         f"RMSE={rmse.mean() if len(rmse) else 0:.4f}  NaN={nan_pct:.1f}%")
log.info(f"  Coverage: {len(eval_df)}/{len(test_df)} ({coverage:.1f}%)")
log.info(f"  Eval proteins: {eval_df.uniprot_id.nunique()}")

# Q1-Q4 by pKi quartile
q25, q50, q75 = np.percentile(eval_df.pki.values, [25, 50, 75])
eval_df["pki_quartile"] = pd.cut(eval_df.pki, bins=[-np.inf, q25, q50, q75, np.inf],
                                  labels=["Q1", "Q2", "Q3", "Q4"])
log.info(f"\n  pKi quartiles: Q1<={q25:.2f}, Q2<={q50:.2f}, Q3<={q75:.2f}, Q4>{q75:.2f}")
for q in ["Q1", "Q2", "Q3", "Q4"]:
    sub = eval_df[eval_df.pki_quartile == q]
    if len(sub) >= 10:
        ci_q, auc_q, rmse_q = pp_metrics(sub, "concise_anchor")
        log.info(f"    {q}: CI={ci_q.mean():.4f}, AUROC={auc_q.mean() if len(auc_q) else 0:.4f}, "
                 f"RMSE={rmse_q.mean() if len(rmse_q) else 0:.4f}, n={len(sub)}")

# Save
eval_df.to_csv(RESULTS_DIR / "concise_anchor_dtc_test.csv", index=False)
summary = pd.DataFrame([{
    "method": "ConciseAnchor-Bilinear-10ep", "ci": ci.mean(),
    "auroc": auc.mean() if len(auc) else np.nan,
    "rmse": rmse.mean() if len(rmse) else np.nan,
    "nan_pct": nan_pct, "coverage_pct": coverage,
    "n_proteins": eval_df.uniprot_id.nunique(),
    "n_interactions": len(eval_df),
}])
summary.to_csv(RESULTS_DIR / "concise_anchor_dtc_summary.csv", index=False)

log.info(f"\n{'='*70}")
log.info("DONE")
