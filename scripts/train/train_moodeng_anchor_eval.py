"""Evaluate pretrained CoNCISE + train ConciseAnchor on MooDengDB.

1. Load pretrained CoNCISE (their published model) — evaluate only
2. Train ConciseAnchor (bilinear, BCE) on MooDengDB train split
3. Compare both on test set with AUROC/AUPR
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ================================================================
# Load MooDengDB
# ================================================================
log.info("Loading MooDengDB...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
val_raw = pd.read_csv(MOODENG / "val.csv", low_memory=False)
test_raw = pd.read_csv(MOODENG / "test.csv", low_memory=False)
for df in [train_raw, val_raw, test_raw]:
    df.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)

import hashlib
def seq_to_id(seq):
    """Deterministic protein ID from sequence using md5."""
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

all_seqs = {}
for df in [train_raw, val_raw, test_raw]:
    df["prot_id"] = df.sequence.apply(seq_to_id)
    for _, r in df.drop_duplicates("prot_id").iterrows():
        all_seqs[r.prot_id] = r.sequence
log.info(f"Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}, "
         f"Proteins: {len(all_seqs)}")

# ================================================================
# Load cached Raygun embeddings + Morgan FPs
# ================================================================
RAYGUN_CACHE = RESULTS / "raygun_moodeng_embeddings.pt"
FP_CACHE = RESULTS / "morgan_moodeng_fp.pkl"

if not RAYGUN_CACHE.exists():
    log.error(f"Raygun cache not found at {RAYGUN_CACHE}. Run embedding computation first.")
    sys.exit(1)

log.info("Loading cached Raygun embeddings...")
raygun_embs = torch.load(RAYGUN_CACHE, map_location="cpu", weights_only=False)
log.info(f"Raygun: {len(raygun_embs)} proteins, shape: {next(iter(raygun_embs.values())).shape}")

if FP_CACHE.exists():
    fp_dict = pickle.load(open(FP_CACHE, "rb"))
    log.info(f"FPs: {len(fp_dict)} drugs")
else:
    log.info("Computing Morgan FPs...")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    all_smiles = set()
    for df in [train_raw, val_raw, test_raw]:
        all_smiles.update(df.smiles.unique())
    fp_dict = {}
    for smi in all_smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp_dict[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
        except: pass
    pickle.dump(fp_dict, open(FP_CACHE, "wb"))
    log.info(f"  Computed {len(fp_dict)}/{len(all_smiles)} FPs")

# Filter + drop sequences to save memory
train_df = train_raw.loc[train_raw.prot_id.isin(raygun_embs) & train_raw.smiles.isin(fp_dict),
                          ["prot_id", "smiles", "label"]].copy()
val_df = val_raw.loc[val_raw.prot_id.isin(raygun_embs) & val_raw.smiles.isin(fp_dict),
                      ["prot_id", "smiles", "label"]].copy()
test_df = test_raw.loc[test_raw.prot_id.isin(raygun_embs) & test_raw.smiles.isin(fp_dict),
                        ["prot_id", "smiles", "label"]].copy()
del train_raw, val_raw, test_raw
import gc; gc.collect()
log.info(f"Train: {len(train_df)} ({(train_df.label==1).sum()} pos), "
         f"Val: {len(val_df)}, Test: {len(test_df)}")

# ================================================================
# Build anchors from training positives
# ================================================================
log.info("Building anchor pool...")
train_pos = train_df[train_df.label == 1]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors:
        drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs with positive binders")

# ================================================================
# Metrics
# ================================================================
from sklearn.metrics import roc_auc_score, average_precision_score

# ================================================================
# PHASE 1: Load pretrained CoNCISE and evaluate on test
# ================================================================
log.info(f"\n{'='*60}")
log.info("PHASE 1: Pretrained CoNCISE evaluation")
log.info(f"{'='*60}")

log.info("Loading pretrained CoNCISE (v2)...")
concise_model = torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v2", pretrained=True)
concise_model = concise_model.eval().to(DEVICE)
log.info(f"  Params: {sum(p.numel() for p in concise_model.parameters()):,}")

log.info("CoNCISE predictions on test set...")
concise_preds = []
with torch.no_grad():
    for i in range(0, len(test_df), 64):
        batch = test_df.iloc[i:i+64]
        fps_batch = torch.tensor(np.array([fp_dict[s] for s in batch.smiles])).to(DEVICE)
        embs_batch = torch.stack([raygun_embs[p] for p in batch.prot_id]).to(DEVICE)
        pred = concise_model(fps_batch, embs_batch, is_morgan_fingerprint=True)["binding"]
        # Cosine output [-1,1] -> [0,1] for AUROC
        scores = ((pred + 1) / 2).cpu()
        concise_preds.extend(scores.tolist() if scores.dim() > 0 else [scores.item()])
        if (i + 64) % 10000 < 64:
            log.info(f"  {min(i+64, len(test_df))}/{len(test_df)}")

test_labels = test_df.label.values.astype(float)
concise_preds = np.array(concise_preds)
auroc_c = roc_auc_score(test_labels, concise_preds)
aupr_c = average_precision_score(test_labels, concise_preds)
log.info(f"  Pretrained CoNCISE: AUROC={auroc_c:.4f}, AUPR={aupr_c:.4f}")
del concise_model; torch.cuda.empty_cache()

# ================================================================
# PHASE 2: Train ConciseAnchor (bilinear, BCE)
# ================================================================
log.info(f"\n{'='*60}")
log.info("PHASE 2: Train ConciseAnchor")
log.info(f"{'='*60}")

from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class AnchorBinaryDS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, drug_to_anchors):
        fps, anchor_pids, query_pids, labels = [], [], [], []
        for _, r in df.iterrows():
            pid, smi, label = r.prot_id, r.smiles, float(r.label)
            if smi not in drug_to_anchors or smi not in fp_dict or pid not in raygun_embs: continue
            anchor = None
            for a in drug_to_anchors[smi]:
                if a != pid: anchor = a; break
            if anchor is None: continue
            fps.append(fp_dict[smi])
            anchor_pids.append(anchor); query_pids.append(pid); labels.append(label)
        self.fps = torch.tensor(np.array(fps))
        self.anchor_pids = anchor_pids; self.query_pids = query_pids
        self.labels = torch.tensor(labels, dtype=torch.float32)
        log.info(f"  AnchorBinaryDS: {len(self.labels)} ({(self.labels==1).sum().item()} pos)")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return (self.fps[i], raygun_embs[self.anchor_pids[i]],
                raygun_embs[self.query_pids[i]], self.labels[i])

class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(
            ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)
    def forward(self, drug_fp, anchor_emb, query_emb):
        logit = self.backbone(drug_fp, anchor_emb, query_emb)
        return logit  # raw logit, use BCEWithLogitsLoss

ANCHOR_DIR = Path("models/concise_anchor_moodeng"); ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_BEST = ANCHOR_DIR / "best_model.pt"

if ANCHOR_BEST.exists():
    log.info(f"ConciseAnchor exists at {ANCHOR_BEST}, skipping training")
else:
    log.info("Training ConciseAnchor (bilinear, BCEWithLogits)...")
    model_a = ConciseAnchorBinary().to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad):,}")
    train_ds = AnchorBinaryDS(train_df, fp_dict, raygun_embs, drug_to_anchors)
    val_ds = AnchorBinaryDS(val_df, fp_dict, raygun_embs, drug_to_anchors)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_aupr = -1
    for ep in range(1, 21):
        t0 = time.time(); model_a.train(); total_loss, nb = 0, 0
        for drug, anchor, query, label in tqdm(train_loader, desc=f"Anchor Ep {ep}", leave=False):
            drug, anchor, query, label = drug.to(DEVICE), anchor.to(DEVICE), query.to(DEVICE), label.to(DEVICE)
            logit = model_a(drug, anchor, query)
            loss = criterion(logit, label)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(label); nb += len(label)

        model_a.eval(); vp, vt = [], []
        with torch.no_grad():
            for d, a, q, l in val_loader:
                logit = model_a(d.to(DEVICE), a.to(DEVICE), q.to(DEVICE))
                vp.extend(torch.sigmoid(logit).cpu().tolist()); vt.extend(l.tolist())
        vt, vp = np.array(vt), np.array(vp)
        val_auroc = roc_auc_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        val_aupr = average_precision_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        tag = "*BEST*" if val_aupr > best_aupr else ""
        if val_aupr > best_aupr:
            best_aupr = val_aupr
            torch.save({"model_state_dict": model_a.state_dict(), "epoch": ep}, str(ANCHOR_BEST))
        log.info(f"  Anchor Ep {ep} [{time.time()-t0:.0f}s] Loss={total_loss/nb:.4f} "
                 f"AUROC={val_auroc:.4f} AUPR={val_aupr:.4f} {tag}")
    del model_a; torch.cuda.empty_cache()

# ================================================================
# PHASE 3: Evaluate ConciseAnchor on test set
# ================================================================
log.info(f"\n{'='*60}")
log.info("PHASE 3: ConciseAnchor evaluation on test set")
log.info(f"{'='*60}")

model_a = ConciseAnchorBinary().to(DEVICE)
ckpt = torch.load(str(ANCHOR_BEST), map_location=DEVICE, weights_only=False)
model_a.load_state_dict(ckpt["model_state_dict"]); model_a.eval()

anchor_preds = np.full(len(test_df), np.nan)
with torch.no_grad():
    for i, (_, r) in enumerate(test_df.iterrows()):
        if r.smiles not in drug_to_anchors or r.prot_id not in raygun_embs: continue
        anchor = None
        for a in drug_to_anchors[r.smiles]:
            if a != r.prot_id: anchor = a; break
        if anchor is None: continue
        fp = torch.tensor(fp_dict[r.smiles]).unsqueeze(0).to(DEVICE)
        anc = raygun_embs[anchor].unsqueeze(0).to(DEVICE)
        qry = raygun_embs[r.prot_id].unsqueeze(0).to(DEVICE)
        anchor_preds[i] = torch.sigmoid(model_a(fp, anc, qry)).item()
del model_a; torch.cuda.empty_cache()

valid = ~np.isnan(anchor_preds)
log.info(f"ConciseAnchor: {valid.sum()}/{len(test_df)} predictions with anchors")

# ================================================================
# FINAL RESULTS
# ================================================================
log.info(f"\n{'='*60}")
log.info(f"  FINAL RESULTS — MooDengDB Test Set ({len(test_df)} interactions)")
log.info(f"  {'Method':<25s} {'AUROC':>8s} {'AUPR':>8s}  {'Coverage':>10s}")
log.info(f"  {'-'*55}")
log.info(f"  {'Pretrained CoNCISE':<25s} {auroc_c:8.4f} {aupr_c:8.4f}  {len(test_df)}/{len(test_df)}")

if valid.sum() > 10:
    auroc_a = roc_auc_score(test_labels[valid], anchor_preds[valid])
    aupr_a = average_precision_score(test_labels[valid], anchor_preds[valid])
    log.info(f"  {'ConciseAnchor':<25s} {auroc_a:8.4f} {aupr_a:8.4f}  {valid.sum()}/{len(test_df)}")

    # Fair comparison on same subset
    auroc_c_sub = roc_auc_score(test_labels[valid], concise_preds[valid])
    aupr_c_sub = average_precision_score(test_labels[valid], concise_preds[valid])
    log.info(f"  {'CoNCISE (same subset)':<25s} {auroc_c_sub:8.4f} {aupr_c_sub:8.4f}  {valid.sum()}/{len(test_df)}")

log.info(f"  {'-'*55}")

# Save
out = test_df.copy()
out["concise_pred"] = concise_preds
out["anchor_pred"] = anchor_preds
out.to_csv(RESULTS / "moodeng_test_results.csv", index=False)
log.info(f"\nSaved results/moodeng_test_results.csv")
log.info("Done!")
