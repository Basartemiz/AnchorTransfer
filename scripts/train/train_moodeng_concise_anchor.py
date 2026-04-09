"""Train CoNCISE + ConciseAnchor on MooDengDB with binary classification.

Uses their exact train/val/test splits. Binary labels (binds/not binds).
CoNCISE: original architecture (cosine_prediction=True, drug_dim=256, gelu).
ConciseAnchor: bilinear head with BCE loss + anchor from positive examples.
Evaluates both on test set with bootstrap CI.
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
sys.path.insert(0, "src/src")
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ================================================================
# PHASE 0: Load MooDengDB
# ================================================================
log.info("Loading MooDengDB...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
val_raw = pd.read_csv(MOODENG / "val.csv", low_memory=False)
test_raw = pd.read_csv(MOODENG / "test.csv", low_memory=False)
log.info(f"Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

# Use 'Remapped Entry' as protein sequence (it's the actual target sequence for Raygun)
# 'Target Sequence' column has a different (possibly query) sequence
# Actually both contain sequences. Let's use Target Sequence for embedding.
for df in [train_raw, val_raw, test_raw]:
    df.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)

# Build unique protein ID from sequence hash (MooDengDB uses sequences, not UniProt IDs)
import hashlib
def seq_to_id(seq):
    """Deterministic protein ID from sequence using md5."""
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

all_seqs = {}
for df in [train_raw, val_raw, test_raw]:
    df["prot_id"] = df.sequence.apply(seq_to_id)
    for _, r in df.drop_duplicates("prot_id").iterrows():
        all_seqs[r.prot_id] = r.sequence

log.info(f"Unique proteins: {len(all_seqs)}, train: {train_raw.prot_id.nunique()}, "
         f"val: {val_raw.prot_id.nunique()}, test: {test_raw.prot_id.nunique()}")

# ================================================================
# PHASE 1: Compute Raygun embeddings for all proteins
# ================================================================
RAYGUN_CACHE = RESULTS / "raygun_moodeng_embeddings.pt"

if RAYGUN_CACHE.exists():
    raygun_embs = torch.load(RAYGUN_CACHE, map_location="cpu", weights_only=False)
    log.info(f"Loaded Raygun: {len(raygun_embs)} proteins")
else:
    log.info(f"Computing Raygun for {len(all_seqs)} MooDengDB proteins...")
    log.info("  Streaming ESM2 -> Raygun (no intermediate storage to avoid OOM)...")
    import esm

    # Load BOTH models at once, process ESM2->Raygun per protein, discard ESM2 immediately
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(DEVICE)

    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)

    items = [(pid, seq[:1022].upper()) for pid, seq in all_seqs.items() if len(seq) >= 25]
    log.info(f"  {len(items)} proteins to process")
    raygun_embs = {}

    # Process ONE protein at a time: ESM2 -> Raygun -> save, discard ESM2 embedding
    for idx, (pid, seq) in enumerate(items):
        try:
            _, _, toks = bc([(pid, seq)])
            with torch.no_grad():
                out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
                esm_emb = out["representations"][33][0, 1:len(seq)+1, :]  # stays on GPU
                ray_out = raygun_model(esm_emb.unsqueeze(0))
                # ray_out is tuple: (pass-through, compressed). Use [1] = (50,1280)
                proj = ray_out[1] if isinstance(ray_out, tuple) else ray_out
                raygun_embs[pid] = proj.squeeze(0).cpu()  # (50,1280) fixed-length
            del out, esm_emb, ray_out, proj, toks
        except Exception as e:
            if "modulo by zero" not in str(e):  # skip Raygun short-seq errors silently
                log.info(f"  Error {pid} (len={len(seq)}): {e}")
        if (idx+1) % 200 == 0:
            log.info(f"    {idx+1}/{len(items)} done")
            # Periodic checkpoint to avoid losing all work on crash
            if (idx+1) % 2000 == 0:
                torch.save(raygun_embs, str(RAYGUN_CACHE) + ".partial")
                log.info(f"    Checkpoint: {len(raygun_embs)} saved")

    del esm_model, raygun_model; torch.cuda.empty_cache()
    import gc; gc.collect()
    torch.save(raygun_embs, str(RAYGUN_CACHE))
    log.info(f"  Saved {len(raygun_embs)} Raygun embeddings")

# ================================================================
# PHASE 2: Compute Morgan FPs for all drugs
# ================================================================
FP_CACHE = RESULTS / "morgan_moodeng_fp.pkl"

if FP_CACHE.exists():
    fp_dict = pickle.load(open(FP_CACHE, "rb"))
    log.info(f"Loaded FPs: {len(fp_dict)} drugs")
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

# Filter to proteins+drugs with embeddings; drop heavy sequence columns to save RAM
for name, df in [("train", train_raw), ("val", val_raw), ("test", test_raw)]:
    mask = df.prot_id.isin(raygun_embs) & df.smiles.isin(fp_dict)
    n_before = len(df)
    filtered = df.loc[mask, ["prot_id", "smiles", "label"]].copy()
    if name == "train": train_df = filtered
    elif name == "val": val_df = filtered
    else: test_df = filtered
    log.info(f"  {name}: {n_before} -> {len(filtered)} after filtering")

# Free raw DataFrames (they hold full protein sequences = huge memory)
del train_raw, val_raw, test_raw, all_seqs
import gc; gc.collect()
log.info(f"Train pos: {(train_df.label==1).sum()}, neg: {(train_df.label==0).sum()}")

# ================================================================
# PHASE 3: Build anchors from training positives
# ================================================================
log.info("Building anchor pool from training positive examples...")
# For binary labels: anchor = a positive protein for the drug (drug binds to this protein)
train_pos = train_df[train_df.label == 1]
drug_to_anchors = {}
for smi, grp in train_pos.groupby("smiles"):
    anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
    if anchors:
        drug_to_anchors[smi] = anchors
log.info(f"Anchors: {len(drug_to_anchors)} drugs with positive binders in training")

# ================================================================
# Metrics
# ================================================================
from sklearn.metrics import roc_auc_score, average_precision_score

def bootstrap_mean_ci(values, n_boot=1000, ci=0.95):
    if len(values) < 2: return np.mean(values), np.nan, np.nan
    rng = np.random.RandomState(42)
    means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(means, (1-ci)/2 * 100)
    hi = np.percentile(means, (1+ci)/2 * 100)
    return np.mean(values), lo, hi

# ================================================================
# PHASE 4: Train CoNCISE (original architecture, BCE loss)
# ================================================================
from concise.model.concise import Concise

class BinaryDS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs):
        fps, pids, labels = [], [], []
        for _, r in df.iterrows():
            if r.prot_id in raygun_embs and r.smiles in fp_dict:
                fps.append(fp_dict[r.smiles])
                pids.append(r.prot_id)
                labels.append(float(r.label))
        self.fps = torch.tensor(np.array(fps))
        self.pids = pids
        self.labels = torch.tensor(labels, dtype=torch.float32)
        log.info(f"  BinaryDS: {len(self.labels)} ({(self.labels==1).sum().item()} pos)")
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.fps[i], raygun_embs[self.pids[i]], self.labels[i]

class ConciseOriginal(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Concise(drug_layers=[[32],[32],[32]], ligand_dim=2048, residue_dim=1280,
            drug_dim=256, proj_dim=256, nheads=32, activation="gelu", cosine_prediction=True)
    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]

CONCISE_DIR = Path("models/concise_moodeng"); CONCISE_DIR.mkdir(parents=True, exist_ok=True)
CONCISE_BEST = CONCISE_DIR / "best_model.pt"

if CONCISE_BEST.exists():
    log.info(f"CoNCISE exists at {CONCISE_BEST}, skipping")
else:
    log.info("Training CoNCISE (original arch, BCE loss)...")
    model_c = ConciseOriginal().to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_c.parameters() if p.requires_grad):,}")
    train_ds = BinaryDS(train_df, fp_dict, raygun_embs)
    val_ds = BinaryDS(val_df, fp_dict, raygun_embs)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    # Match their training: AdamW lr=3e-4, 20 epochs, select by val AUPR
    optimizer = torch.optim.AdamW(model_c.parameters(), lr=3e-4)
    best_aupr = -1
    for ep in range(1, 21):
        t0 = time.time(); model_c.train(); total_loss, nb = 0, 0
        for drug, prot, label in tqdm(train_loader, desc=f"CoNCISE Ep {ep}", leave=False):
            drug, prot, label = drug.to(DEVICE), prot.to(DEVICE), label.to(DEVICE)
            pred = model_c(drug, prot)
            # Cosine output [-1,1], shift to [0,1] for BCE
            pred_prob = (pred + 1) / 2
            loss = F.binary_cross_entropy(pred_prob.clamp(1e-7, 1-1e-7), label)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(label); nb += len(label)

        model_c.eval(); vp, vt = [], []
        with torch.no_grad():
            for d, p, l in val_loader:
                pred = model_c(d.to(DEVICE), p.to(DEVICE))
                vp.extend(((pred + 1) / 2).cpu().tolist()); vt.extend(l.tolist())
        vt, vp = np.array(vt), np.array(vp)
        val_auroc = roc_auc_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        val_aupr = average_precision_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        tag = "*BEST*" if val_aupr > best_aupr else ""
        if val_aupr > best_aupr:
            best_aupr = val_aupr
            torch.save({"model_state_dict": model_c.state_dict(), "epoch": ep}, str(CONCISE_BEST))
        log.info(f"  CoNCISE Ep {ep} [{time.time()-t0:.0f}s] Loss={total_loss/nb:.4f} "
                 f"AUROC={val_auroc:.4f} AUPR={val_aupr:.4f} {tag}")
    del model_c, train_ds, val_ds, train_loader, val_loader; torch.cuda.empty_cache()

# ================================================================
# PHASE 5: Train ConciseAnchor (bilinear head, BCE loss)
# ================================================================
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class AnchorBinaryDS(Dataset):
    def __init__(self, df, fp_dict, raygun_embs, drug_to_anchors):
        fps, anchor_pids, query_pids, labels = [], [], [], []
        for _, r in df.iterrows():
            pid, smi, label = r.prot_id, r.smiles, float(r.label)
            if smi not in drug_to_anchors or smi not in fp_dict or pid not in raygun_embs: continue
            # Pick anchor: a positive binder different from query protein
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

# Modify ConciseAnchorBilinear to output sigmoid for binary classification
class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(
            ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2)
        # Replace final layer bias (originally 6.5 for pKi regression)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)
    def forward(self, drug_fp, anchor_emb, query_emb):
        logit = self.backbone(drug_fp, anchor_emb, query_emb)
        return torch.sigmoid(logit)

ANCHOR_DIR = Path("models/concise_anchor_moodeng"); ANCHOR_DIR.mkdir(parents=True, exist_ok=True)
ANCHOR_BEST = ANCHOR_DIR / "best_model.pt"

if ANCHOR_BEST.exists():
    log.info(f"ConciseAnchor exists at {ANCHOR_BEST}, skipping")
else:
    log.info("Training ConciseAnchor (bilinear, BCE loss)...")
    model_a = ConciseAnchorBinary().to(DEVICE)
    log.info(f"  Params: {sum(p.numel() for p in model_a.parameters() if p.requires_grad):,}")
    train_ds = AnchorBinaryDS(train_df, fp_dict, raygun_embs, drug_to_anchors)
    val_ds = AnchorBinaryDS(val_df, fp_dict, raygun_embs, drug_to_anchors)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=4)
    optimizer = torch.optim.AdamW(model_a.parameters(), lr=3e-4)
    best_aupr = -1
    for ep in range(1, 21):
        t0 = time.time(); model_a.train(); total_loss, nb = 0, 0
        for drug, anchor, query, label in tqdm(train_loader, desc=f"Anchor Ep {ep}", leave=False):
            drug, anchor, query, label = drug.to(DEVICE), anchor.to(DEVICE), query.to(DEVICE), label.to(DEVICE)
            pred = model_a(drug, anchor, query)
            loss = F.binary_cross_entropy(pred.clamp(1e-7, 1-1e-7), label)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model_a.parameters(), 1.0); optimizer.step()
            total_loss += loss.item() * len(label); nb += len(label)

        model_a.eval(); vp, vt = [], []
        with torch.no_grad():
            for d, a, q, l in val_loader:
                pred = model_a(d.to(DEVICE), a.to(DEVICE), q.to(DEVICE))
                vp.extend(pred.cpu().tolist()); vt.extend(l.tolist())
        vt, vp = np.array(vt), np.array(vp)
        val_auroc = roc_auc_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        val_aupr = average_precision_score(vt, vp) if len(set(vt.astype(int))) > 1 else 0
        tag = "*BEST*" if val_aupr > best_aupr else ""
        if val_aupr > best_aupr:
            best_aupr = val_aupr
            torch.save({"model_state_dict": model_a.state_dict(), "epoch": ep}, str(ANCHOR_BEST))
        log.info(f"  Anchor Ep {ep} [{time.time()-t0:.0f}s] Loss={total_loss/nb:.4f} "
                 f"AUROC={val_auroc:.4f} AUPR={val_aupr:.4f} {tag}")
    del model_a, train_ds, val_ds, train_loader, val_loader; torch.cuda.empty_cache()

# ================================================================
# PHASE 6: Evaluate on test set
# ================================================================
log.info(f"\n{'='*60}")
log.info(f"TEST SET EVALUATION ({len(test_df)} interactions)")
log.info(f"{'='*60}")

# CoNCISE predictions
log.info("CoNCISE predictions on test set...")
model_c = ConciseOriginal().to(DEVICE)
ckpt = torch.load(str(CONCISE_BEST), map_location=DEVICE, weights_only=False)
model_c.load_state_dict(ckpt["model_state_dict"]); model_c.eval()
concise_preds = []
test_labels = test_df.label.values.astype(float)
with torch.no_grad():
    for i in range(0, len(test_df), 128):
        batch = test_df.iloc[i:i+128]
        fps = torch.tensor(np.array([fp_dict[s] for s in batch.smiles])).to(DEVICE)
        embs = torch.stack([raygun_embs[p] for p in batch.prot_id]).to(DEVICE)
        pred = model_c(fps, embs)
        concise_preds.extend(((pred + 1) / 2).cpu().tolist())
del model_c; torch.cuda.empty_cache()

# ConciseAnchor predictions
log.info("ConciseAnchor predictions on test set...")
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
        anchor_preds[i] = model_a(fp, anc, qry).item()
del model_a; torch.cuda.empty_cache()

# Report
concise_preds = np.array(concise_preds)
valid_anchor = ~np.isnan(anchor_preds)

log.info(f"\n  {'Method':<20s} {'AUROC (95% boot)':>25s} {'AUPR (95% boot)':>25s}")
log.info(f"  {'-'*75}")

# CoNCISE
auroc_c = roc_auc_score(test_labels, concise_preds)
aupr_c = average_precision_score(test_labels, concise_preds)
log.info(f"  {'CoNCISE':<20s} {auroc_c:7.4f}                    {aupr_c:7.4f}")

# ConciseAnchor (on subset with anchors)
if valid_anchor.sum() > 10:
    auroc_a = roc_auc_score(test_labels[valid_anchor], anchor_preds[valid_anchor])
    aupr_a = average_precision_score(test_labels[valid_anchor], anchor_preds[valid_anchor])
    log.info(f"  {'ConciseAnchor':<20s} {auroc_a:7.4f}                    {aupr_a:7.4f}"
             f"  ({valid_anchor.sum()}/{len(test_df)} with anchors)")

    # Also CoNCISE on same subset for fair comparison
    auroc_c_sub = roc_auc_score(test_labels[valid_anchor], concise_preds[valid_anchor])
    aupr_c_sub = average_precision_score(test_labels[valid_anchor], concise_preds[valid_anchor])
    log.info(f"  {'CoNCISE (same sub)':<20s} {auroc_c_sub:7.4f}                    {aupr_c_sub:7.4f}"
             f"  (same {valid_anchor.sum()} interactions)")

log.info(f"  {'-'*75}")

# Save
out = test_df.copy()
out["concise_pred"] = concise_preds
out["anchor_pred"] = anchor_preds
out.to_csv(RESULTS / "moodeng_test_results.csv", index=False)
log.info(f"\nSaved results/moodeng_test_results.csv")
log.info("Done!")
