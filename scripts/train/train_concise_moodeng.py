"""Train CoNCISE (no anchor) on MooDeng v1 for fair LCIdb comparison.

Same training data/splits as ConciseAnchor MooDeng, but without anchor conditioning.
Uses the original CoNCISE architecture with Raygun embeddings + Morgan FPs.
Binary BCE loss (same as ConciseAnchor).

Usage:
  python scripts/train/train_concise_moodeng.py
"""
import hashlib, logging, os, pickle, random, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS = Path("results"); RESULTS.mkdir(exist_ok=True)
SEED = 42; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def seq_to_id(seq):
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]

# ================================================================
# Load MooDeng v1
# ================================================================
log.info("Loading MooDengDB v1...")
MOODENG = Path("data/moodeng-v1")
train_raw = pd.read_csv(MOODENG / "train.csv", low_memory=False)
val_raw = pd.read_csv(MOODENG / "val.csv", low_memory=False)
test_raw = pd.read_csv(MOODENG / "test.csv", low_memory=False)
for df in [train_raw, val_raw, test_raw]:
    df.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
    df["prot_id"] = df.sequence.apply(seq_to_id)

all_seqs = {}
for df in [train_raw, val_raw, test_raw]:
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
    log.info("Computing Raygun embeddings...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(DEVICE)
    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(DEVICE)
    raygun_embs = {}
    for idx, (pid, seq) in enumerate(all_seqs.items()):
        seq = seq[:1022].upper()
        if len(seq) < 25: continue
        try:
            _, _, toks = bc([(pid, seq)])
            with torch.no_grad():
                out = esm_model(toks.to(DEVICE), repr_layers=[33], return_contacts=False)
                e = out["representations"][33][0, 1:len(seq)+1, :]
                r = raygun_model(e.unsqueeze(0))
                raygun_embs[pid] = r[1].squeeze(0).cpu()
            del out, e, r, toks
        except: pass
        if (idx+1) % 200 == 0:
            log.info(f"  {idx+1}/{len(all_seqs)}")
            torch.save(raygun_embs, str(RAYGUN_CACHE))
    del esm_model, raygun_model; torch.cuda.empty_cache()
    torch.save(raygun_embs, str(RAYGUN_CACHE))
    log.info(f"Saved {len(raygun_embs)} Raygun embeddings")
else:
    raygun_embs = torch.load(RAYGUN_CACHE, map_location="cpu", weights_only=False)
    log.info(f"Loaded {len(raygun_embs)} Raygun embeddings")

if FP_CACHE.exists():
    fp_dict = pickle.load(open(FP_CACHE, "rb"))
    log.info(f"Loaded {len(fp_dict)} Morgan FPs")
else:
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
    log.info(f"Computed {len(fp_dict)} Morgan FPs")

# Filter
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
# Dataset
# ================================================================
class ConciseBinaryDS(Dataset):
    """Drug FP + Raygun protein embedding → binary label."""
    def __init__(self, df, fp_dict, raygun_embs):
        fps, prot_pids, labels = [], [], []
        for _, r in df.iterrows():
            pid, smi, label = r.prot_id, r.smiles, float(r.label)
            if smi not in fp_dict or pid not in raygun_embs: continue
            fps.append(fp_dict[smi])
            prot_pids.append(pid)
            labels.append(label)
        self.fps = torch.tensor(np.array(fps))
        self.prot_pids = prot_pids
        self.labels = torch.tensor(labels, dtype=torch.float32)
        log.info(f"  ConciseBinaryDS: {len(self.labels)} ({(self.labels==1).sum().item()} pos)")

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        return self.fps[i], raygun_embs[self.prot_pids[i]], self.labels[i]

# ================================================================
# Model: CoNCISE from scratch with binary head
# ================================================================
from concise.model.concise import Concise

class ConciseBinary(nn.Module):
    """CoNCISE architecture trained from scratch → binary logit (BCEWithLogitsLoss)."""
    def __init__(self):
        super().__init__()
        # Build CoNCISE from scratch (random weights, NOT pretrained)
        self.backbone = Concise(
            drug_layers=[[32], [32], [32]],
            ligand_dim=2048,
            residue_dim=1280,  # Raygun dim
            drug_dim=256,
            proj_dim=256,
            nheads=16,
            activation="gelu",
            cosine_prediction=False,
        )
        # Replace final head with binary classifier
        n_drug_codes = 3
        fused_dim = n_drug_codes * 256 + 256
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.backbone.final[-1].bias, 0.0)

    def forward(self, drug_fp, prot_emb):
        out = self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)
        return out["binding"]  # raw logit

# ================================================================
# Train
# ================================================================
MODEL_DIR = Path("models/concise_moodeng"); MODEL_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = MODEL_DIR / "best_model.pt"

if BEST_PATH.exists():
    log.info(f"Model exists at {BEST_PATH}, skipping training")
else:
    log.info("Training CoNCISE (binary) on MooDeng v1...")
    model = ConciseBinary().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Params: {n_params:,}")

    train_ds = ConciseBinaryDS(train_df, fp_dict, raygun_embs)
    val_ds = ConciseBinaryDS(val_df, fp_dict, raygun_embs)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_auroc = 0.0
    patience, patience_count = 10, 0

    for epoch in range(30):
        model.train()
        losses = []
        for fps, embs, labels in train_loader:
            fps, embs, labels = fps.to(DEVICE), embs.to(DEVICE), labels.to(DEVICE)
            logits = model(fps, embs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for fps, embs, labels in val_loader:
                fps, embs = fps.to(DEVICE), embs.to(DEVICE)
                logits = model(fps, embs)
                val_preds.extend(torch.sigmoid(logits).cpu().tolist())
                val_labels.extend(labels.tolist())

        val_auroc = roc_auc_score(val_labels, val_preds)
        val_auprc = average_precision_score(val_labels, val_preds)
        log.info(f"  Epoch {epoch+1}: loss={np.mean(losses):.4f}, "
                 f"val_AUROC={val_auroc:.4f}, val_AUPRC={val_auprc:.4f}")

        if val_auroc > best_auroc:
            best_auroc = val_auroc
            patience_count = 0
            torch.save({"epoch": epoch+1, "model_state_dict": model.state_dict(),
                        "val_auroc": val_auroc}, str(BEST_PATH))
            log.info(f"    Saved best model (AUROC={val_auroc:.4f})")
        else:
            patience_count += 1
            if patience_count >= patience:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    del model; torch.cuda.empty_cache()

# ================================================================
# Evaluate on test
# ================================================================
log.info("\nEvaluating on MooDeng v1 test set...")
model = ConciseBinary().to(DEVICE)
ckpt = torch.load(str(BEST_PATH), map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
log.info(f"  Loaded epoch {ckpt['epoch']}, val_AUROC={ckpt['val_auroc']:.4f}")

test_preds, test_labels = [], []
with torch.no_grad():
    for i in range(0, len(test_df), 256):
        batch = test_df.iloc[i:i+256]
        fps = torch.tensor(np.array([fp_dict[s] for s in batch.smiles])).to(DEVICE)
        embs = torch.stack([raygun_embs[p] for p in batch.prot_id]).to(DEVICE)
        logits = model(fps, embs)
        test_preds.extend(torch.sigmoid(logits).cpu().tolist())
        test_labels.extend(batch.label.tolist())

auroc = roc_auc_score(test_labels, test_preds)
auprc = average_precision_score(test_labels, test_preds)
log.info(f"\n  CoNCISE (ours, MooDeng-trained): AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
log.info("Done.")
