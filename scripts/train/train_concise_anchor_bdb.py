"""Train ConciseAnchor-Bilinear on BindingDB for cross-dataset eval on Davis.

Uses Raygun embeddings + Morgan FPs.
Anchor = strongest binder per drug (pKi >= 7) from training set, excluding self.
5 epochs, saves best model by val loss.
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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).resolve().parents[2] / "data")))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Load data
# ============================================================
log.info("Loading BindingDB...")
bdb = pd.read_csv(DATA_DIR / "processed" / "bindingdb_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed" / "merged_sequences.json"))
log.info(f"BDB: {len(bdb)}, Seqs: {len(seqs)}")

# ============================================================
# 2. Load Raygun embeddings (precomputed by train_concise_bdb.py)
# ============================================================
RAYGUN_CACHE = Path("results/raygun_bdb_embeddings.pt")
if not RAYGUN_CACHE.exists():
    log.error(f"Raygun cache not found at {RAYGUN_CACHE}. Run train_concise_bdb.py first.")
    sys.exit(1)
raygun_embs = torch.load(RAYGUN_CACHE, map_location='cpu', weights_only=False)
log.info(f"Raygun embeddings: {len(raygun_embs)} proteins, dim={next(iter(raygun_embs.values())).shape}")

# ============================================================
# 3. Load Morgan fingerprints (precomputed by train_concise_bdb.py)
# ============================================================
FP_CACHE = Path("results/concise_bdb_morgan_fp.pkl")
if not FP_CACHE.exists():
    log.error(f"Morgan FP cache not found at {FP_CACHE}. Run train_concise_bdb.py first.")
    sys.exit(1)
with open(FP_CACHE, 'rb') as f:
    fp_dict = pickle.load(f)
log.info(f"Morgan FPs: {len(fp_dict)} drugs")

# ============================================================
# 4. BDB train/val split (same seed as concise_bdb)
# ============================================================
random.seed(42)
bdb_prots = sorted(set(bdb.uniprot_id) & set(raygun_embs.keys()) & set(seqs.keys()))
random.shuffle(bdb_prots)
nv = max(1, int(len(bdb_prots) * 0.1))
val_prots = set(bdb_prots[:nv])
train_prots = set(bdb_prots[nv:])

bdb_filt = bdb[bdb.uniprot_id.isin(raygun_embs.keys()) & bdb.ligand_smiles.isin(fp_dict)].copy()
train_df = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
val_df = bdb_filt[bdb_filt.uniprot_id.isin(val_prots)]
log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ============================================================
# 5. Build anchors: strongest binder per drug (pKi >= 7), excl self
# ============================================================
train_data = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
drug_to_anchors = {}
for smi, grp in train_data.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    candidates = [(u, p) for u, p in zip(s.uniprot_id.values, s.pki.values)
                   if p >= 7.0 and u in raygun_embs]
    if candidates:
        drug_to_anchors[smi] = candidates
log.info(f"Anchors: {len(drug_to_anchors)} drugs with pKi >= 7 binders")

del fp_dict
import gc; gc.collect()

# ============================================================
# 6. Dataset
# ============================================================
class ConciseAnchorDataset(Dataset):
    def __init__(self, df, fp_cache_path, raygun_embs, drug_to_anchors):
        with open(fp_cache_path, 'rb') as f:
            fp_dict = pickle.load(f)
        self.raygun_embs = raygun_embs
        fps = []
        self.anchor_uids = []
        self.query_uids = []
        pkis = []
        for _, row in df.iterrows():
            uid, smi, pki = row['uniprot_id'], row['ligand_smiles'], row['pki']
            if smi not in drug_to_anchors or smi not in fp_dict:
                continue
            if uid not in raygun_embs:
                continue
            candidates = drug_to_anchors[smi]
            anchor = None
            for au, ap in candidates:
                if au != uid:
                    anchor = au
                    break
            if anchor is None:
                continue
            fps.append(np.array(fp_dict[smi], dtype=np.float32))
            self.anchor_uids.append(anchor)
            self.query_uids.append(uid)
            pkis.append(pki)

        self.drug_fps = torch.tensor(np.array(fps))
        self.pkis = torch.tensor(pkis, dtype=torch.float32)
        log.info(f"  Dataset: {len(self.pkis)} interactions")

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, i):
        return (
            self.drug_fps[i],
            self.raygun_embs[self.anchor_uids[i]],
            self.raygun_embs[self.query_uids[i]],
            self.pkis[i],
        )

# ============================================================
# 7. Model
# ============================================================
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

log.info("Building model...")
model = ConciseAnchorBilinear(
    ligand_dim=2048,
    residue_dim=1280,
    proj_dim=256,
    n_codes=3,
    dropout=0.2,
).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"ConciseAnchor parameters: {n_params:,}")

train_ds = ConciseAnchorDataset(train_df, FP_CACHE, raygun_embs, drug_to_anchors)
val_ds = ConciseAnchorDataset(val_df, FP_CACHE, raygun_embs, drug_to_anchors)
BATCH_SIZE = 4096
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=4e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

best_val = float('inf')
os.makedirs("models/concise_anchor_bdb", exist_ok=True)

NUM_EPOCHS = 5
for ep in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, nb = 0, 0
    for drug, anchor, query, pki in tqdm(train_loader, desc=f"Ep {ep:3d}", leave=False):
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
            drug = drug.to(DEVICE)
            anchor = anchor.to(DEVICE)
            query = query.to(DEVICE)
            pred = model(drug, anchor, query)
            val_preds.extend(pred.cpu().tolist())
            val_true.extend(pki.tolist())

    val_true = np.array(val_true)
    val_preds = np.array(val_preds)
    val_loss = np.mean((val_true - val_preds) ** 2)
    val_r = np.corrcoef(val_true, val_preds)[0, 1] if len(val_true) > 1 else 0

    improved = val_loss < best_val
    if improved:
        best_val = val_loss
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                    'models/concise_anchor_bdb/best_model.pt')

    # Save every epoch
    torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                f'models/concise_anchor_bdb/epoch_{ep}.pt')

    log.info(f"Ep {ep:3d} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {'*BEST*' if improved else ''}")

log.info("Training complete")
