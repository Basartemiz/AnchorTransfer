"""Train CoNCISE on BindingDB for cross-dataset eval on Davis.

Uses ESM-2 650M → Raygun → CoNCISE (regression).
Precomputes Raygun embeddings for all BDB proteins.
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
# 2. Compute Raygun protein embeddings (ESM-2 650M → Raygun)
# ============================================================
ESM_CACHE = Path("results/esm2_bdb_embeddings.pt")
RAYGUN_CACHE = Path("results/raygun_bdb_embeddings.pt")
os.makedirs("results", exist_ok=True)

if RAYGUN_CACHE.exists():
    log.info(f"Loading cached Raygun embeddings from {RAYGUN_CACHE}")
    raygun_embs = torch.load(RAYGUN_CACHE, map_location='cpu', weights_only=False)
else:
    # Step 1: ESM-2 embeddings (cached separately)
    if ESM_CACHE.exists():
        log.info(f"Loading cached ESM-2 embeddings from {ESM_CACHE}")
        esm_embeddings = torch.load(ESM_CACHE, map_location='cpu', weights_only=False)
    else:
        log.info("Computing ESM-2 650M embeddings...")
        import esm
        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = esm_alphabet.get_batch_converter()
        esm_model = esm_model.to(DEVICE)
        esm_model.eval()

        all_prots = sorted(set(bdb.uniprot_id) & set(seqs.keys()))
        log.info(f"Computing ESM-2 650M for {len(all_prots)} proteins...")

        esm_embeddings = {}
        with torch.no_grad():
            for i, uid in enumerate(all_prots):
                seq = seqs[uid][:1022]
                _, _, tokens = bc([(uid, seq)])
                emb = esm_model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
                esm_embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
                if (i + 1) % 100 == 0:
                    log.info(f"  ESM-2: {i+1}/{len(all_prots)}")

        del esm_model
        torch.cuda.empty_cache()
        torch.save(esm_embeddings, ESM_CACHE)
        log.info(f"Cached {len(esm_embeddings)} ESM-2 embeddings to {ESM_CACHE}")

    # Step 2: Raygun encoding
    log.info("Running Raygun encoder...")
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(DEVICE)
    raymodel.eval()

    raygun_embs = {}
    skipped = 0
    with torch.no_grad():
        for i, (uid, emb) in enumerate(esm_embeddings.items()):
            try:
                ray_enc = raymodel.encoder(emb.to(DEVICE)).squeeze().cpu()
                if ray_enc.dim() == 2 and ray_enc.size(0) == 50:
                    raygun_embs[uid] = ray_enc
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log.warning(f"  Skipped {uid} (seq len {emb.size(1)}): {e}")
            if (i + 1) % 100 == 0:
                log.info(f"  Raygun: {i+1}/{len(esm_embeddings)} (skipped {skipped})")

    del raymodel, esm_embeddings
    torch.cuda.empty_cache()
    torch.save(raygun_embs, RAYGUN_CACHE)
    log.info(f"Saved {len(raygun_embs)} Raygun embeddings to {RAYGUN_CACHE}")

log.info(f"Raygun embeddings: {len(raygun_embs)} proteins, dim={next(iter(raygun_embs.values())).shape}")

# ============================================================
# 3. Compute Morgan fingerprints
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path("results/concise_bdb_morgan_fp.pkl")
if FP_CACHE.exists():
    log.info(f"Loading cached fingerprints from {FP_CACHE}")
    with open(FP_CACHE, 'rb') as f:
        fp_dict = pickle.load(f)
else:
    log.info("Computing Morgan fingerprints...")
    all_smiles = sorted(set(bdb.ligand_smiles.unique()))
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for i, smi in enumerate(all_smiles):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except:
            pass
        if (i + 1) % 20000 == 0:
            log.info(f"  FP: {i+1}/{len(all_smiles)}")
    log.info(f"Computed {len(fp_dict)} fingerprints")
    os.makedirs("results", exist_ok=True)
    with open(FP_CACHE, 'wb') as f:
        pickle.dump(fp_dict, f)
log.info(f"FP dict: {len(fp_dict)} entries")

# ============================================================
# 4. BDB train/val split (protein-level)
# ============================================================
random.seed(42)
bdb_prots = sorted(set(bdb.uniprot_id) & set(raygun_embs.keys()) & set(seqs.keys()))
random.shuffle(bdb_prots)
nv = max(1, int(len(bdb_prots) * 0.1))
val_prots = set(bdb_prots[:nv])
train_prots = set(bdb_prots[nv:])

bdb_filt = bdb[bdb.uniprot_id.isin(raygun_embs.keys()) & bdb.ligand_smiles.isin(fp_dict)].copy()

# Apple-to-apple: filter to same samples ConciseAnchor uses (drugs with pKi>=7 anchor)
# This ensures CoNCISE baseline and ConciseAnchor train on identical data subsets.
ANCHOR_FILTER = os.environ.get("ANCHOR_FILTER", "1") == "1"
if ANCHOR_FILTER:
    train_subset = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
    drug_to_anchors = {}
    for smi, grp in train_subset.groupby('ligand_smiles'):
        s = grp.sort_values('pki', ascending=False)
        candidates = [(u, p) for u, p in zip(s.uniprot_id.values, s.pki.values)
                       if p >= 7.0 and u in raygun_embs]
        if candidates:
            drug_to_anchors[smi] = candidates
    anchor_drugs = set(drug_to_anchors.keys())
    bdb_filt = bdb_filt[bdb_filt.ligand_smiles.isin(anchor_drugs)]
    log.info(f"Anchor filter: kept {len(anchor_drugs)} drugs with pKi>=7 binders")

train_df = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
val_df = bdb_filt[bdb_filt.uniprot_id.isin(val_prots)]
log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

del fp_dict
import gc; gc.collect()

# ============================================================
# 5. Dataset
# ============================================================
class ConciseDTADataset(Dataset):
    def __init__(self, df, fp_cache_path, raygun_embs):
        with open(fp_cache_path, 'rb') as f:
            fp_dict = pickle.load(f)
        self.uids = df.uniprot_id.values
        self.pkis = torch.tensor(df.pki.values, dtype=torch.float32)
        self.drug_fps = torch.tensor(
            np.array([np.array(fp_dict[s], dtype=np.float32) for s in df.ligand_smiles.values])
        )
        self.raygun_embs = raygun_embs
        log.info(f"  Dataset: {len(self.pkis)} interactions")

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, i):
        return self.drug_fps[i], self.raygun_embs[self.uids[i]], self.pkis[i]

# ============================================================
# 6. Model
# ============================================================
from concise.model.concise import Concise

class ConciseRegression(nn.Module):
    def __init__(self, nheads=32):
        super().__init__()
        drug_layers = [[32], [32], [32]]
        proj_dim = 256
        self.backbone = Concise(
            drug_layers=drug_layers, ligand_dim=2048, residue_dim=1280,
            drug_dim=proj_dim, proj_dim=proj_dim, nheads=nheads,
            activation="gelu", cosine_prediction=False,
        )
        fused_dim = len(drug_layers) * proj_dim + proj_dim
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )
        nn.init.constant_(self.backbone.final[-1].bias, 6.5)
    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]

# ============================================================
# 7. Training (5 epochs)
# ============================================================
log.info("Building model...")
model = ConciseRegression(nheads=32).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"Model parameters: {n_params:,}")

train_ds = ConciseDTADataset(train_df, FP_CACHE, raygun_embs)
val_ds = ConciseDTADataset(val_df, FP_CACHE, raygun_embs)
BATCH_SIZE = 8192
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

best_val = float('inf')
os.makedirs("models/concise_bdb", exist_ok=True)

NUM_EPOCHS = 5
for ep in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, nb = 0, 0
    for drug, prot, pki in tqdm(train_loader, desc=f"Ep {ep:3d}", leave=False):
        drug = drug.to(DEVICE, non_blocking=True)
        prot = prot.to(DEVICE, non_blocking=True)
        pki = pki.to(DEVICE, non_blocking=True)
        pred = model(drug, prot)
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
        for drug, prot, pki in val_loader:
            drug, prot = drug.to(DEVICE), prot.to(DEVICE)
            pred = model(drug, prot)
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
                    'models/concise_bdb/best_model.pt')

    # Save every epoch
    torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                f'models/concise_bdb/epoch_{ep}.pt')

    log.info(f"Ep {ep:3d} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {'*BEST*' if improved else ''}")

log.info("Training complete")
