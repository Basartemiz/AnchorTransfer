"""Train CoNCISE with CORRECT original defaults on BDB.

Fixes from incorrect wrapper:
- drug_dim=128 (was 256)
- activation=tanh (was gelu)
- Simpler regression head (2-layer instead of 3-layer with dropout)
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
from concise.model.concise import Concise

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("data")

bdb = pd.read_csv(DATA_DIR / "processed/bindingdb_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed/merged_sequences.json"))
raygun_embs = torch.load("data/processed/raygun_bdb_embeddings.pt", map_location="cpu", weights_only=False)
fp_dict = pickle.load(open("data/processed/concise_bdb_morgan_fp.pkl", "rb"))
log.info(f"BDB: {len(bdb)}, Raygun: {len(raygun_embs)}, FPs: {len(fp_dict)}")

# Split
random.seed(42)
bdb_prots = sorted(set(bdb.uniprot_id) & set(raygun_embs.keys()) & set(seqs.keys()))
random.shuffle(bdb_prots)
nv = max(1, int(len(bdb_prots) * 0.1))
val_prots = set(bdb_prots[:nv])
train_prots = set(bdb_prots[nv:])
bdb_filt = bdb[bdb.uniprot_id.isin(raygun_embs.keys()) & bdb.ligand_smiles.isin(fp_dict)].copy()

# Apple-to-apple: filter to same drug subset as ConciseAnchor (drugs with pKi>=7 anchors)
train_subset = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
drug_to_anchors = {}
for smi, grp in train_subset.groupby("ligand_smiles"):
    s = grp.sort_values("pki", ascending=False)
    candidates = [(u, p) for u, p in zip(s.uniprot_id.values, s.pki.values)
                   if p >= 7.0 and u in raygun_embs]
    if candidates:
        drug_to_anchors[smi] = candidates
anchor_drugs = set(drug_to_anchors.keys())
bdb_filt = bdb_filt[bdb_filt.ligand_smiles.isin(anchor_drugs)]
log.info(f"Anchor-filtered to {len(bdb_filt)} interactions ({len(anchor_drugs)} drugs with pKi>=7 binders)")

train_df = bdb_filt[bdb_filt.uniprot_id.isin(train_prots)]
val_df = bdb_filt[bdb_filt.uniprot_id.isin(val_prots)]
log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")


class DS(Dataset):
    def __init__(self, df):
        fps, uids, pkis = [], [], []
        for _, r in df.iterrows():
            if r.uniprot_id in raygun_embs and r.ligand_smiles in fp_dict:
                fps.append(np.array(fp_dict[r.ligand_smiles], dtype=np.float32))
                uids.append(r.uniprot_id)
                pkis.append(r.pki)
        self.fps = torch.tensor(np.array(fps))
        self.uids = uids
        self.pkis = torch.tensor(pkis, dtype=torch.float32)
        log.info(f"  DS: {len(self.pkis)}")

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, i):
        return self.fps[i], raygun_embs[self.uids[i]], self.pkis[i]


class ConciseFixed(nn.Module):
    """CoNCISE with ORIGINAL defaults: drug_dim=128, tanh activation."""

    def __init__(self):
        super().__init__()
        drug_layers = [[32], [32], [32]]
        self.backbone = Concise(
            drug_layers=drug_layers,
            ligand_dim=2048,
            residue_dim=1280,
            drug_dim=128,
            proj_dim=256,
            nheads=32,
            activation="tanh",
            cosine_prediction=False,
        )
        # Replace sigmoid head with regression (simpler than before)
        fused_dim = len(drug_layers) * 256 + 256
        self.backbone.final = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.constant_(self.backbone.final[-1].bias, 6.5)

    def forward(self, drug_fp, prot_emb):
        return self.backbone(drug_fp, prot_emb, is_morgan_fingerprint=True)["binding"]


model = ConciseFixed().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"ConciseFixed params: {n_params:,}")

train_ds = DS(train_df)
val_ds = DS(val_df)
train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=4, pin_memory=False)
val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False, num_workers=4, pin_memory=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

best_val = float("inf")
os.makedirs("models/concise_bdb_fixed", exist_ok=True)

for ep in range(1, 6):
    t0 = time.time()
    model.train()
    total_loss, nb = 0, 0
    for drug, prot, pki in tqdm(train_loader, desc=f"Ep {ep}", leave=False):
        drug, prot, pki = drug.to(DEVICE), prot.to(DEVICE), pki.to(DEVICE)
        pred = model(drug, prot)
        loss = F.mse_loss(pred, pki)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(pki)
        nb += len(pki)
    scheduler.step()

    model.eval()
    vp, vt = [], []
    with torch.no_grad():
        for d, p, pk in val_loader:
            d, p = d.to(DEVICE), p.to(DEVICE)
            vp.extend(model(d, p).cpu().tolist())
            vt.extend(pk.tolist())
    vt, vp = np.array(vt), np.array(vp)
    val_loss = np.mean((vt - vp) ** 2)
    val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0

    tag = "*BEST*" if val_loss < best_val else ""
    if val_loss < best_val:
        best_val = val_loss
        torch.save({"model_state_dict": model.state_dict(), "epoch": ep},
                    "models/concise_bdb_fixed/best_model.pt")
    torch.save({"model_state_dict": model.state_dict(), "epoch": ep},
                f"models/concise_bdb_fixed/epoch_{ep}.pt")
    log.info(f"Ep {ep} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} r={val_r:.4f} {tag}")

# Quick drug-variance test
log.info("Testing drug variance...")
model.eval()
prot_uid = list(raygun_embs.keys())[0]
emb = raygun_embs[prot_uid].unsqueeze(0).to(DEVICE)
drugs = list(fp_dict.keys())[:10]
with torch.no_grad():
    preds = []
    for d in drugs:
        fp = torch.tensor(np.array(fp_dict[d], dtype=np.float32)).unsqueeze(0).to(DEVICE)
        preds.append(model(fp, emb).item())
log.info(f"Prot {prot_uid}: preds={[round(p, 4) for p in preds]}")
log.info(f"std={np.std(preds):.8f}")
log.info("Done")
