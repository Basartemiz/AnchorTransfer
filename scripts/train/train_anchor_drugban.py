"""Train AnchorDrugBAN on DTC, evaluate on Davis.

Anchor-based bilinear attention model: compares per-atom binding patterns
between anchor and query proteins using shared bilinear weight.
"""
import os, sys, json, logging, random, time, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch, Data
from sklearn.metrics import roc_auc_score, mean_squared_error
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()
sys.path.insert(0, 'src')

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================
# 1. Load data
# ============================================================
log.info("Loading datasets...")
dtc = pd.read_csv(DATA_DIR / "processed" / "dtc_training_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed" / "merged_sequences.json"))
log.info(f"DTC: {len(dtc)}, Seqs: {len(seqs)}")

# Load or build graph cache
from anchor_transfer.model.drug_encoder import smiles_to_graph

GRAPH_CACHE_PATH = DATA_DIR / "processed" / "drugban_graph_cache.pt"
if GRAPH_CACHE_PATH.exists():
    raw = torch.load(GRAPH_CACHE_PATH, map_location='cpu')
    graph_cache = {smi: Data(x=raw['x'][smi], edge_index=raw['edge_index'][smi]) for smi in raw['x']}
    log.info(f'Graph cache: {len(graph_cache)} drugs (loaded)')
else:
    log.info('Building graph cache from scratch...')
    all_smiles = sorted(dtc.ligand_smiles.unique())
    graph_cache = {}
    x_dict, edge_dict = {}, {}
    for i, smi in enumerate(all_smiles):
        try:
            g = smiles_to_graph(smi)
            if g is not None:
                graph_cache[smi] = g
                x_dict[smi] = g.x
                edge_dict[smi] = g.edge_index
        except Exception:
            pass
        if (i + 1) % 20000 == 0:
            log.info(f'  Graph cache: {i+1}/{len(all_smiles)} ({len(graph_cache)} ok)')
    os.makedirs(str(DATA_DIR / "processed"), exist_ok=True)
    torch.save({'x': x_dict, 'edge_index': edge_dict}, GRAPH_CACHE_PATH)
    log.info(f'Graph cache: {len(graph_cache)} drugs (built and saved)')

# ============================================================
# 2. DTC train/val split
# ============================================================
CHARPROTSET = {"A":1,"C":2,"B":3,"E":4,"D":5,"G":6,"F":7,"I":8,"H":9,"K":10,"M":11,"L":12,"O":13,"N":14,"Q":15,"P":16,"S":17,"R":18,"U":19,"T":20,"W":21,"V":22,"Y":23,"X":24,"Z":25}
def enc_prot(s, ml=1000): return [CHARPROTSET.get(c,0) for c in s[:ml]] + [0]*(ml-len(s[:ml]))

random.seed(42)
dtc_prots = sorted(set(dtc.uniprot_id) & set(seqs.keys()))
random.shuffle(dtc_prots)
nt = max(1, int(len(dtc_prots) * 0.1)); nv = max(1, int(len(dtc_prots) * 0.1))
train_prots = set(dtc_prots[nt + nv:])
val_prots = set(dtc_prots[nt:nt + nv])

dtc_filt = dtc[dtc.uniprot_id.isin(seqs.keys()) & dtc.ligand_smiles.isin(graph_cache)].copy()
train_df = dtc_filt[dtc_filt.uniprot_id.isin(train_prots)]
val_df = dtc_filt[dtc_filt.uniprot_id.isin(val_prots)]
log.info(f"Train: {len(train_df)}, Val: {len(val_df)}")

# ============================================================
# 3. Build anchors: strongest binder per drug (pKi >= 7)
# ============================================================
train_data = dtc_filt[dtc_filt.uniprot_id.isin(train_prots)]
drug_to_anchor = {}
drug_to_second = {}
for smi, grp in train_data.groupby('ligand_smiles'):
    s = grp.sort_values('pki', ascending=False)
    uids, pkis = s.uniprot_id.values, s.pki.values
    if pkis[0] >= 7.0 and uids[0] in seqs:
        drug_to_anchor[smi] = (uids[0], pkis[0])
        if len(uids) > 1 and uids[1] in seqs:
            drug_to_second[smi] = (uids[1], pkis[1])
log.info(f"Anchors: {len(drug_to_anchor)} drugs with pKi >= 7")

# ============================================================
# 4. Dataset
# ============================================================
class AnchorDrugBANDataset(Dataset):
    def __init__(self, df, seqs, graph_cache, drug_to_anchor, drug_to_second):
        self.seqs = seqs
        self.graph_cache = graph_cache
        self.rows = []
        for _, row in df.iterrows():
            uid, smi, pki = row['uniprot_id'], row['ligand_smiles'], row['pki']
            if smi not in drug_to_anchor or smi not in graph_cache:
                continue
            au, ap = drug_to_anchor[smi]
            # Self-check: if anchor is the query, use second strongest
            if au == uid:
                if smi not in drug_to_second:
                    continue
                au, ap = drug_to_second[smi]
            if au not in seqs:
                continue
            self.rows.append((uid, smi, pki, au))
        log.info(f"  Dataset: {len(self.rows)} interactions")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        uid, smi, pki, au = self.rows[i]
        return {
            'graph': self.graph_cache[smi].clone(),
            'anchor_prot': torch.tensor(enc_prot(self.seqs[au]), dtype=torch.long),
            'query_prot': torch.tensor(enc_prot(self.seqs[uid]), dtype=torch.long),
            'pki': pki,
        }

def collate_fn(batch):
    return {
        'graph': Batch.from_data_list([b['graph'] for b in batch]),
        'anchor_prot': torch.stack([b['anchor_prot'] for b in batch]),
        'query_prot': torch.stack([b['query_prot'] for b in batch]),
        'pki': torch.tensor([b['pki'] for b in batch], dtype=torch.float32),
    }

# ============================================================
# 5. Training
# ============================================================
from anchor_transfer.model.anchor_drugban import AnchorDrugBAN

model = AnchorDrugBAN(hidden_dim=128, dropout=0.2).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
log.info(f"AnchorDrugBAN parameters: {n_params:,}")

train_ds = AnchorDrugBANDataset(train_df, seqs, graph_cache, drug_to_anchor, drug_to_second)
val_ds = AnchorDrugBANDataset(val_df, seqs, graph_cache, drug_to_anchor, drug_to_second)
train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, collate_fn=collate_fn, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, collate_fn=collate_fn, num_workers=0)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind = np.argsort(y); y = y[ind]; f = f[ind]
    n = np.sum(np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1)))
    if n == 0: return np.nan
    z = np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T < np.tile(f, (len(f), 1)))) + 0.5 * np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T == np.tile(f, (len(f), 1))))
    return z / n

best_val = float('inf')
patience = 0
os.makedirs("models/anchor_drugban_dtc", exist_ok=True)

for ep in range(1, 101):
    t0 = time.time()
    model.train()
    total_loss, nb = 0, 0
    for batch in tqdm(train_loader, desc=f"Ep {ep:3d}", leave=False):
        graph = batch['graph'].to(DEVICE)
        anchor = batch['anchor_prot'].to(DEVICE)
        query = batch['query_prot'].to(DEVICE)
        pki = batch['pki'].to(DEVICE)

        pred = model(graph, anchor, query)
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
        for batch in val_loader:
            graph = batch['graph'].to(DEVICE)
            anchor = batch['anchor_prot'].to(DEVICE)
            query = batch['query_prot'].to(DEVICE)
            pred = model(graph, anchor, query)
            val_preds.extend(pred.cpu().tolist())
            val_true.extend(batch['pki'].tolist())

    vt, vp = np.array(val_true), np.array(val_preds)
    val_loss = np.mean((vt - vp) ** 2)
    if len(vt) > 5000:
        idx = np.random.choice(len(vt), 5000, replace=False)
        val_ci = ci_fn(vt[idx], vp[idx])
    else:
        val_ci = ci_fn(vt, vp)
    val_r = np.corrcoef(vt, vp)[0, 1] if len(vt) > 1 else 0

    improved = val_loss < best_val
    if improved:
        best_val = val_loss
        patience = 0
        torch.save({'model_state_dict': model.state_dict(), 'epoch': ep},
                    'models/anchor_drugban_dtc/best_model.pt')
    else:
        patience += 1

    log.info(f"Ep {ep:3d} [{time.time()-t0:.0f}s] Train={total_loss/nb:.4f} Val={val_loss:.4f} CI={val_ci:.4f} r={val_r:.4f} {'*' if improved else f'p={patience}'}")

    if patience >= 20:
        log.info("Early stopping")
        break

log.info("Training complete")
