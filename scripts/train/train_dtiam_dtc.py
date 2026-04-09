"""Train DTIAM on DTC, evaluate on Davis.

Uses BerMol for drug features + ESM-2 650M for protein features.
We already have ESM-2 embeddings; only BerMol needs computation.
AutoGluon tabular ensemble for prediction.
"""
import os, sys, json, pickle, time, logging
import numpy as np
import pandas as pd
import torch
import dill
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log = logging.getLogger()

DTIAM_DIR = Path(__file__).parent.parent / "DTIAM"
BERMOL_MODEL = DTIAM_DIR / "models" / "BerMolModel_base.pkl"
DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))

# ============================================================
# 1. Load DTC training data + Davis eval data
# ============================================================
log.info("Loading datasets...")
dtc = pd.read_csv(DATA_DIR / "processed" / "dtc_training_interactions.csv")
log.info(f"DTC: {len(dtc)} interactions, {dtc.uniprot_id.nunique()} proteins, {dtc.ligand_smiles.nunique()} drugs")

davis = pd.read_csv(DATA_DIR / "raw" / "davis" / "davis_benchmark.csv")
davis = davis.rename(columns={'protein_name': 'uniprot_id', 'drug_smiles': 'ligand_smiles', 'pKd': 'pki'})
log.info(f"Davis: {len(davis)} interactions, {davis.uniprot_id.nunique()} proteins, {davis.ligand_smiles.nunique()} drugs")

# Load sequences
seqs = json.load(open(DATA_DIR / "processed" / "merged_sequences.json"))
# Davis has its own sequences
d_seqs = dict(zip(davis.uniprot_id, davis.get('protein_sequence', [None]*len(davis))))
if 'protein_sequence' in pd.read_csv(DATA_DIR / "raw" / "davis" / "davis_benchmark.csv").columns:
    raw_davis = pd.read_csv(DATA_DIR / "raw" / "davis" / "davis_benchmark.csv")
    d_seqs = dict(zip(raw_davis.protein_name, raw_davis.protein_sequence))
    seqs.update(d_seqs)
log.info(f"Sequences: {len(seqs)}")

# Load ESM-2 650M embeddings (already precomputed)
esm = {}
for f in ['esm2_35m_dtc_proteins_full.pt', 'esm2_35m_all_proteins.pt', 'esm2_35m_benchmark.pt']:
    p = DATA_DIR / "processed" / f
    if p.exists():
        esm.update(torch.load(p, map_location='cpu'))
log.info(f"ESM-2 35M: {len(esm)} proteins")

# ============================================================
# 2. Extract BerMol compound features
# ============================================================
COMP_CACHE = Path("results/dtiam_compound_features.pkl")
if COMP_CACHE.exists():
    log.info(f"Loading cached compound features from {COMP_CACHE}")
    with open(COMP_CACHE, 'rb') as f:
        comp_feat = pickle.load(f)
else:
    log.info("Extracting BerMol compound features...")
    sys.path.insert(0, str(DTIAM_DIR / "code" / "BerMol"))
    import io
    class CPUUnpickler(dill.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
            return super().find_class(module, name)
    with open(BERMOL_MODEL, 'rb') as f:
        comp_model = CPUUnpickler(f).load()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    comp_model.model.to(device)
    comp_model.model.eval()

    # Get all unique SMILES from both DTC and Davis
    all_smiles = set(dtc.ligand_smiles.unique()) | set(davis.ligand_smiles.unique())
    log.info(f"Computing features for {len(all_smiles)} unique SMILES...")

    comp_feat = {}
    for i, smi in enumerate(all_smiles):
        try:
            output = comp_model.transform(smi, device)
            comp_feat[smi] = output[1].cpu().detach().numpy().reshape(-1)
        except Exception as e:
            pass  # Skip failed molecules
        if (i + 1) % 10000 == 0:
            log.info(f"  {i+1}/{len(all_smiles)}")

    log.info(f"Computed {len(comp_feat)} compound features")
    os.makedirs("results", exist_ok=True)
    with open(COMP_CACHE, 'wb') as f:
        pickle.dump(comp_feat, f)

# Check feature dimensions
sample_comp = next(iter(comp_feat.values()))
sample_prot = next(iter(esm.values())).numpy()
log.info(f"Compound feature dim: {sample_comp.shape[0]}, Protein feature dim: {sample_prot.shape[0]}")

# ============================================================
# 3. DTC train/val split (same as all other models)
# ============================================================
import random
random.seed(42)

dtc_valid_prots = sorted(set(dtc.uniprot_id) & set(esm.keys()))
random.shuffle(dtc_valid_prots)
nt = max(1, int(len(dtc_valid_prots) * 0.1))
nv = max(1, int(len(dtc_valid_prots) * 0.1))
test_prots = set(dtc_valid_prots[:nt])
val_prots = set(dtc_valid_prots[nt:nt+nv])
train_prots = set(dtc_valid_prots[nt+nv:])

# Filter to proteins with ESM embeddings and drugs with BerMol features
dtc_filt = dtc[dtc.uniprot_id.isin(esm.keys()) & dtc.ligand_smiles.isin(comp_feat)].copy()
train_df = dtc_filt[dtc_filt.uniprot_id.isin(train_prots)]
val_df = dtc_filt[dtc_filt.uniprot_id.isin(val_prots)]
log.info(f"DTC train: {len(train_df)}, val: {len(val_df)}")

# ============================================================
# 4. Pack features into tabular format
# ============================================================
def pack_features(df, comp_feat, esm_feat):
    """Concatenate compound + protein features into tabular format."""
    rows = []
    labels = []
    skipped = 0
    for _, row in df.iterrows():
        smi = row['ligand_smiles']
        uid = row['uniprot_id']
        if smi not in comp_feat or uid not in esm_feat:
            skipped += 1
            continue
        cfeat = comp_feat[smi]
        pfeat = esm_feat[uid].numpy() if isinstance(esm_feat[uid], torch.Tensor) else esm_feat[uid]
        rows.append(np.concatenate([cfeat, pfeat]))
        labels.append(row['pki'])
    if skipped:
        log.info(f"  Skipped {skipped} interactions (missing features)")
    feat_df = pd.DataFrame(rows)
    feat_df['y'] = labels
    return feat_df

log.info("Packing DTC training features...")
train_tab = pack_features(train_df, comp_feat, esm)
log.info(f"Train: {len(train_tab)} rows, {train_tab.shape[1]-1} features")

log.info("Packing Davis eval features...")
davis_tab = pack_features(davis, comp_feat, esm)
log.info(f"Davis: {len(davis_tab)} rows")

# ============================================================
# 5. Train AutoGluon on DTC
# ============================================================
from autogluon.tabular import TabularPredictor

log.info("Training AutoGluon...")
predictor = TabularPredictor(
    label='y',
    path='results/dtiam_dtc_model',
).fit(
    train_data=train_tab,
    time_limit=3600,  # 1 hour max
    presets='best_quality',
)

# ============================================================
# 6. Predict on Davis
# ============================================================
log.info("Predicting on Davis...")
davis_nolab = davis_tab.drop(columns=['y'])
davis_pred = predictor.predict(davis_nolab)
davis_tab['pred'] = davis_pred.values

# ============================================================
# 7. Compute metrics
# ============================================================
from sklearn.metrics import roc_auc_score, mean_squared_error

def ci_fn(y, f):
    if len(y) < 2: return np.nan
    ind = np.argsort(y); y = y[ind]; f = f[ind]
    n = np.sum(np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1)))
    if n == 0: return np.nan
    z = np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T < np.tile(f, (len(f), 1)))) + 0.5 * np.sum((np.tile(y, (len(y), 1)).T < np.tile(y, (len(y), 1))) * (np.tile(f, (len(f), 1)).T == np.tile(f, (len(f), 1))))
    return z / n

y = davis_tab['y'].values
p = davis_tab['pred'].values

overall_ci = ci_fn(y, p)
overall_rmse = np.sqrt(mean_squared_error(y, p))
bin_labels = (y >= 7).astype(int)
if bin_labels.sum() > 0 and (1 - bin_labels).sum() > 0:
    overall_auroc = roc_auc_score(bin_labels, p)
else:
    overall_auroc = np.nan

log.info(f"\n=== DTIAM (DTC-trained) on Davis ===")
log.info(f"Overall: CI={overall_ci:.4f}, RMSE={overall_rmse:.4f}, AUROC={overall_auroc:.4f}")
log.info(f"N={len(y)} interactions")

# Save predictions
davis_tab.to_csv('results/dtiam_davis_predictions.csv', index=False)
log.info("Saved predictions to results/dtiam_davis_predictions.csv")
log.info("DONE")
