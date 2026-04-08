"""Evaluate ConciseAnchor-Bilinear (BDB-trained) on Davis.

Standalone eval — does NOT require concise-dti package.
Only evaluates ConciseAnchor (our model), not the CoNCISE baseline.
"""
import os, sys, json, logging, random, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, mean_squared_error
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))


def ci_fn(y, f):
    n = len(y)
    if n < 2:
        return 0.5
    y, f = np.array(y), np.array(f)
    if n * (n - 1) // 2 > 100000:
        i = np.random.randint(0, n, 100000)
        j = np.random.randint(0, n, 100000)
        m = i != j
        i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1)
        i, j = idx[0], idx[1]
    dt = y[i] - y[j]
    dp = f[i] - f[j]
    t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5


def auroc_safe(trues, preds):
    binder = trues >= 7.0
    non_binder = trues <= 5.0
    mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))


# 1. Load Davis
davis_raw = pd.read_csv(DATA_DIR / "raw" / "davis" / "davis_benchmark.csv")
davis = davis_raw.rename(columns={"protein_name": "uniprot_id", "drug_smiles": "ligand_smiles", "pKd": "pki"})
seqs = {}
if "protein_sequence" in davis_raw.columns:
    for _, r in davis_raw.drop_duplicates("protein_name").iterrows():
        seqs[r["protein_name"]] = r["protein_sequence"]
merged_seq = DATA_DIR / "processed" / "merged_sequences.json"
if merged_seq.exists():
    seqs.update(json.load(open(merged_seq)))
log.info("Davis: %d interactions, %d proteins, %d drugs", len(davis), davis.uniprot_id.nunique(), davis.ligand_smiles.nunique())

# 2. Load BDB training data
bdb = pd.read_csv(DATA_DIR / "processed" / "bindingdb_interactions.csv")
log.info("BDB: %d interactions", len(bdb))

# 3. Exclude overlapping drugs (canonical SMILES)
from rdkit import Chem


def canon(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else smi
    except Exception:
        return smi


bdb_canon = set(canon(s) for s in bdb.ligand_smiles.unique())
davis["canon_smi"] = davis.ligand_smiles.apply(canon)
overlap = set(davis.canon_smi) & bdb_canon
davis_clean = davis[~davis.canon_smi.isin(overlap)].copy()
log.info("Davis after drug exclusion: %d (%d drugs excluded)", len(davis_clean), len(overlap))

# 4. Tanimoto anchor retrieval from BDB
from rdkit.Chem import AllChem, DataStructs

bdb_strong = bdb[bdb.pki >= 7.0].copy()
bdb_anchor_pool = bdb_strong.loc[bdb_strong.groupby("ligand_smiles")["pki"].idxmax()].copy()
log.info("BDB anchor pool: %d drugs with pKi >= 7", len(bdb_anchor_pool))


def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=True)


bdb_fps = {}
for _, row in bdb_anchor_pool.iterrows():
    fp = morgan_fp(row.ligand_smiles)
    if fp is not None:
        bdb_fps[row.ligand_smiles] = (fp, row.uniprot_id, row.pki)
log.info("BDB FPs: %d", len(bdb_fps))

davis_drugs = sorted(davis_clean.ligand_smiles.unique())
drug_to_anchor = {}
for smi in davis_drugs:
    fp = morgan_fp(smi)
    if fp is None:
        continue
    best_sim, best_anc, best_pki = -1, None, 0
    for bdb_smi, (bdb_fp, anc_uid, anc_pki) in bdb_fps.items():
        sim = DataStructs.TanimotoSimilarity(fp, bdb_fp)
        if sim > best_sim:
            best_sim, best_anc, best_pki = sim, anc_uid, anc_pki
    if best_anc:
        drug_to_anchor[smi] = (best_anc, best_pki, best_sim)
log.info("Anchored %d/%d Davis drugs", len(drug_to_anchor), len(davis_drugs))

subset = davis_clean[davis_clean.ligand_smiles.isin(drug_to_anchor)].copy()
subset["anchor_uid"] = subset.ligand_smiles.map(lambda s: drug_to_anchor[s][0])
subset["anchor_pki"] = subset.ligand_smiles.map(lambda s: drug_to_anchor[s][1])
subset["tanimoto"] = subset.ligand_smiles.map(lambda s: drug_to_anchor[s][2])

# Quartiles
edges = subset.anchor_pki.quantile([0, 0.25, 0.5, 0.75, 1.0]).values
qlabels = ["Q1 (weakest)", "Q2", "Q3", "Q4 (strongest)"]
subset["anchor_q"] = pd.cut(subset.anchor_pki, bins=edges, labels=qlabels, include_lowest=True)
log.info("Anchored subset: %d interactions, tanimoto mean=%.3f", len(subset), subset.tanimoto.mean())

# 5. Load Raygun embeddings + Morgan FPs
raygun_embs = torch.load("results/raygun_bdb_embeddings.pt", map_location="cpu", weights_only=False)
log.info("Raygun: %d proteins", len(raygun_embs))

# Compute Raygun for Davis proteins not in BDB cache
davis_prots_needed = set(subset.uniprot_id.unique()) | set(subset.anchor_uid.unique())
missing = davis_prots_needed - set(raygun_embs.keys())
if missing:
    log.info("Computing Raygun for %d Davis proteins...", len(missing))
    import esm
    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(device)
    esm_model.eval()
    davis_esm = {}
    for uid in sorted(missing):
        if uid not in seqs:
            continue
        seq = seqs[uid][:1022]
        _, _, tokens = bc([(uid, seq)])
        with torch.no_grad():
            emb = esm_model(tokens.to(device), repr_layers=[33], return_contacts=False)
            davis_esm[uid] = emb["representations"][33][:, 1:-1, :].cpu()
    del esm_model
    torch.cuda.empty_cache()
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(device)
    raymodel.eval()
    with torch.no_grad():
        for uid, emb in davis_esm.items():
            try:
                r = raymodel.encoder(emb.to(device)).squeeze().cpu()
                if r.dim() == 2 and r.size(0) == 50:
                    raygun_embs[uid] = r
            except Exception:
                pass
    del raymodel
    torch.cuda.empty_cache()
    log.info("Total Raygun: %d", len(raygun_embs))

from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path("results/concise_davis_fp.pkl")
if FP_CACHE.exists():
    with open(FP_CACHE, "rb") as f:
        fp_dict = pickle.load(f)
else:
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for smi in sorted(set(subset.ligand_smiles.unique())):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except Exception:
            pass
    with open(FP_CACHE, "wb") as f:
        pickle.dump(fp_dict, f)
log.info("Morgan FPs: %d drugs", len(fp_dict))

# 6. Load ConciseAnchor and predict
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

anchor_model = ConciseAnchorBilinear(
    ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2
).to(device)
ckpt = torch.load("models/concise_anchor_bdb/best_model.pt", map_location=device, weights_only=False)
anchor_model.load_state_dict(ckpt["model_state_dict"])
anchor_model.eval()
log.info("Loaded ConciseAnchor-BDB (epoch %s)", ckpt.get("epoch", "?"))

preds, valid_mask = [], []
for _, row in subset.iterrows():
    uid, smi, au = row["uniprot_id"], row["ligand_smiles"], row["anchor_uid"]
    if uid not in raygun_embs or smi not in fp_dict or au not in raygun_embs:
        preds.append(np.nan)
        valid_mask.append(False)
        continue
    fp = torch.tensor(fp_dict[smi]).unsqueeze(0).to(device)
    qry = raygun_embs[uid].unsqueeze(0).to(device)
    anc = raygun_embs[au].unsqueeze(0).to(device)
    with torch.no_grad():
        preds.append(anchor_model(fp, anc, qry).item())
    valid_mask.append(True)

subset["pred"] = preds
sv = subset[valid_mask].copy()
log.info("Valid predictions: %d", len(sv))

# 7. Results
t, p = sv.pki.values, sv.pred.values
ci = ci_fn(t, p)
rmse = np.sqrt(mean_squared_error(t, p))
auroc = auroc_safe(t, p)
r = float(np.corrcoef(t, p)[0, 1]) if len(t) > 1 else 0
log.info("=" * 60)
log.info("ConciseAnchor-BDB -> Davis: CI=%.4f AUROC=%.4f RMSE=%.4f r=%.4f n=%d", ci, auroc, rmse, r, len(t))
for q in qlabels:
    sub = sv[sv.anchor_q == q]
    if len(sub) < 5:
        continue
    tv, pv = sub.pki.values, sub.pred.values
    log.info("  %-16s n=%-6d CI=%.4f AUROC=%.4f RMSE=%.4f", q, len(sub), ci_fn(tv, pv), auroc_safe(tv, pv), np.sqrt(np.mean((tv - pv) ** 2)))

os.makedirs("results", exist_ok=True)
sv.to_csv("results/concise_anchor_bdb_davis.csv", index=False)
log.info("Saved to results/concise_anchor_bdb_davis.csv")
