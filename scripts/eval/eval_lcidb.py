#!/usr/bin/env python3
"""Cross-dataset evaluation on LCIdb benchmark.

Evaluates anchor transfer models and baselines on LCIdb — a large-scale
drug-target interaction dataset from ChEMBL/PubChem/BindingDB (via Consensus DB).

Protocol:
  1. Download and load LCIdb_v2.csv from Zenodo (record 12178118).
  2. Remove ALL overlap with DTC training data:
     - Protein overlap by UniProt sequence identity
     - Drug overlap by canonical SMILES identity
     - Interaction overlap (same drug+protein pair)
  3. Filter to proteins with ESM-2 embeddings (extract if needed).
  4. Retrieve Tanimoto anchors from DTC training pool (paper protocol).
  5. Evaluate V2-650M, V2-35M, DeepDTA, ESM-DTA on the same anchored subset.
  6. Report CI, AUROC, AUPRC, RMSE, Pearson r with quartile breakdown.

Usage:
  python scripts/eval/eval_lcidb.py [--device cuda] [--skip-esm2]
"""
import json, logging, os, random, sys, hashlib
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()
sys.path.insert(0, "src")

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT = Path(__file__).resolve().parents[2]

# ── Character encodings (DeepDTA protocol) ──
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
    "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "[": 50,
    "T": 17, "]": 51, "V": 18, "Y": 19, "c": 20, "e": 21, "l": 22,
    "n": 23, "o": 24, "r": 25, "s": 26, "t": 27, "u": 28,
}
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}
def encode_smi(smi, ml=100):
    return [CHARISOSMISET.get(c, 0) for c in smi[:ml]] + [0] * max(0, ml - len(smi))
def encode_prot(seq, ml=1000):
    return [CHARPROTSET.get(c, 0) for c in seq[:ml]] + [0] * max(0, ml - len(seq))


# ── Metrics ──
def ci_fn(yt, yp):
    n = len(yt)
    if n < 2: return 0.5
    yt, yp = np.array(yt), np.array(yp)
    if n*(n-1)//2 > 100000:
        i = np.random.randint(0, n, 100000); j = np.random.randint(0, n, 100000)
        m = i != j; i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1); i, j = idx
    dt = yt[i]-yt[j]; dp = yp[i]-yp[j]; t = dt == 0
    return float(((dt*dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

def auroc_safe(t, p):
    """AUROC with paper protocol: >=7 positive, <=5 negative, ambiguous excluded."""
    b = t >= 7; nb = t <= 5; m = b | nb
    if m.sum() == 0 or b[m].sum() == 0 or nb[m].sum() == 0: return float("nan")
    return float(roc_auc_score(b[m].astype(int), p[m]))

def auprc_safe(t, p):
    b = t >= 7; nb = t <= 5; m = b | nb
    if m.sum() == 0 or b[m].sum() == 0 or nb[m].sum() == 0: return float("nan")
    return float(average_precision_score(b[m].astype(int), p[m]))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-esm2", action="store_true",
                        help="Skip ESM-2 extraction (use precomputed)")
    parser.add_argument("--lcidb-path", default=None,
                        help="Path to LCIdb_v2.csv (auto-downloads if missing)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    global device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # ================================================================
    # Step 1: Download / Load LCIdb
    # ================================================================
    log.info("=" * 60)
    log.info("Step 1: Loading LCIdb")
    log.info("=" * 60)

    lcidb_path = Path(args.lcidb_path) if args.lcidb_path else PROJECT / "data" / "LCIdb_v2.csv"
    if not lcidb_path.exists():
        log.info(f"Downloading LCIdb_v2.csv to {lcidb_path}...")
        lcidb_path.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run([
            "curl", "-L", "-o", str(lcidb_path),
            "https://zenodo.org/api/records/12178118/files/LCIdb_v2.csv/content"
        ], check=True)

    lci = pd.read_csv(lcidb_path, low_memory=False)
    log.info(f"LCIdb raw: {len(lci)} interactions, {lci['fasta'].nunique()} proteins, "
             f"{lci['smiles'].nunique()} drugs")

    # Use mean pKi where available, fall back to mean pIC50, then score
    lci["pki"] = lci["mean pKi"]
    mask_no_pki = lci["pki"].isna()
    lci.loc[mask_no_pki, "pki"] = lci.loc[mask_no_pki, "mean pIC50"]
    mask_still_na = lci["pki"].isna()
    lci.loc[mask_still_na, "pki"] = lci.loc[mask_still_na, "score"]
    lci = lci.dropna(subset=["pki"])
    lci = lci[lci["pki"] > 0]  # remove zero/negative
    log.info(f"LCIdb with valid pKi: {len(lci)} interactions")

    # Rename columns for consistency
    lci = lci.rename(columns={"smiles": "ligand_smiles", "fasta": "protein_sequence"})

    # Create protein IDs from sequence hash (LCIdb uses UniProt but sequences are more reliable)
    def seq_to_id(seq):
        return "L" + hashlib.md5(str(seq).encode()).hexdigest()[:12]
    lci["protein_id"] = lci["protein_sequence"].apply(seq_to_id)

    log.info(f"LCIdb final: {len(lci)} interactions, {lci.protein_id.nunique()} proteins, "
             f"{lci.ligand_smiles.nunique()} drugs")
    log.info(f"  pKi range: [{lci.pki.min():.1f}, {lci.pki.max():.1f}], "
             f"mean={lci.pki.mean():.2f}, median={lci.pki.median():.2f}")

    # ================================================================
    # Step 2: Load DTC training data and remove ALL overlap
    # ================================================================
    log.info("=" * 60)
    log.info("Step 2: Removing DTC training overlap")
    log.info("=" * 60)

    dtc = pd.read_csv(PROJECT / "data/processed/dtc_training_interactions.csv")
    seqs = json.load(open(PROJECT / "data/processed/merged_sequences.json"))

    # Recreate DTC 80/10/10 protein split (same as paper)
    esm35 = torch.load(PROJECT / "data/processed/esm2_35m_dtc.pt",
                        map_location="cpu", weights_only=False)
    esm650 = torch.load(PROJECT / "data/processed/esm2_650m_dtc.pt",
                         map_location="cpu", weights_only=False)
    esm35 = {k: v for k, v in esm35.items() if torch.isfinite(v).all()}
    esm650 = {k: v for k, v in esm650.items() if torch.isfinite(v).all()}

    dtc_valid = dtc[dtc.uniprot_id.isin(esm35)]
    all_prots = sorted(set(dtc_valid.uniprot_id) & set(esm35.keys()))
    rng = random.Random(SEED)
    rng.shuffle(all_prots)
    nt = max(1, int(len(all_prots) * 0.1))
    nv = max(1, int(len(all_prots) * 0.1))
    train_prots = set(all_prots[nt + nv:])
    dtc_train = dtc_valid[dtc_valid.uniprot_id.isin(train_prots)]
    log.info(f"DTC train split: {len(dtc_train)} interactions, {len(train_prots)} proteins")

    # Build DTC training sequences and canonical SMILES
    dtc_train_seqs = set()
    for uid in train_prots:
        if uid in seqs:
            dtc_train_seqs.add(seqs[uid])

    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.*")
    except ImportError:
        raise SystemExit("RDKit required")

    def canonicalize(smi):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToSmiles(mol, canonical=True) if mol else None

    def smiles_to_fp(smi):
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, useChirality=True)

    dtc_train_canonical = set()
    for smi in dtc_train.ligand_smiles.unique():
        c = canonicalize(smi)
        if c: dtc_train_canonical.add(c)
    log.info(f"DTC train: {len(dtc_train_seqs)} unique sequences, "
             f"{len(dtc_train_canonical)} unique canonical SMILES")

    # Remove overlap
    before = len(lci)

    # 2a. Protein sequence overlap
    lci_no_prot = lci[~lci.protein_sequence.isin(dtc_train_seqs)].copy()
    removed_prot = before - len(lci_no_prot)
    log.info(f"  Removed {removed_prot} interactions by protein sequence overlap "
             f"({before - len(lci_no_prot)} interactions, "
             f"{lci.protein_id.nunique() - lci_no_prot.protein_id.nunique()} proteins)")

    # 2b. Drug canonical SMILES overlap
    lci_canonical = {}
    for smi in lci_no_prot.ligand_smiles.unique():
        c = canonicalize(smi)
        if c: lci_canonical[smi] = c

    lci_no_prot["canonical"] = lci_no_prot.ligand_smiles.map(lci_canonical)
    lci_clean = lci_no_prot[~lci_no_prot.canonical.isin(dtc_train_canonical)].copy()
    removed_drug = len(lci_no_prot) - len(lci_clean)
    log.info(f"  Removed {removed_drug} interactions by drug canonical SMILES overlap")

    log.info(f"LCIdb after overlap removal: {len(lci_clean)} interactions "
             f"({lci_clean.protein_id.nunique()} proteins, "
             f"{lci_clean.ligand_smiles.nunique()} drugs)")
    log.info(f"  Total removed: {before - len(lci_clean)} ({100*(before-len(lci_clean))/before:.1f}%)")

    if len(lci_clean) < 100:
        log.error("Too few interactions after overlap removal!")
        return

    # ================================================================
    # Step 3: Compute ESM-2 embeddings for LCIdb proteins
    # ================================================================
    log.info("=" * 60)
    log.info("Step 3: ESM-2 embeddings for LCIdb proteins")
    log.info("=" * 60)

    lci_esm35_path = results_dir / "esm2_35m_lcidb.pt"
    lci_esm650_path = results_dir / "esm2_650m_lcidb.pt"

    # Unique proteins needing embeddings
    lci_prot_seqs = {}
    for _, r in lci_clean.drop_duplicates("protein_id").iterrows():
        lci_prot_seqs[r.protein_id] = r.protein_sequence

    if not args.skip_esm2:
        for model_name, dim, cache_path in [
            ("esm2_t6_8M_UR50D", 320, None),  # skip 8M
            ("esm2_t12_35M_UR50D", 480, lci_esm35_path),
            ("esm2_t33_650M_UR50D", 1280, lci_esm650_path),
        ]:
            if cache_path is None: continue
            if cache_path.exists():
                existing = torch.load(str(cache_path), map_location="cpu", weights_only=False)
                missing = {pid: seq for pid, seq in lci_prot_seqs.items() if pid not in existing}
                if not missing:
                    log.info(f"  {model_name}: {len(existing)} cached, 0 missing — skipping")
                    continue
                log.info(f"  {model_name}: {len(existing)} cached, {len(missing)} to compute")
            else:
                existing = {}
                missing = lci_prot_seqs

            log.info(f"  Computing {model_name} for {len(missing)} proteins...")
            import esm
            model_fn = getattr(esm.pretrained, model_name)
            esm_model, alphabet = model_fn()
            bc = alphabet.get_batch_converter()
            esm_model = esm_model.eval().to(device)
            repr_layer = {"esm2_t12_35M_UR50D": 12, "esm2_t33_650M_UR50D": 33}[model_name]

            for idx, (pid, seq) in enumerate(missing.items()):
                try:
                    _, _, toks = bc([(pid, seq[:1022])])
                    with torch.no_grad():
                        out = esm_model(toks.to(device), repr_layers=[repr_layer],
                                        return_contacts=False)
                        emb = out["representations"][repr_layer][0, 1:len(seq[:1022])+1, :].mean(0)
                        existing[pid] = emb.cpu()
                except Exception as e:
                    log.warning(f"  Failed {pid}: {e}")
                if (idx + 1) % 100 == 0:
                    log.info(f"    {idx+1}/{len(missing)}")
                    torch.save(existing, str(cache_path))

            torch.save(existing, str(cache_path))
            del esm_model
            torch.cuda.empty_cache()
            log.info(f"  Saved {len(existing)} embeddings → {cache_path.name}")

    # Load embeddings
    lci_esm35 = torch.load(str(lci_esm35_path), map_location="cpu", weights_only=False) if lci_esm35_path.exists() else {}
    lci_esm650 = torch.load(str(lci_esm650_path), map_location="cpu", weights_only=False) if lci_esm650_path.exists() else {}

    # Merge with DTC embeddings (for anchor lookup)
    all_esm35 = {**esm35, **lci_esm35}
    all_esm650 = {**esm650, **lci_esm650}

    # Filter to proteins with both embeddings
    valid_prots = set(lci_esm35.keys()) & set(lci_esm650.keys())
    lci_eval = lci_clean[lci_clean.protein_id.isin(valid_prots)].copy()
    log.info(f"LCIdb with ESM-2 embeddings: {len(lci_eval)} interactions, "
             f"{lci_eval.protein_id.nunique()} proteins")

    # ================================================================
    # Step 4: Build Tanimoto anchor pool and retrieve anchors
    # ================================================================
    log.info("=" * 60)
    log.info("Step 4: Tanimoto anchor retrieval from DTC")
    log.info("=" * 60)

    # Build anchor pool from DTC training set (pKi >= 7, no LCIdb canonical overlap)
    lci_canonical_set = set(lci_clean.canonical.dropna().unique())

    anchor_pool = {}
    for smi, group in dtc_train.groupby("ligand_smiles"):
        best = group.sort_values("pki", ascending=False).iloc[0]
        uid, pki = best["uniprot_id"], float(best["pki"])
        if pki < 7.0 or uid not in all_esm35: continue
        c = canonicalize(smi)
        if c and c in lci_canonical_set: continue  # exclude drugs also in LCIdb
        fp = smiles_to_fp(smi)
        if fp is None: continue
        anchor_pool[smi] = {"uid": uid, "pki": pki, "canonical": c, "fp": fp}
    log.info(f"Anchor pool (after LCIdb drug exclusion): {len(anchor_pool)} drugs")

    # Compute fingerprints for LCIdb drugs
    lci_drug_fps = {}
    for smi in lci_eval.ligand_smiles.unique():
        fp = smiles_to_fp(smi)
        if fp: lci_drug_fps[smi] = fp
    log.info(f"LCIdb drug fingerprints: {len(lci_drug_fps)}")

    # Tanimoto nearest-neighbor
    anchor_fps = [(smi, meta["fp"], meta) for smi, meta in anchor_pool.items()]
    nearest = {}
    for idx, (lci_smi, lci_fp) in enumerate(lci_drug_fps.items()):
        best_sim, best_meta, best_smi = -1, None, None
        for a_smi, a_fp, a_meta in anchor_fps:
            sim = DataStructs.TanimotoSimilarity(lci_fp, a_fp)
            if sim > best_sim:
                best_sim, best_meta, best_smi = sim, a_meta, a_smi
        if best_meta:
            nearest[lci_smi] = {"anchor_uid": best_meta["uid"], "anchor_pki": best_meta["pki"],
                                "tanimoto": best_sim, "anchor_drug": best_smi}
        if (idx + 1) % 1000 == 0:
            log.info(f"  Tanimoto search: {idx+1}/{len(lci_drug_fps)}")
    log.info(f"Tanimoto anchors: {len(nearest)}/{len(lci_drug_fps)} LCIdb drugs matched")

    # Build self-anchor exclusion (sequence-based)
    lci_seq_set = set(lci_eval.protein_sequence.unique())

    # Build anchored subset
    rows, anc_uids, anc_pkis, tanimotos = [], [], [], []
    for i, row in lci_eval.iterrows():
        smi = row["ligand_smiles"]
        if smi not in nearest: continue
        match = nearest[smi]
        au = match["anchor_uid"]
        # Self-anchor exclusion: skip if anchor protein sequence matches LCIdb protein
        if au in seqs and seqs[au] == row["protein_sequence"]:
            continue
        if au not in all_esm35 or au not in all_esm650: continue
        rows.append(i)
        anc_uids.append(au)
        anc_pkis.append(match["anchor_pki"])
        tanimotos.append(match["tanimoto"])

    subset = lci_eval.loc[rows].copy()
    subset["anchor_uid"] = anc_uids
    subset["anchor_pki"] = anc_pkis
    subset["tanimoto"] = tanimotos
    log.info(f"Anchored subset: {len(subset)} interactions, "
             f"{subset.protein_id.nunique()} proteins, {subset.ligand_smiles.nunique()} drugs")
    log.info(f"  Tanimoto mean={np.mean(tanimotos):.3f}, median={np.median(tanimotos):.3f}")

    # ================================================================
    # Step 5: Load models
    # ================================================================
    log.info("=" * 60)
    log.info("Step 5: Loading models")
    log.info("=" * 60)

    from anchor_transfer.model.anchor_transfer_v2 import AnchorTransferDTAv2
    from anchor_transfer.model.esm_dta import EsmDTAModel

    models = {}

    # V2-650M
    p = PROJECT / "models/v2_650m/best_model.pt"
    if p.exists():
        m = AnchorTransferDTAv2(esm2_dim=1280).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
        m.eval(); models["V2-650M"] = m; log.info("  Loaded V2-650M")

    # V2-35M
    p = PROJECT / "models/v2_35m/best_model.pt"
    if p.exists():
        m = AnchorTransferDTAv2(esm2_dim=480).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
        m.eval(); models["V2-35M"] = m; log.info("  Loaded V2-35M")

    # DeepDTA
    class DeepDTAModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
            self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
            self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
            self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
            self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1); self.do = nn.Dropout(0.1)
        def forward(self, s, p):
            s = self.smiles_embed(s).permute(0, 2, 1)
            s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s)); s = s.max(2)[0]
            p = self.protein_embed(p).permute(0, 2, 1)
            p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = p.max(2)[0]
            x = torch.cat([s, p], 1)
            x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
            x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)

    p = PROJECT / "models/deepdta_dtc/best_model.pt"
    if p.exists():
        m = DeepDTAModel().to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
        m.eval(); models["DeepDTA"] = m; log.info("  Loaded DeepDTA")

    # ESM-DTA
    p = PROJECT / "models/esm_dta_dtc/best_model.pt"
    if p.exists():
        m = EsmDTAModel(esm2_dim=480).to(device)
        m.load_state_dict(torch.load(p, map_location=device, weights_only=False)["model_state_dict"])
        m.eval(); models["ESM-DTA"] = m; log.info("  Loaded ESM-DTA")

    log.info(f"  Models: {list(models.keys())}")

    # ================================================================
    # Step 6: Predict on anchored subset
    # ================================================================
    log.info("=" * 60)
    log.info("Step 6: Inference on anchored subset")
    log.info("=" * 60)

    BS = 512
    all_preds = {m: [] for m in models}

    for start in range(0, len(subset), BS):
        batch = subset.iloc[start:start+BS]
        pids = batch.protein_id.values
        smis = batch.ligand_smiles.values
        seqs_batch = batch.protein_sequence.values
        ancs = batch.anchor_uid.values

        dt = torch.tensor([encode_smi(s) for s in smis], dtype=torch.long, device=device)

        with torch.no_grad():
            if "V2-650M" in models:
                a650 = torch.stack([all_esm650[a] for a in ancs]).to(device)
                q650 = torch.stack([lci_esm650[p] for p in pids]).to(device)
                pred = models["V2-650M"](a650, q650, dt)["pki_pred"].cpu().tolist()
                all_preds["V2-650M"].extend(pred)

            if "V2-35M" in models:
                a35 = torch.stack([all_esm35[a] for a in ancs]).to(device)
                q35 = torch.stack([lci_esm35[p] for p in pids]).to(device)
                pred = models["V2-35M"](a35, q35, dt)["pki_pred"].cpu().tolist()
                all_preds["V2-35M"].extend(pred)

            if "DeepDTA" in models:
                dp = torch.tensor([encode_prot(s) for s in seqs_batch], dtype=torch.long, device=device)
                pred = models["DeepDTA"](dt, dp).cpu().tolist()
                all_preds["DeepDTA"].extend(pred)

            if "ESM-DTA" in models:
                e35 = torch.stack([lci_esm35[p] for p in pids]).to(device)
                pred = models["ESM-DTA"](dt, e35).cpu().tolist()
                all_preds["ESM-DTA"].extend(pred)

        if (start + BS) % 10000 < BS:
            log.info(f"  {min(start+BS, len(subset))}/{len(subset)}")

    # ================================================================
    # Step 7: Evaluate and report
    # ================================================================
    log.info("=" * 60)
    log.info("Step 7: Results")
    log.info("=" * 60)

    true = subset.pki.values
    model_order = [m for m in ["V2-650M", "V2-35M", "DeepDTA", "ESM-DTA"] if m in models]

    print(f"\n{'='*90}")
    print(f"  LCIdb Cross-Dataset Evaluation (DTC → LCIdb, Tanimoto anchors)")
    print(f"  {len(subset)} interactions, {subset.protein_id.nunique()} proteins, "
          f"{subset.ligand_smiles.nunique()} drugs")
    print(f"  Zero overlap with DTC training (protein + drug + pair excluded)")
    print(f"{'='*90}")

    header = f"  {'Metric':<20s} {'N':>7s}"
    for m in model_order:
        header += f"  {m:>12s}"
    print(header)
    print(f"  {'-'*len(header)}")

    for metric_name, metric_fn in [
        ("CI", lambda t, p: ci_fn(t, p)),
        ("AUROC", lambda t, p: auroc_safe(t, p)),
        ("AUPRC", lambda t, p: auprc_safe(t, p)),
        ("RMSE", lambda t, p: float(np.sqrt(np.mean((t - p)**2)))),
        ("Pearson r", lambda t, p: float(pearsonr(t, p)[0]) if len(t) > 2 else float("nan")),
    ]:
        line = f"  {metric_name:<20s} {len(true):>7d}"
        for m in model_order:
            pred = np.array(all_preds[m])
            val = metric_fn(true, pred)
            line += f"  {val:>12.4f}"
        print(line)

    # Per-protein macro-averaged CI
    per_prot_cis = {m: [] for m in model_order}
    for pid in subset.protein_id.unique():
        mask = subset.protein_id.values == pid
        if mask.sum() < 2: continue
        t = true[mask]
        for m in model_order:
            p = np.array(all_preds[m])[mask]
            per_prot_cis[m].append(ci_fn(t, p))

    line = f"  {'CI (macro)':<20s} {len(per_prot_cis[model_order[0]]):>7d}"
    for m in model_order:
        line += f"  {np.mean(per_prot_cis[m]):>12.4f}"
    print(line)
    print(f"  {'-'*len(header)}")

    # ── Anchor pKi quartile breakdown ──
    print(f"\n  Anchor pKi Quartile Breakdown (CI):")
    q_edges = np.quantile(subset.anchor_pki.values, [0, 0.25, 0.5, 0.75, 1.0])
    for qi in range(4):
        lo, hi = q_edges[qi], q_edges[qi+1]
        if qi < 3:
            mask = (subset.anchor_pki.values >= lo) & (subset.anchor_pki.values < hi)
        else:
            mask = (subset.anchor_pki.values >= lo) & (subset.anchor_pki.values <= hi)
        if mask.sum() < 10: continue
        line = f"  Q{qi+1} [{lo:.1f}-{hi:.1f}] {mask.sum():>7d}"
        for m in model_order:
            p = np.array(all_preds[m])[mask]
            val = ci_fn(true[mask], p)
            line += f"  {val:>12.4f}"
        print(line)

    # ── Tanimoto similarity bin breakdown ──
    print(f"\n  Tanimoto Similarity Bin Breakdown (CI):")
    for lo, hi, label in [(0, 0.3, "<0.3"), (0.3, 0.5, "0.3-0.5"),
                          (0.5, 0.7, "0.5-0.7"), (0.7, 1.01, ">0.7")]:
        mask = (subset.tanimoto.values >= lo) & (subset.tanimoto.values < hi)
        if mask.sum() < 10: continue
        line = f"  Tan {label:<8s} {mask.sum():>7d}"
        for m in model_order:
            p = np.array(all_preds[m])[mask]
            val = ci_fn(true[mask], p)
            line += f"  {val:>12.4f}"
        print(line)

    print(f"\n{'='*90}")
    log.info("Done.")

    # Save predictions
    out_df = subset[["protein_id", "ligand_smiles", "pki", "anchor_uid", "anchor_pki", "tanimoto"]].copy()
    for m in model_order:
        out_df[f"pred_{m}"] = all_preds[m]
    out_path = results_dir / "lcidb_eval_predictions.csv"
    out_df.to_csv(out_path, index=False)
    log.info(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    main()
