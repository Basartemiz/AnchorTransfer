#!/usr/bin/env python3
"""Cross-dataset evaluation on LCIdb benchmark.

Evaluates CoNCISE (pretrained), ConciseAnchor (MooDeng binary checkpoint),
and Prot-kNN on LCIdb — a large-scale DTI dataset (ChEMBL/PubChem/BindingDB).

Binary classification: pKi >= 7 → positive, pKi <= 5 → negative,
ambiguous (5 < pKi < 7) excluded.

Protocol:
  1. Download LCIdb_v2.csv from Zenodo.
  2. Binarize: pKi >= 7 positive, pKi <= 5 negative, drop ambiguous.
  3. Remove ALL overlap with MooDeng v1 training data (protein sequence + drug SMILES).
  4. Compute Raygun embeddings + Morgan FPs for LCIdb proteins/drugs.
  5. Build anchor pool from MooDeng v1 training positives.
  6. Evaluate CoNCISE, ConciseAnchor, Prot-kNN on the same fair subset.
  7. Report AUROC, AUPRC.

Usage:
  python scripts/eval/eval_lcidb.py --device cuda
"""
import argparse, hashlib, logging, os, pickle, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, "src")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seq_to_id(seq: str) -> str:
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]


def compute_raygun_embeddings(sequences: dict, device: torch.device,
                              cache_path: Path, batch_label: str = "") -> dict:
    if cache_path.exists():
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        missing = {pid: seq for pid, seq in sequences.items() if pid not in cached}
        if not missing:
            log.info(f"  Loaded {len(cached)} cached Raygun embeddings from {cache_path.name}")
            return cached
        log.info(f"  Loaded {len(cached)} cached, {len(missing)} remaining")
    else:
        cached = {}
        missing = sequences

    if not missing:
        return cached

    log.info(f"  Computing Raygun embeddings for {len(missing)} {batch_label} proteins...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(device)
    raygun_model, _, _ = torch.hub.load("rohitsinghlab/raygun",
                                         "pretrained_uniref50_95000_750M")
    raygun_model = raygun_model.eval().to(device)

    items = [(pid, seq[:1022].upper()) for pid, seq in missing.items() if len(seq) >= 25]
    for idx, (pid, seq) in enumerate(items):
        try:
            _, _, toks = bc([(pid, seq)])
            with torch.no_grad():
                out = esm_model(toks.to(device), repr_layers=[33], return_contacts=False)
                esm_emb = out["representations"][33][0, 1:len(seq)+1, :]
                ray_out = raygun_model(esm_emb.unsqueeze(0))
                cached[pid] = ray_out[1].squeeze(0).cpu()
            del out, esm_emb, ray_out, toks
        except Exception:
            pass
        if (idx + 1) % 200 == 0:
            log.info(f"    {idx+1}/{len(items)} done")
            torch.save(cached, str(cache_path))

    del esm_model, raygun_model
    torch.cuda.empty_cache()
    torch.save(cached, str(cache_path))
    log.info(f"  Saved {len(cached)} Raygun embeddings -> {cache_path.name}")
    return cached


def compute_morgan_fps(smiles_list, cache_path: Path, fp_dict: dict = None) -> dict:
    if fp_dict is None:
        fp_dict = {}
    if cache_path.exists() and not fp_dict:
        fp_dict = pickle.load(open(cache_path, "rb"))
        log.info(f"  Loaded {len(fp_dict)} cached Morgan FPs from {cache_path.name}")

    from rdkit import Chem
    from rdkit.Chem import AllChem

    new_count = 0
    for smi in smiles_list:
        if smi in fp_dict:
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp_dict[smi] = np.array(
                    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048),
                    dtype=np.float32)
                new_count += 1
        except Exception:
            pass

    if new_count > 0:
        pickle.dump(fp_dict, open(cache_path, "wb"))
        log.info(f"  Computed {new_count} new Morgan FPs (total: {len(fp_dict)})")
    return fp_dict


# ---------------------------------------------------------------------------
# Model definition (matches MooDeng training)
# ---------------------------------------------------------------------------
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(
            ligand_dim=2048, residue_dim=1280, proj_dim=256,
            n_codes=3, dropout=0.2)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)

    def forward(self, drug_fp, anchor_emb, query_emb):
        return self.backbone(drug_fp, anchor_emb, query_emb)


# ---------------------------------------------------------------------------
# Prediction functions
# ---------------------------------------------------------------------------
BS = 1024


def predict_concise(prot_ids, smiles_arr, emb_dict, fp_dict, model, device):
    preds = np.full(len(prot_ids), np.nan)
    with torch.no_grad():
        for i in range(0, len(prot_ids), BS):
            bp, bs = prot_ids[i:i+BS], smiles_arr[i:i+BS]
            idx, fps, embs = [], [], []
            for j, (pid, smi) in enumerate(zip(bp, bs)):
                if pid in emb_dict and smi in fp_dict:
                    idx.append(i+j); fps.append(fp_dict[smi]); embs.append(emb_dict[pid])
            if not fps:
                continue
            pred = model(torch.tensor(np.array(fps)).to(device),
                         torch.stack(embs).to(device),
                         is_morgan_fingerprint=True)["binding"]
            scores = ((pred + 1) / 2).cpu().numpy()
            for j, ix in enumerate(idx):
                preds[ix] = scores[j]
            if (i + BS) % 100000 < BS:
                log.info(f"    CoNCISE: {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
    return preds


def predict_anchor(prot_ids, smiles_arr, emb_dict, fp_dict, model,
                   drug_to_anchors, all_embs, device):
    preds = np.full(len(prot_ids), np.nan)
    with torch.no_grad():
        for i in range(0, len(prot_ids), BS):
            bp, bs = prot_ids[i:i+BS], smiles_arr[i:i+BS]
            idx, fps, ancs, qrys = [], [], [], []
            for j, (pid, smi) in enumerate(zip(bp, bs)):
                if smi not in drug_to_anchors or smi not in fp_dict or pid not in emb_dict:
                    continue
                anc = None
                for a in drug_to_anchors[smi]:
                    if a != pid and a in all_embs:
                        anc = a; break
                if anc is None:
                    continue
                idx.append(i+j); fps.append(fp_dict[smi])
                ancs.append(all_embs[anc]); qrys.append(emb_dict[pid])
            if not fps:
                continue
            logit = model(torch.tensor(np.array(fps)).to(device),
                          torch.stack(ancs).to(device),
                          torch.stack(qrys).to(device))
            scores = torch.sigmoid(logit).cpu().numpy()
            for j, ix in enumerate(idx):
                preds[ix] = scores[j]
            if (i + BS) % 100000 < BS:
                log.info(f"    Anchor: {min(i+BS, len(prot_ids))}/{len(prot_ids)}")
    return preds


# ---------------------------------------------------------------------------
# Fair evaluation
# ---------------------------------------------------------------------------
def evaluate_fair(name, prot_ids, smiles_arr, labels, preds_dict):
    from sklearn.metrics import roc_auc_score, average_precision_score

    fair = np.ones(len(labels), dtype=bool)
    for preds in preds_dict.values():
        fair &= ~np.isnan(preds)
    n_fair = fair.sum()

    print(f"\n{'='*65}", flush=True)
    print(f"  {name}")
    print(f"  Fair subset: {n_fair:,} interactions, "
          f"{len(np.unique(prot_ids[fair])):,} proteins")
    print(f"  {int(labels[fair].sum()):,} pos, "
          f"{int((1-labels[fair]).sum()):,} neg "
          f"(1:{(1-labels[fair]).sum()/max(labels[fair].sum(),1):.1f})")
    print(f"{'='*65}")
    print(f"  {'Method':<22s} {'AUROC':>8s} {'AUPRC':>8s}")
    print(f"  {'-'*42}", flush=True)

    if n_fair >= 10 and len(set(labels[fair].astype(int))) >= 2:
        for mname, preds in preds_dict.items():
            auroc = roc_auc_score(labels[fair], preds[fair])
            auprc = average_precision_score(labels[fair], preds[fair])
            print(f"  {mname:<22s} {auroc:8.4f} {auprc:8.4f}")
    else:
        print(f"  (fair subset too small for comparison)")
    print(f"  {'-'*42}", flush=True)

    # Per-method coverage (always print)
    print(f"\n  {'Method':<22s} {'AUROC':>8s} {'AUPRC':>8s} {'Coverage':>12s}")
    print(f"  {'-'*55}")
    for mname, preds in preds_dict.items():
        valid = ~np.isnan(preds)
        if valid.sum() < 10 or len(set(labels[valid].astype(int))) < 2:
            print(f"  {mname:<22s}  insufficient coverage ({valid.sum()})")
            continue
        auroc = roc_auc_score(labels[valid], preds[valid])
        auprc = average_precision_score(labels[valid], preds[valid])
        print(f"  {mname:<22s} {auroc:8.4f} {auprc:8.4f} "
              f"{valid.sum():>6,}/{len(labels):,}")
    print(f"  {'-'*55}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="LCIdb evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-ckpt", required=True,
                        help="Path to ConciseAnchor binary checkpoint (MooDeng)")
    parser.add_argument("--lcidb-path", default="data/LCIdb_v2.csv")
    parser.add_argument("--moodeng-dir", default="data/moodeng-v1",
                        help="MooDeng v1 directory (for training data / anchor pool)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    moodeng = Path(args.moodeng_dir)

    # ── 1. Load LCIdb ──
    log.info("Step 1/7: Loading LCIdb...")
    lcidb_path = Path(args.lcidb_path)
    if not lcidb_path.exists():
        log.info(f"Downloading LCIdb_v2.csv...")
        lcidb_path.parent.mkdir(parents=True, exist_ok=True)
        import subprocess
        subprocess.run([
            "curl", "-L", "-o", str(lcidb_path),
            "https://zenodo.org/api/records/12178118/files/LCIdb_v2.csv/content"
        ], check=True)

    lci = pd.read_csv(lcidb_path, low_memory=False)
    log.info(f"  LCIdb raw: {len(lci)} interactions, {lci['fasta'].nunique()} proteins, "
             f"{lci['smiles'].nunique()} drugs")

    # Use mean pKi where available, fall back to mean pIC50, then score
    lci["pki"] = lci["mean pKi"]
    mask_no = lci["pki"].isna()
    lci.loc[mask_no, "pki"] = lci.loc[mask_no, "mean pIC50"]
    mask_no2 = lci["pki"].isna()
    lci.loc[mask_no2, "pki"] = lci.loc[mask_no2, "score"]
    lci = lci.dropna(subset=["pki"])
    lci = lci[lci["pki"] > 0]

    # Binarize: pKi >= 7 positive, pKi <= 5 negative, drop ambiguous
    lci_pos = lci[lci["pki"] >= 7].copy()
    lci_neg = lci[lci["pki"] <= 5].copy()
    lci_pos["label"] = 1.0
    lci_neg["label"] = 0.0
    lci_bin = pd.concat([lci_pos, lci_neg], ignore_index=True)
    lci_bin.rename(columns={"smiles": "smiles", "fasta": "sequence"}, inplace=True)
    lci_bin["prot_id"] = lci_bin["sequence"].apply(seq_to_id)
    log.info(f"  After binarization: {len(lci_bin)} interactions "
             f"({(lci_bin.label==1).sum()} pos, {(lci_bin.label==0).sum()} neg), "
             f"dropped {len(lci) - len(lci_bin)} ambiguous (5 < pKi < 7)")

    # ── 2. Load MooDeng v1 training data and remove overlap ──
    log.info("Step 2/7: Removing MooDeng v1 training overlap...")
    train_raw = pd.read_csv(moodeng / "train.csv", low_memory=False)
    train_raw.rename(columns={"Target Sequence": "sequence",
                               "SMILES": "smiles", "Label": "label"}, inplace=True)
    train_raw["prot_id"] = train_raw.sequence.apply(seq_to_id)

    train_seqs = set(train_raw.sequence.unique())
    train_drugs = set(train_raw.smiles.unique())

    before = len(lci_bin)
    # Remove protein overlap
    lci_bin = lci_bin[~lci_bin.sequence.isin(train_seqs)].copy()
    removed_prot = before - len(lci_bin)
    # Remove drug overlap
    before2 = len(lci_bin)
    lci_bin = lci_bin[~lci_bin.smiles.isin(train_drugs)].copy()
    removed_drug = before2 - len(lci_bin)

    log.info(f"  Removed {removed_prot} by protein overlap, {removed_drug} by drug overlap")
    log.info(f"  LCIdb clean: {len(lci_bin)} interactions "
             f"({lci_bin.prot_id.nunique()} proteins, {lci_bin.smiles.nunique()} drugs)")
    log.info(f"  {(lci_bin.label==1).sum()} pos, {(lci_bin.label==0).sum()} neg")

    if len(lci_bin) < 100:
        log.error("Too few interactions after overlap removal!")
        return

    # ── 3. Compute Raygun embeddings + Morgan FPs ──
    log.info("Step 3/7: Raygun embeddings for LCIdb proteins...")
    all_seqs = {}
    for df in [train_raw, lci_bin]:
        for _, r in df.drop_duplicates("prot_id").iterrows():
            all_seqs[r.prot_id] = r.sequence

    raygun_cache = results_dir / "raygun_lcidb_embeddings.pt"
    raygun_embs = compute_raygun_embeddings(all_seqs, device, raygun_cache,
                                            batch_label="LCIdb+MooDeng")

    log.info("Step 3/7: Morgan fingerprints...")
    fp_cache = results_dir / "morgan_lcidb_fp.pkl"
    all_smiles = list(set(train_raw.smiles.unique()) | set(lci_bin.smiles.unique()))
    fp_dict = compute_morgan_fps(all_smiles, fp_cache)

    # Filter to available
    lci_eval = lci_bin.loc[
        lci_bin.prot_id.isin(raygun_embs) & lci_bin.smiles.isin(fp_dict),
        ["prot_id", "smiles", "label", "sequence", "pki"]].copy()
    log.info(f"  Filtered to available: {len(lci_eval)} interactions")

    # ── 4. Build anchor pool + Tanimoto retrieval ──
    log.info("Step 4/8: Building anchor pool from MooDeng training...")
    train_pos = train_raw[train_raw.label == 1]
    train_filt = train_pos[
        train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]

    # Map each MooDeng training drug → its known protein binders
    moodeng_drug_to_anchors = {}
    for smi, grp in train_filt.groupby("smiles"):
        anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
        if anchors:
            moodeng_drug_to_anchors[smi] = anchors
    log.info(f"  MooDeng anchor pool: {len(moodeng_drug_to_anchors)} drugs with known binders")

    # Tanimoto retrieval: for each LCIdb drug, find nearest MooDeng drug
    log.info("  Computing Tanimoto retrieval (LCIdb drugs → MooDeng anchor pool)...")
    from rdkit import DataStructs
    moodeng_fps = [(smi, fp_dict[smi]) for smi in moodeng_drug_to_anchors if smi in fp_dict]
    moodeng_fp_array = np.array([fp for _, fp in moodeng_fps])
    moodeng_smi_array = [smi for smi, _ in moodeng_fps]

    # GPU-batched Tanimoto for speed
    moodeng_fp_gpu = torch.tensor(moodeng_fp_array, dtype=torch.float32, device=device)
    moodeng_bits = moodeng_fp_gpu.sum(1)

    drug_to_anchors = {}  # LCIdb drug → anchor proteins (via Tanimoto nearest MooDeng drug)
    tanimoto_sims = {}    # LCIdb drug → best Tanimoto similarity
    lci_unique_drugs = list(set(lci_eval.smiles.unique()) & set(fp_dict.keys()))
    CHUNK = 2000
    for ci in range(0, len(lci_unique_drugs), CHUNK):
        chunk_smis = lci_unique_drugs[ci:ci+CHUNK]
        chunk_fps = np.array([fp_dict[s] for s in chunk_smis])
        chunk_gpu = torch.tensor(chunk_fps, dtype=torch.float32, device=device)
        inter = chunk_gpu @ moodeng_fp_gpu.T
        q_bits = chunk_gpu.sum(1, keepdim=True)
        tani = inter / torch.clamp(q_bits + moodeng_bits.unsqueeze(0) - inter, min=1)
        best_idx = tani.argmax(dim=1).cpu().numpy()
        best_sim = tani.max(dim=1).values.cpu().numpy()
        for j, smi in enumerate(chunk_smis):
            nearest_smi = moodeng_smi_array[best_idx[j]]
            drug_to_anchors[smi] = moodeng_drug_to_anchors[nearest_smi]
            tanimoto_sims[smi] = float(best_sim[j])
        if (ci + CHUNK) % 20000 < CHUNK:
            log.info(f"    {min(ci+CHUNK, len(lci_unique_drugs))}/{len(lci_unique_drugs)}")

    del moodeng_fp_gpu, moodeng_fp_array
    torch.cuda.empty_cache()
    log.info(f"  Tanimoto anchors: {len(drug_to_anchors)} LCIdb drugs matched")
    if tanimoto_sims:
        sims = list(tanimoto_sims.values())
        log.info(f"  Tanimoto similarity: mean={np.mean(sims):.3f}, "
                 f"median={np.median(sims):.3f}, min={np.min(sims):.3f}, max={np.max(sims):.3f}")

    del train_raw, train_pos, train_filt
    import gc; gc.collect()

    # ── 5. Load models ──
    log.info("Step 5/7: Loading models...")
    concise_model = torch.hub.load("rohitsinghlab/CoNCISE",
                                    "pretrained_concise_v2", pretrained=True)
    concise_model = concise_model.eval().to(device)
    log.info(f"  CoNCISE loaded (pretrained_concise_v2)")

    anchor_model = ConciseAnchorBinary().to(device)
    ckpt = torch.load(args.anchor_ckpt, map_location=device, weights_only=False)
    anchor_model.load_state_dict(ckpt["model_state_dict"])
    anchor_model.eval()
    log.info(f"  ConciseAnchor loaded from {args.anchor_ckpt} "
             f"(epoch {ckpt.get('epoch', '?')})")

    # ── 5b. Build oracle anchors from LCIdb itself ──
    log.info("Step 5b/8: Building oracle anchors from LCIdb...")
    lci_pos_clean = lci_eval[lci_eval.label == 1.0]
    oracle_drug_to_anchors = {}
    for smi, grp in lci_pos_clean.groupby("smiles"):
        anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
        if anchors:
            oracle_drug_to_anchors[smi] = anchors
    log.info(f"  Oracle anchors: {len(oracle_drug_to_anchors)} drugs with LCIdb-internal binders")

    # ── 6. Predict ──
    log.info("Step 6/8: Predictions on LCIdb...")
    pids = lci_eval.prot_id.values
    smis = lci_eval.smiles.values
    labels = lci_eval.label.values.astype(float)

    p_concise = predict_concise(pids, smis, raygun_embs, fp_dict,
                                concise_model, device)
    del concise_model; torch.cuda.empty_cache()

    p_anchor = predict_anchor(pids, smis, raygun_embs, fp_dict,
                              anchor_model, drug_to_anchors, raygun_embs, device)

    # Oracle: same ConciseAnchor model but with LCIdb-internal anchors
    log.info("  Oracle (LCIdb-internal anchors)...")
    p_oracle = predict_anchor(pids, smis, raygun_embs, fp_dict,
                              anchor_model, oracle_drug_to_anchors, raygun_embs, device)
    del anchor_model; torch.cuda.empty_cache()

    # ── 7. Report ──
    log.info("Step 7/8: Results")

    all_methods = {
        "CoNCISE (pretrained)": p_concise,
        "ConciseAnchor": p_anchor,
        "ConciseAnchor-Oracle": p_oracle,
    }

    evaluate_fair("LCIdb Benchmark (pKi>=7 pos, pKi<=5 neg, zero MooDeng overlap)",
                  pids, smis, labels, all_methods)

    # ── 8. Quartile analysis by pKi ──
    log.info("Step 8/8: Quartile analysis by pKi")
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Get the original pKi values for quartile binning
    lci_eval_pkis = lci_eval["pki"].values if "pki" in lci_eval.columns else None

    # Oracle anchor pKi: for each interaction, get the pKi of the oracle anchor
    oracle_anchor_pkis = np.full(len(pids), np.nan)
    for i, (pid, smi) in enumerate(zip(pids, smis)):
        if smi not in oracle_drug_to_anchors: continue
        for a in oracle_drug_to_anchors[smi]:
            if a != pid and a in raygun_embs:
                # Get the pKi of this anchor-drug pair from LCIdb
                match = lci_pos_clean[(lci_pos_clean.prot_id == a) & (lci_pos_clean.smiles == smi)]
                if len(match) > 0 and "pki" in match.columns:
                    oracle_anchor_pkis[i] = float(match.iloc[0]["pki"])
                else:
                    oracle_anchor_pkis[i] = 7.0  # it's a positive (pKi >= 7)
                break

    # Quartile by oracle anchor pKi
    has_oracle_pki = ~np.isnan(oracle_anchor_pkis)
    if has_oracle_pki.sum() > 100:
        print(f"\n{'='*80}")
        print(f"  Quartile Analysis: by oracle anchor pKi")
        print(f"{'='*80}")

        valid_pkis = oracle_anchor_pkis[has_oracle_pki]
        q_edges = np.quantile(valid_pkis, [0, 0.25, 0.5, 0.75, 1.0])

        method_names = list(all_methods.keys())
        header = f"  {'Quartile':<20s} {'N':>7s}"
        for m in method_names:
            header += f"  {m:>20s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for qi in range(4):
            lo, hi = q_edges[qi], q_edges[qi+1]
            if qi < 3:
                mask = has_oracle_pki & (oracle_anchor_pkis >= lo) & (oracle_anchor_pkis < hi)
            else:
                mask = has_oracle_pki & (oracle_anchor_pkis >= lo) & (oracle_anchor_pkis <= hi)

            if mask.sum() < 20: continue
            lab = labels[mask]
            if len(set(lab.astype(int))) < 2: continue

            line = f"  Q{qi+1} [{lo:.1f}-{hi:.1f}] {mask.sum():>7d}"
            for m in method_names:
                p = np.array(all_methods[m])[mask]
                v = ~np.isnan(p)
                if v.sum() < 10 or len(set(lab[v].astype(int))) < 2:
                    line += f"  {'N/A':>20s}"
                else:
                    auroc = roc_auc_score(lab[v], p[v])
                    line += f"  {auroc:>20.4f}"
            print(line)

        # Overall
        lab = labels[has_oracle_pki]
        line = f"  {'Overall':<20s} {has_oracle_pki.sum():>7d}"
        for m in method_names:
            p = np.array(all_methods[m])[has_oracle_pki]
            v = ~np.isnan(p)
            if v.sum() < 10 or len(set(lab[v].astype(int))) < 2:
                line += f"  {'N/A':>20s}"
            else:
                auroc = roc_auc_score(lab[v], p[v])
                line += f"  {auroc:>20.4f}"
        print(line)

    # Also report by interaction pKi quartile (the true pKi of the query)
    if lci_eval_pkis is not None:
        print(f"\n{'='*80}")
        print(f"  Quartile Analysis: by interaction pKi (true affinity)")
        print(f"{'='*80}")

        q_edges = np.quantile(lci_eval_pkis, [0, 0.25, 0.5, 0.75, 1.0])

        header = f"  {'Quartile':<20s} {'N':>7s}"
        for m in method_names:
            header += f"  {m:>20s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for qi in range(4):
            lo, hi = q_edges[qi], q_edges[qi+1]
            if qi < 3:
                mask = (lci_eval_pkis >= lo) & (lci_eval_pkis < hi)
            else:
                mask = (lci_eval_pkis >= lo) & (lci_eval_pkis <= hi)

            if mask.sum() < 20: continue
            lab = labels[mask]
            if len(set(lab.astype(int))) < 2: continue

            line = f"  Q{qi+1} [{lo:.1f}-{hi:.1f}] {mask.sum():>7d}"
            for m in method_names:
                p = np.array(all_methods[m])[mask]
                v = ~np.isnan(p)
                if v.sum() < 10 or len(set(lab[v].astype(int))) < 2:
                    line += f"  {'N/A':>20s}"
                else:
                    auroc = roc_auc_score(lab[v], p[v])
                    line += f"  {auroc:>20.4f}"
            print(line)

    print(f"\n{'='*80}")
    log.info("\nAll done.")


if __name__ == "__main__":
    main()
