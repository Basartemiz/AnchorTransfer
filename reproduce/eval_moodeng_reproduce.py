#!/usr/bin/env python3
"""Reproducible evaluation: ConciseAnchor vs pretrained CoNCISE on MooDengDB.

Evaluates on:
  1. MooDengDB IID test set (binary, same-distribution)
  2. MooDengDB OOD test set (LCIdb holdout, unseen ligands)

All methods evaluated on the SAME interaction subset (fair comparison).
Methods: pretrained CoNCISE, ConciseAnchor (binary), Prot-kNN k=1.

Usage:
  python reproduce/eval_moodeng_reproduce.py \\
    --device cuda --anchor-ckpt models/concise_anchor_moodeng/best_model.pt \\
    --moodeng-dir data/moodeng-v1 --ood-test data/moodeng-v2-extended/test.csv
"""
import argparse, hashlib, logging, os, pickle, sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def seq_to_id(seq: str) -> str:
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]


def compute_raygun_embeddings(sequences: dict, device: torch.device,
                              cache_path: Path, batch_label: str = "") -> dict:
    """Compute Raygun (50×1280) embeddings, streaming one protein at a time."""
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
                cached[pid] = ray_out[1].squeeze(0).cpu()  # (50, 1280) compressed
            del out, esm_emb, ray_out, toks
        except Exception:
            pass
        if (idx + 1) % 200 == 0:
            log.info(f"    {idx+1}/{len(items)} done")
            torch.save(cached, str(cache_path))

    del esm_model, raygun_model
    torch.cuda.empty_cache()
    torch.save(cached, str(cache_path))
    log.info(f"  Saved {len(cached)} Raygun embeddings → {cache_path.name}")
    return cached


def compute_morgan_fps(smiles_list, cache_path: Path, fp_dict: dict = None) -> dict:
    """Compute Morgan fingerprints (radius=2, 2048 bits)."""
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
# Model definition (must match training exactly)
# ---------------------------------------------------------------------------
from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class ConciseAnchorBinary(nn.Module):
    """Binary wrapper: bilinear backbone → raw logit (BCEWithLogitsLoss at train,
    sigmoid at eval)."""
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
BS = 1024  # batch size for GPU inference


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


def predict_knn(prot_ids, smiles_arr, emb_dict, fp_dict,
                train_prot_normed, train_drug_fps, int_mat, device):
    """Prot-kNN k=1 with GPU-batched drug Tanimoto."""
    preds = np.full(len(prot_ids), np.nan)

    # Protein top-1
    unique_prots = list(set(prot_ids))
    prot_embs = np.array([
        emb_dict[p].mean(dim=0).numpy() if isinstance(emb_dict[p], torch.Tensor)
        else emb_dict[p].mean(axis=0)
        for p in unique_prots])
    qn = prot_embs / (np.linalg.norm(prot_embs, axis=1, keepdims=True) + 1e-10)
    ps = qn @ train_prot_normed.T
    prot_topk = {}
    for i, p in enumerate(unique_prots):
        best = np.argmax(ps[i])
        if ps[i, best] > 0:
            prot_topk[p] = (best, ps[i, best])
    log.info(f"    kNN: protein top-1 computed ({len(unique_prots)} proteins)")

    # Drug Tanimoto on GPU
    unique_drugs = list(set(smiles_arr))
    drug_fps_np = np.array([fp_dict[d] for d in unique_drugs if d in fp_dict])
    valid_drugs = [d for d in unique_drugs if d in fp_dict]
    drug_idx_map = {d: i for i, d in enumerate(valid_drugs)}

    CHUNK = 4000
    train_fps_gpu = torch.tensor(train_drug_fps, dtype=torch.float32).to(device)
    train_bits = train_fps_gpu.sum(1)
    ds = np.empty((len(valid_drugs), train_drug_fps.shape[0]), dtype=np.float32)
    for ci in range(0, len(valid_drugs), CHUNK):
        chunk = torch.tensor(drug_fps_np[ci:ci+CHUNK], dtype=torch.float32).to(device)
        inter = chunk @ train_fps_gpu.T
        q_bits = chunk.sum(1, keepdim=True)
        tani = inter / torch.clamp(q_bits + train_bits.unsqueeze(0) - inter, min=1)
        ds[ci:ci+CHUNK] = tani.cpu().numpy()
    del train_fps_gpu
    torch.cuda.empty_cache()
    log.info(f"    kNN: drug Tanimoto computed ({len(valid_drugs)} drugs)")

    # Score each interaction
    for i in range(len(prot_ids)):
        pid, smi = prot_ids[i], smiles_arr[i]
        if pid not in prot_topk or smi not in drug_idx_map:
            continue
        best_pi, best_psim = prot_topk[pid]
        bound_mask = int_mat[:, best_pi] > 0
        if not bound_mask.any():
            continue
        di = drug_idx_map[smi]
        max_sim = ds[di, bound_mask].max()
        if max_sim > 0:
            preds[i] = max_sim * best_psim
        if (i + 1) % 200000 == 0:
            log.info(f"    kNN scoring: {i+1}/{len(prot_ids)}")
    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_fair(name, prot_ids, smiles_arr, labels, preds_dict):
    """Report AUROC/AUPRC on the fair subset (all methods have predictions)."""
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

    if n_fair < 10 or len(set(labels[fair].astype(int))) < 2:
        print(f"  Too few interactions or single class in fair subset")
        return

    for mname, preds in preds_dict.items():
        auroc = roc_auc_score(labels[fair], preds[fair])
        auprc = average_precision_score(labels[fair], preds[fair])
        print(f"  {mname:<22s} {auroc:8.4f} {auprc:8.4f}")
    print(f"  {'-'*42}", flush=True)

    # Also show individual coverage
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
    parser = argparse.ArgumentParser(description="MooDengDB evaluation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-ckpt", required=True,
                        help="Path to ConciseAnchor binary checkpoint")
    parser.add_argument("--moodeng-dir", default="data/moodeng-v1",
                        help="Path to MooDengDB v1 directory (train.csv, test.csv)")
    parser.add_argument("--ood-test", default="data/moodeng-v2-extended/test.csv",
                        help="Path to OOD test CSV (tab-separated)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)
    moodeng = Path(args.moodeng_dir)

    # ── 1. Load MooDengDB training data ──
    log.info("Step 1/7: Loading MooDengDB training data...")
    train_raw = pd.read_csv(moodeng / "train.csv", low_memory=False)
    test_raw = pd.read_csv(moodeng / "test.csv", low_memory=False)
    for df in [train_raw, test_raw]:
        df.rename(columns={"Target Sequence": "sequence",
                           "SMILES": "smiles", "Label": "label"}, inplace=True)
        df["prot_id"] = df.sequence.apply(seq_to_id)

    all_seqs = {}
    for df in [train_raw, test_raw]:
        for _, r in df.drop_duplicates("prot_id").iterrows():
            all_seqs[r.prot_id] = r.sequence
    log.info(f"  Train: {len(train_raw):,}, Test: {len(test_raw):,}, "
             f"Proteins: {len(all_seqs):,}")

    # ── 2. Compute/load Raygun embeddings + Morgan FPs ──
    log.info("Step 2/7: Raygun embeddings for MooDengDB...")
    raygun_cache = results_dir / "raygun_moodeng_embeddings.pt"
    raygun_embs = compute_raygun_embeddings(all_seqs, device, raygun_cache,
                                            batch_label="MooDengDB")

    log.info("Step 2/7: Morgan fingerprints...")
    fp_cache = results_dir / "morgan_moodeng_fp.pkl"
    all_smiles = list(set(train_raw.smiles.unique()) | set(test_raw.smiles.unique()))
    fp_dict = compute_morgan_fps(all_smiles, fp_cache)

    # Filter to available
    train_df = train_raw.loc[
        train_raw.prot_id.isin(raygun_embs) & train_raw.smiles.isin(fp_dict),
        ["prot_id", "smiles", "label"]].copy()
    test_df = test_raw.loc[
        test_raw.prot_id.isin(raygun_embs) & test_raw.smiles.isin(fp_dict),
        ["prot_id", "smiles", "label"]].copy()
    log.info(f"  Filtered — Train: {len(train_df):,}, Test: {len(test_df):,}")

    # ── 3. Build anchor pool + kNN retrieval structures ──
    log.info("Step 3/7: Building anchor pool and kNN index...")
    train_pos = train_df[train_df.label == 1]
    drug_to_anchors = {}
    for smi, grp in train_pos.groupby("smiles"):
        anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
        if anchors:
            drug_to_anchors[smi] = anchors
    log.info(f"  Anchors: {len(drug_to_anchors):,} drugs with known binders")

    train_prot_ids = sorted(set(train_pos.prot_id) & set(raygun_embs.keys()))
    train_prot_embs = np.array([raygun_embs[p].mean(dim=0).numpy()
                                 for p in train_prot_ids])
    train_prot_normed = train_prot_embs / (
        np.linalg.norm(train_prot_embs, axis=1, keepdims=True) + 1e-10)

    train_drug_ids = sorted(set(train_pos.smiles) & set(fp_dict.keys()))
    train_drug_fps = np.array([fp_dict[d] for d in train_drug_ids])
    train_drug_idx = {d: i for i, d in enumerate(train_drug_ids)}
    train_prot_idx = {p: i for i, p in enumerate(train_prot_ids)}

    int_mat = np.zeros((len(train_drug_ids), len(train_prot_ids)), dtype=np.float32)
    for _, r in train_pos.iterrows():
        di = train_drug_idx.get(r.smiles, -1)
        pi = train_prot_idx.get(r.prot_id, -1)
        if di >= 0 and pi >= 0:
            int_mat[di, pi] = 1.0
    log.info(f"  Interaction matrix: {int_mat.shape}, nnz={np.count_nonzero(int_mat):,}")

    del train_raw, test_raw, train_pos
    import gc; gc.collect()

    # ── 4. Load models ──
    log.info("Step 4/7: Loading models...")
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

    # ── 5. Evaluate on MooDengDB IID test ──
    log.info("Step 5/7: MooDengDB IID test predictions...")
    pids = test_df.prot_id.values
    smis = test_df.smiles.values
    labels = test_df.label.values.astype(float)

    p_concise = predict_concise(pids, smis, raygun_embs, fp_dict,
                                concise_model, device)
    p_anchor = predict_anchor(pids, smis, raygun_embs, fp_dict,
                              anchor_model, drug_to_anchors, raygun_embs, device)
    p_knn = predict_knn(pids, smis, raygun_embs, fp_dict,
                        train_prot_normed, train_drug_fps, int_mat, device)

    evaluate_fair("MooDengDB IID Test",
                  pids, smis, labels,
                  {"CoNCISE (pretrained)": p_concise,
                   "ConciseAnchor": p_anchor,
                   "Prot-kNN k=1": p_knn})

    del concise_model
    torch.cuda.empty_cache()

    # ── 6. Load OOD test + compute embeddings ──
    log.info("Step 6/7: Loading OOD test set...")
    ood_raw = pd.read_csv(args.ood_test, sep="\t", low_memory=False)
    ood_raw.rename(columns={"Target Sequence": "sequence",
                            "SMILES": "smiles", "Label": "label"}, inplace=True)
    ood_raw["prot_id"] = ood_raw.sequence.apply(seq_to_id)
    log.info(f"  OOD: {len(ood_raw):,} int, {ood_raw.smiles.nunique():,} drugs, "
             f"{(ood_raw.label==1).sum():,} pos, {(ood_raw.label==0).sum():,} neg")

    # OOD Raygun embeddings
    ood_seqs = {seq_to_id(s): s for s in ood_raw.sequence.unique()
                if seq_to_id(s) not in raygun_embs}
    if ood_seqs:
        ood_cache = results_dir / "raygun_ood_embeddings.pt"
        ood_raygun = compute_raygun_embeddings(ood_seqs, device, ood_cache,
                                                batch_label="OOD")
        for pid, emb in ood_raygun.items():
            raygun_embs[pid] = emb

    # OOD Morgan FPs
    ood_smiles = list(ood_raw.smiles.unique())
    fp_dict = compute_morgan_fps(ood_smiles, fp_cache, fp_dict)

    # Filter to anchor subset (fair comparison from the start)
    def has_anchor(row):
        smi, pid = row.smiles, row.prot_id
        if smi not in drug_to_anchors or smi not in fp_dict or pid not in raygun_embs:
            return False
        for a in drug_to_anchors[smi]:
            if a != pid and a in raygun_embs:
                return True
        return False

    log.info("  Filtering OOD to anchor subset...")
    mask = ood_raw.apply(has_anchor, axis=1)
    ood_df = ood_raw[mask].copy()
    log.info(f"  Anchor subset: {len(ood_df):,} ({100*len(ood_df)/len(ood_raw):.1f}%), "
             f"{(ood_df.label==1).sum():,} pos, {(ood_df.label==0).sum():,} neg")
    del ood_raw; gc.collect()

    # ── 7. Evaluate on OOD ──
    log.info("Step 7/7: OOD predictions...")
    concise_model = torch.hub.load("rohitsinghlab/CoNCISE",
                                    "pretrained_concise_v2", pretrained=True)
    concise_model = concise_model.eval().to(device)

    pids_ood = ood_df.prot_id.values
    smis_ood = ood_df.smiles.values
    labels_ood = ood_df.label.values.astype(float)

    p_concise_ood = predict_concise(pids_ood, smis_ood, raygun_embs, fp_dict,
                                     concise_model, device)
    del concise_model; torch.cuda.empty_cache()

    p_anchor_ood = predict_anchor(pids_ood, smis_ood, raygun_embs, fp_dict,
                                   anchor_model, drug_to_anchors, raygun_embs, device)
    del anchor_model; torch.cuda.empty_cache()

    p_knn_ood = predict_knn(pids_ood, smis_ood, raygun_embs, fp_dict,
                             train_prot_normed, train_drug_fps, int_mat, device)

    evaluate_fair("MooDengDB OOD Test (LCIdb holdout, anchor subset)",
                  pids_ood, smis_ood, labels_ood,
                  {"CoNCISE (pretrained)": p_concise_ood,
                   "ConciseAnchor": p_anchor_ood,
                   "Prot-kNN k=1": p_knn_ood})

    log.info("\nAll done.")


if __name__ == "__main__":
    main()
