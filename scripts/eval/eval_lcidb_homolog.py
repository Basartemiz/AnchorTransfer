#!/usr/bin/env python3
"""LCIdb evaluation with 30% sequence homology filtering.

Same as eval_lcidb.py but adds:
  1. V2-extended train overlap removal (protein + drug)
  2. 30% sequence identity homology filtering via MMseqs2:
     any LCIdb protein with >30% identity to any MooDeng v1+v2 train protein
     is removed.

Usage:
  python scripts/eval/eval_lcidb_homolog.py --device cuda \
    --anchor-ckpt models/concise_anchor_moodeng/best_model.pt
"""
import argparse, hashlib, logging, os, pickle, subprocess, sys, tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
sys.path.insert(0, "src")

BS = 1024


def seq_to_id(seq: str) -> str:
    return "P" + hashlib.md5(seq.encode()).hexdigest()[:12]


def write_fasta(seqs: dict, path: Path):
    with open(path, "w") as f:
        for pid, seq in seqs.items():
            f.write(f">{pid}\n{seq}\n")


def homology_filter_mmseqs2(query_seqs: dict, target_seqs: dict,
                             identity_threshold: float = 0.3) -> set:
    """Return set of query protein IDs that have >identity_threshold match to any target."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_fasta = tmpdir / "query.fasta"
        target_fasta = tmpdir / "target.fasta"
        write_fasta(query_seqs, query_fasta)
        write_fasta(target_seqs, target_fasta)

        result_file = tmpdir / "result.tsv"
        querydb = tmpdir / "queryDB"
        targetdb = tmpdir / "targetDB"
        resultdb = tmpdir / "resultDB"
        tmp = tmpdir / "tmp"

        log.info(f"  MMseqs2: {len(query_seqs)} query × {len(target_seqs)} target "
                 f"(threshold={identity_threshold})")

        subprocess.run(["mmseqs", "createdb", str(query_fasta), str(querydb)],
                       capture_output=True, check=True)
        subprocess.run(["mmseqs", "createdb", str(target_fasta), str(targetdb)],
                       capture_output=True, check=True)
        subprocess.run([
            "mmseqs", "search", str(querydb), str(targetdb), str(resultdb), str(tmp),
            "--min-seq-id", str(identity_threshold),
            "-s", "7.5",  # sensitivity
            "--threads", "4",
        ], capture_output=True, check=True)
        subprocess.run([
            "mmseqs", "convertalis", str(querydb), str(targetdb), str(resultdb),
            str(result_file), "--format-output", "query,target,fident",
        ], capture_output=True, check=True)

        # Read hits
        homologs = set()
        if result_file.exists():
            for line in open(result_file):
                parts = line.strip().split("\t")
                if len(parts) >= 3 and float(parts[2]) >= identity_threshold:
                    homologs.add(parts[0])

        log.info(f"  MMseqs2: {len(homologs)} query proteins have >{identity_threshold*100:.0f}% "
                 f"identity to target")
        return homologs


def compute_raygun_embeddings(sequences, device, cache_path, batch_label=""):
    if cache_path.exists():
        cached = torch.load(str(cache_path), map_location="cpu", weights_only=False)
        missing = {pid: seq for pid, seq in sequences.items() if pid not in cached}
        if not missing:
            log.info(f"  Loaded {len(cached)} cached Raygun embeddings")
            return cached
        log.info(f"  Loaded {len(cached)} cached, {len(missing)} remaining")
    else:
        cached = {}
        missing = sequences
    if not missing:
        return cached

    log.info(f"  Computing Raygun for {len(missing)} {batch_label} proteins...")
    import esm
    esm_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = alphabet.get_batch_converter()
    esm_model = esm_model.eval().to(device)
    raygun, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raygun = raygun.eval().to(device)

    for idx, (pid, seq) in enumerate(missing.items()):
        if len(seq) < 25: continue
        try:
            _, _, toks = bc([(pid, seq[:1022])])
            with torch.no_grad():
                out = esm_model(toks.to(device), repr_layers=[33], return_contacts=False)
                e = out["representations"][33][0, 1:len(seq[:1022])+1, :]
                r = raygun(e.unsqueeze(0))
                cached[pid] = r[1].squeeze(0).cpu()
            del out, e, r, toks
        except: pass
        if (idx+1) % 200 == 0:
            log.info(f"    {idx+1}/{len(missing)}")
            torch.save(cached, str(cache_path))
    del esm_model, raygun; torch.cuda.empty_cache()
    torch.save(cached, str(cache_path))
    return cached


def compute_morgan_fps(smiles_list, cache_path, fp_dict=None):
    if fp_dict is None: fp_dict = {}
    if cache_path.exists() and not fp_dict:
        fp_dict = pickle.load(open(cache_path, "rb"))
        log.info(f"  Loaded {len(fp_dict)} cached Morgan FPs")
    from rdkit import Chem
    from rdkit.Chem import AllChem
    new = 0
    for smi in smiles_list:
        if smi in fp_dict: continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp_dict[smi] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)
                new += 1
        except: pass
    if new > 0:
        pickle.dump(fp_dict, open(cache_path, "wb"))
        log.info(f"  Computed {new} new Morgan FPs (total: {len(fp_dict)})")
    return fp_dict


from anchor_transfer.model.concise_anchor_bilinear import ConciseAnchorBilinear

class ConciseAnchorBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConciseAnchorBilinear(
            ligand_dim=2048, residue_dim=1280, proj_dim=256, n_codes=3, dropout=0.2)
        nn.init.constant_(self.backbone.regressor[-1].bias, 0.0)
    def forward(self, drug_fp, anchor_emb, query_emb):
        return self.backbone(drug_fp, anchor_emb, query_emb)


def predict_concise(prot_ids, smiles_arr, emb_dict, fp_dict, model, device):
    preds = np.full(len(prot_ids), np.nan)
    with torch.no_grad():
        for i in range(0, len(prot_ids), BS):
            bp, bs = prot_ids[i:i+BS], smiles_arr[i:i+BS]
            idx, fps, embs = [], [], []
            for j, (pid, smi) in enumerate(zip(bp, bs)):
                if pid in emb_dict and smi in fp_dict:
                    idx.append(i+j); fps.append(fp_dict[smi]); embs.append(emb_dict[pid])
            if not fps: continue
            pred = model(torch.tensor(np.array(fps)).to(device),
                         torch.stack(embs).to(device),
                         is_morgan_fingerprint=True)["binding"]
            scores = ((pred + 1) / 2).cpu().numpy()
            for j, ix in enumerate(idx): preds[ix] = scores[j]
            if (i+BS) % 100000 < BS:
                log.info(f"    CoNCISE: {min(i+BS,len(prot_ids))}/{len(prot_ids)}")
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
                if anc is None: continue
                idx.append(i+j); fps.append(fp_dict[smi])
                ancs.append(all_embs[anc]); qrys.append(emb_dict[pid])
            if not fps: continue
            logit = model(torch.tensor(np.array(fps)).to(device),
                          torch.stack(ancs).to(device),
                          torch.stack(qrys).to(device))
            scores = torch.sigmoid(logit).cpu().numpy()
            for j, ix in enumerate(idx): preds[ix] = scores[j]
            if (i+BS) % 100000 < BS:
                log.info(f"    Anchor: {min(i+BS,len(prot_ids))}/{len(prot_ids)}")
    return preds


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
    n_pos = int(labels[fair].sum()) if n_fair > 0 else 0
    n_neg = int((1-labels[fair]).sum()) if n_fair > 0 else 0
    print(f"  {n_pos} pos, {n_neg} neg")
    print(f"{'='*65}")
    print(f"  {'Method':<22s} {'AUROC':>8s} {'AUPRC':>8s}")
    print(f"  {'-'*42}", flush=True)

    if n_fair >= 10 and len(set(labels[fair].astype(int))) >= 2:
        for mname, preds in preds_dict.items():
            auroc = roc_auc_score(labels[fair], preds[fair])
            auprc = average_precision_score(labels[fair], preds[fair])
            print(f"  {mname:<22s} {auroc:8.4f} {auprc:8.4f}")
    print(f"  {'-'*42}")

    print(f"\n  {'Method':<22s} {'AUROC':>8s} {'AUPRC':>8s} {'Coverage':>12s}")
    print(f"  {'-'*55}")
    for mname, preds in preds_dict.items():
        valid = ~np.isnan(preds)
        if valid.sum() < 10 or len(set(labels[valid].astype(int))) < 2:
            print(f"  {mname:<22s}  insufficient ({valid.sum()})")
            continue
        auroc = roc_auc_score(labels[valid], preds[valid])
        auprc = average_precision_score(labels[valid], preds[valid])
        print(f"  {mname:<22s} {auroc:8.4f} {auprc:8.4f} "
              f"{valid.sum():>6,}/{len(labels):,}")
    print(f"  {'-'*55}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--anchor-ckpt", required=True)
    parser.add_argument("--lcidb-path", default="data/LCIdb_v2.csv")
    parser.add_argument("--moodeng-dir", default="data/moodeng-v1")
    parser.add_argument("--moodeng-v2-dir", default="data/moodeng-v2-extended")
    parser.add_argument("--identity-threshold", type=float, default=0.3,
                        help="Sequence identity threshold for homology filtering (default 0.3)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir); results_dir.mkdir(exist_ok=True)

    # ── 1. Load LCIdb ──
    log.info("Step 1/8: Loading LCIdb...")
    lci = pd.read_csv(args.lcidb_path, low_memory=False)
    log.info(f"  Raw: {len(lci)} interactions")

    lci["pki"] = lci["mean pKi"].fillna(lci["mean pIC50"]).fillna(lci["score"])
    lci = lci.dropna(subset=["pki"])
    lci = lci[lci.pki > 0]

    lci_pos = lci[lci.pki >= 7].copy(); lci_pos["label"] = 1.0
    lci_neg = lci[lci.pki <= 5].copy(); lci_neg["label"] = 0.0
    lci_bin = pd.concat([lci_pos, lci_neg], ignore_index=True)
    lci_bin["prot_id"] = lci_bin["fasta"].apply(seq_to_id)
    lci_bin.rename(columns={"smiles": "smiles", "fasta": "sequence"}, inplace=True)
    log.info(f"  Binarized: {len(lci_bin)} ({(lci_bin.label==1).sum()} pos, "
             f"{(lci_bin.label==0).sum()} neg)")

    # ── 2. Remove V1 + V2 train overlap (exact match) ──
    log.info("Step 2/8: Removing V1 + V2 training overlap (exact)...")
    v1_train = pd.read_csv(Path(args.moodeng_dir) / "train.csv", low_memory=False)
    v1_train.rename(columns={"Target Sequence": "sequence", "SMILES": "smiles", "Label": "label"}, inplace=True)
    v1_train["prot_id"] = v1_train.sequence.apply(seq_to_id)

    train_seqs = set(v1_train.sequence.unique())
    train_drugs = set(v1_train.smiles.unique())

    # Add V2 train overlap
    v2_dir = Path(args.moodeng_v2_dir)
    if (v2_dir / "train.csv").exists():
        v2_train = pd.read_csv(v2_dir / "train.csv", sep="\t", low_memory=False)
        v2_train_seqs = set(v2_train["Target Sequence"].unique())
        v2_train_drugs = set(v2_train["SMILES"].unique())
        train_seqs |= v2_train_seqs
        train_drugs |= v2_train_drugs
        log.info(f"  V2 train: {len(v2_train_seqs)} proteins, {len(v2_train_drugs)} drugs added")
        del v2_train

    before = len(lci_bin)
    lci_bin = lci_bin[~lci_bin.sequence.isin(train_seqs)].copy()
    removed_prot = before - len(lci_bin)
    before2 = len(lci_bin)
    lci_bin = lci_bin[~lci_bin.smiles.isin(train_drugs)].copy()
    removed_drug = before2 - len(lci_bin)
    log.info(f"  Removed {removed_prot} by protein, {removed_drug} by drug (exact)")
    log.info(f"  After exact: {len(lci_bin)} ({lci_bin.prot_id.nunique()} proteins)")

    # ── 3. Homology filtering with MMseqs2 ──
    log.info(f"Step 3/8: Homology filtering (>{args.identity_threshold*100:.0f}% identity)...")

    # Build query (LCIdb proteins) and target (all train proteins)
    lci_seqs = {r.prot_id: r.sequence for _, r in lci_bin.drop_duplicates("prot_id").iterrows()}
    train_seq_dict = {seq_to_id(s): s for s in train_seqs}

    homologs = homology_filter_mmseqs2(lci_seqs, train_seq_dict, args.identity_threshold)

    before3 = len(lci_bin)
    lci_bin = lci_bin[~lci_bin.prot_id.isin(homologs)].copy()
    removed_homolog = before3 - len(lci_bin)
    log.info(f"  Removed {removed_homolog} interactions by homology "
             f"({len(homologs)} proteins)")
    log.info(f"  After homology: {len(lci_bin)} ({lci_bin.prot_id.nunique()} proteins, "
             f"{lci_bin.smiles.nunique()} drugs)")
    log.info(f"  {(lci_bin.label==1).sum()} pos, {(lci_bin.label==0).sum()} neg")

    if len(lci_bin) < 100:
        log.error("Too few interactions after filtering!")
        return

    # ── 4. Raygun + Morgan ──
    log.info("Step 4/8: Embeddings...")
    all_seqs = {}
    for _, r in v1_train.drop_duplicates("prot_id").iterrows():
        all_seqs[r.prot_id] = r.sequence
    for _, r in lci_bin.drop_duplicates("prot_id").iterrows():
        all_seqs[r.prot_id] = r.sequence

    raygun_cache = results_dir / "raygun_lcidb_embeddings.pt"
    raygun_embs = compute_raygun_embeddings(all_seqs, device, raygun_cache, "LCIdb+MooDeng")

    fp_cache = results_dir / "morgan_lcidb_fp.pkl"
    all_smiles = list(set(v1_train.smiles.unique()) | set(lci_bin.smiles.unique()))
    fp_dict = compute_morgan_fps(all_smiles, fp_cache)

    lci_eval = lci_bin.loc[
        lci_bin.prot_id.isin(raygun_embs) & lci_bin.smiles.isin(fp_dict),
        ["prot_id", "smiles", "label", "sequence", "pki"]].copy()
    log.info(f"  Filtered: {len(lci_eval)} interactions")

    # ── 5. Anchor pool + Tanimoto retrieval ──
    log.info("Step 5/8: Tanimoto anchor retrieval...")
    train_pos = v1_train[v1_train.label == 1]
    train_filt = train_pos[train_pos.prot_id.isin(raygun_embs) & train_pos.smiles.isin(fp_dict)]

    moodeng_drug_to_anchors = {}
    for smi, grp in train_filt.groupby("smiles"):
        anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
        if anchors: moodeng_drug_to_anchors[smi] = anchors
    log.info(f"  Anchor pool: {len(moodeng_drug_to_anchors)} drugs")

    # Tanimoto retrieval
    moodeng_fps = [(smi, fp_dict[smi]) for smi in moodeng_drug_to_anchors if smi in fp_dict]
    moodeng_fp_array = np.array([fp for _, fp in moodeng_fps])
    moodeng_smi_array = [smi for smi, _ in moodeng_fps]
    moodeng_fp_gpu = torch.tensor(moodeng_fp_array, dtype=torch.float32, device=device)
    moodeng_bits = moodeng_fp_gpu.sum(1)

    drug_to_anchors = {}
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
            drug_to_anchors[smi] = moodeng_drug_to_anchors[moodeng_smi_array[best_idx[j]]]
    del moodeng_fp_gpu; torch.cuda.empty_cache()
    log.info(f"  Tanimoto anchors: {len(drug_to_anchors)} drugs matched")

    # Oracle anchors
    lci_pos_clean = lci_eval[lci_eval.label == 1.0]
    oracle_drug_to_anchors = {}
    for smi, grp in lci_pos_clean.groupby("smiles"):
        anchors = [pid for pid in grp.prot_id.values if pid in raygun_embs]
        if anchors: oracle_drug_to_anchors[smi] = anchors
    log.info(f"  Oracle anchors: {len(oracle_drug_to_anchors)} drugs")

    del v1_train, train_pos, train_filt
    import gc; gc.collect()

    # ── 6. Load models ──
    log.info("Step 6/8: Loading models...")
    concise_model = torch.hub.load("rohitsinghlab/CoNCISE", "pretrained_concise_v2", pretrained=True)
    concise_model = concise_model.eval().to(device)

    anchor_model = ConciseAnchorBinary().to(device)
    ckpt = torch.load(args.anchor_ckpt, map_location=device, weights_only=False)
    anchor_model.load_state_dict(ckpt["model_state_dict"])
    anchor_model.eval()
    log.info(f"  Models loaded")

    # ── 7. Predict ──
    log.info("Step 7/8: Predictions...")
    pids = lci_eval.prot_id.values
    smis = lci_eval.smiles.values
    labels = lci_eval.label.values.astype(float)

    p_concise = predict_concise(pids, smis, raygun_embs, fp_dict, concise_model, device)
    del concise_model; torch.cuda.empty_cache()

    p_anchor = predict_anchor(pids, smis, raygun_embs, fp_dict,
                              anchor_model, drug_to_anchors, raygun_embs, device)

    log.info("  Oracle predictions...")
    p_oracle = predict_anchor(pids, smis, raygun_embs, fp_dict,
                              anchor_model, oracle_drug_to_anchors, raygun_embs, device)
    del anchor_model; torch.cuda.empty_cache()

    # ── 8. Report ──
    log.info("Step 8/8: Results")
    thresh_pct = int(args.identity_threshold * 100)
    all_methods = {
        "CoNCISE (pretrained)": p_concise,
        "ConciseAnchor": p_anchor,
        "ConciseAnchor-Oracle": p_oracle,
    }

    evaluate_fair(
        f"LCIdb ({thresh_pct}% homology filter, V1+V2 overlap removed)",
        pids, smis, labels, all_methods)

    # Quartile by oracle anchor pKi
    from sklearn.metrics import roc_auc_score
    oracle_anchor_pkis = np.full(len(pids), np.nan)
    for i, (pid, smi) in enumerate(zip(pids, smis)):
        if smi not in oracle_drug_to_anchors: continue
        for a in oracle_drug_to_anchors[smi]:
            if a != pid and a in raygun_embs:
                match = lci_pos_clean[(lci_pos_clean.prot_id == a) & (lci_pos_clean.smiles == smi)]
                if len(match) > 0 and "pki" in match.columns:
                    oracle_anchor_pkis[i] = float(match.iloc[0]["pki"])
                else:
                    oracle_anchor_pkis[i] = 7.0
                break

    has_pki = ~np.isnan(oracle_anchor_pkis)
    if has_pki.sum() > 100:
        print(f"\n{'='*80}")
        print(f"  Quartile by oracle anchor pKi")
        print(f"{'='*80}")
        q_edges = np.quantile(oracle_anchor_pkis[has_pki], [0, 0.25, 0.5, 0.75, 1.0])
        method_names = list(all_methods.keys())
        header = f"  {'Quartile':<20s} {'N':>7s}"
        for m in method_names: header += f"  {m:>20s}"
        print(header)
        print(f"  {'-'*len(header)}")

        for qi in range(4):
            lo, hi = q_edges[qi], q_edges[qi+1]
            mask = has_pki & (oracle_anchor_pkis >= lo) & (oracle_anchor_pkis <= hi if qi == 3 else oracle_anchor_pkis < hi)
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
                    line += f"  {roc_auc_score(lab[v], p[v]):>20.4f}"
            print(line)

        lab = labels[has_pki]
        line = f"  {'Overall':<20s} {has_pki.sum():>7d}"
        for m in method_names:
            p = np.array(all_methods[m])[has_pki]
            v = ~np.isnan(p)
            if v.sum() < 10 or len(set(lab[v].astype(int))) < 2:
                line += f"  {'N/A':>20s}"
            else:
                line += f"  {roc_auc_score(lab[v], p[v]):>20.4f}"
        print(line)

    print(f"\n{'='*80}")
    log.info("All done.")


if __name__ == "__main__":
    main()
