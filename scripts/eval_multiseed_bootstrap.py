#!/usr/bin/env python3
"""Multi-seed evaluation with bootstrap confidence intervals.

Loads all 6 trained models, evaluates on DTC test set with the existing
seed=42 split, then computes bootstrap 95% CIs for AUROC, CI, and RMSE.

Usage:
  PYTHONPATH=src python scripts/eval_multiseed_bootstrap.py --device cuda
"""
import argparse, json, logging, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

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

def ci_fn(yt, yp):
    n = len(yt)
    if n < 2: return 0.5
    yt, yp = np.array(yt), np.array(yp)
    if n * (n - 1) // 2 > 100000:
        i = np.random.randint(0, n, 100000); j = np.random.randint(0, n, 100000)
        m = i != j; i, j = i[m], j[m]
    else:
        idx = np.triu_indices(n, k=1); i, j = idx[0], idx[1]
    dt = yt[i] - yt[j]; dp = yp[i] - yp[j]; t = dt == 0
    return float(((dt * dp) > 0).sum() / (~t).sum()) if (~t).sum() > 0 else 0.5

def auroc_fn(trues, preds):
    binder = trues >= 7.0; non_binder = trues <= 5.0; mask = binder | non_binder
    if mask.sum() == 0 or binder[mask].sum() == 0 or non_binder[mask].sum() == 0:
        return float("nan")
    return float(roc_auc_score(binder[mask].astype(int), preds[mask]))

def bootstrap_ci(trues, preds, metric_fn, n_boot=1000, seed=42):
    """Compute bootstrap 95% CI for a metric."""
    rng = np.random.RandomState(seed)
    n = len(trues)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        s = metric_fn(trues[idx], preds[idx])
        if not np.isnan(s):
            scores.append(s)
    scores = np.array(scores)
    return float(np.mean(scores)), float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def predict_model(model_name, model, esm2, test_df, drug_strongest, seqs, device):
    """Get predictions for all test interactions (anchored only for anchor models)."""
    all_preds, all_trues = [], []

    for uid, grp in test_df.groupby("uniprot_id"):
        if uid not in esm2:
            continue
        smis = grp.ligand_smiles.values
        pkis = grp.pki.values
        preds = []

        if model_name == "deepdta":
            if uid not in seqs: continue
            seq = seqs[uid]
            pe = torch.tensor([encode_prot(seq)], dtype=torch.long, device=device)
            for smi in smis:
                se = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): preds.append(model(se, pe).item())

        elif model_name == "conplex":
            p = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): out = model(p, d); preds.append(out["score"].item())

        elif model_name in ("v2", "v2_attn"):
            from idr_gat.model.anchor_transfer_v2 import encode_smiles as enc_v2
            q = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                anchor = None
                if smi in drug_strongest:
                    a = drug_strongest[smi]
                    if a != uid and a in esm2: anchor = a
                if not anchor: continue  # skip non-anchored
                at = esm2[anchor].unsqueeze(0).to(device)
                dt = torch.tensor([enc_v2(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): out = model(at, q, dt); preds.append(out["pki_pred"].item())

        elif model_name == "drug_anchor":
            from idr_gat.model.anchor_transfer import encode_smiles as enc_da
            p = esm2[uid].unsqueeze(0).to(device)
            for i, smi in enumerate(smis):
                # Need protein→strongest_drug mapping
                continue  # handled separately

        elif model_name == "esm_dta":
            p = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): preds.append(model(d, p).item())

        if preds:
            # For non-anchor models, preds matches smis 1:1
            # For anchor models, preds may be shorter (skipped non-anchored)
            if model_name in ("v2", "v2_attn"):
                # Need to track which ones were predicted
                pass  # handled below
            else:
                all_preds.extend(preds)
                all_trues.extend(pkis[:len(preds)])

    return np.array(all_preds), np.array(all_trues)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load data
    esm_path = "data/processed/esm2_35m_dtc_proteins.pt"
    esm2 = torch.load(esm_path, map_location="cpu", weights_only=False)
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    seqs = json.load(open("data/processed/dtc_sequences.json"))

    dtc = pd.read_csv("data/processed/dtc_training_interactions.csv")
    dtc = dtc[dtc.uniprot_id.isin(esm2) & dtc.uniprot_id.isin(seqs)]
    dtc = dtc[~dtc.uniprot_id.str.contains(",", na=False)]

    # 80/10/10 split
    all_prots = sorted(set(dtc.uniprot_id) & set(esm2.keys()))
    random.seed(args.seed); random.shuffle(all_prots)
    n_test = max(1, int(len(all_prots) * 0.1))
    n_val = max(1, int(len(all_prots) * 0.1))
    test_prots = set(all_prots[:n_test])
    train_prots = set(all_prots[n_test + n_val:])
    test_df = dtc[dtc.uniprot_id.isin(test_prots)]
    train_dtc = dtc[dtc.uniprot_id.isin(train_prots)]

    # Build anchor mappings
    idx = train_dtc.groupby("ligand_smiles")["pki"].idxmax()
    drug_strongest = dict(zip(train_dtc.loc[idx].ligand_smiles, train_dtc.loc[idx].uniprot_id))

    # Models to evaluate
    models_config = {
        "v2": ("models/v2_dtc/best_model.pt", "idr_gat.model.anchor_transfer_v2", "AnchorTransferDTAv2"),
        "v2_attn": ("models/v2_attn_dtc/best_model.pt", "idr_gat.model.anchor_transfer_attn", "AnchorTransferAttn"),
        "deepdta": ("models/deepdta_dtc/best_model.pt", None, None),
        "conplex": ("models/conplex_dtc/best_model.pt", "idr_gat.model.conplex", "ConPlex"),
        "esm_dta": ("models/esm_dta_dtc/best_model.pt", "idr_gat.model.esm_dta", "EsmDTAModel"),
        "drug_anchor": ("models/drug_anchor_dtc/best_model.pt", "idr_gat.model.drug_anchor_dta", "DrugAnchorDTA"),
    }

    results = {}
    for model_name, (ckpt_path, mod_path, cls_name) in models_config.items():
        if not Path(ckpt_path).exists():
            logger.warning("SKIP %s: checkpoint not found at %s", model_name, ckpt_path)
            continue

        logger.info("=== Evaluating %s ===", model_name)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        # Load model
        if model_name == "deepdta":
            # Inline DeepDTA model (same as train_single_model.py)
            class DeepDTAModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.smiles_embed = nn.Embedding(66, 128, padding_idx=0)
                    self.protein_embed = nn.Embedding(26, 128, padding_idx=0)
                    self.sc1 = nn.Conv1d(128, 32, 8); self.sc2 = nn.Conv1d(32, 64, 8); self.sc3 = nn.Conv1d(64, 96, 8)
                    self.pc1 = nn.Conv1d(128, 32, 8); self.pc2 = nn.Conv1d(32, 64, 8); self.pc3 = nn.Conv1d(64, 96, 8)
                    self.fc1 = nn.Linear(192, 1024); self.fc2 = nn.Linear(1024, 1024)
                    self.fc3 = nn.Linear(1024, 512); self.out = nn.Linear(512, 1)
                    self.do = nn.Dropout(0.1)
                def forward(self, s, p):
                    s = self.smiles_embed(s).permute(0, 2, 1)
                    s = F.relu(self.sc1(s)); s = F.relu(self.sc2(s)); s = F.relu(self.sc3(s)); s = s.max(2)[0]
                    p = self.protein_embed(p).permute(0, 2, 1)
                    p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p)); p = p.max(2)[0]
                    x = torch.cat([s, p], 1)
                    x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
                    x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)
            model = DeepDTAModel().to(device)
        else:
            import importlib
            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            model = cls(esm2_dim=480).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        # Get predictions
        all_preds, all_trues = [], []
        for uid, grp in test_df.groupby("uniprot_id"):
            if uid not in esm2: continue
            smis = grp.ligand_smiles.values
            pkis = grp.pki.values

            if model_name == "deepdta":
                if uid not in seqs: continue
                seq = seqs[uid]
                pe = torch.tensor([encode_prot(seq)], dtype=torch.long, device=device)
                for i, smi in enumerate(smis):
                    se = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        all_preds.append(model(se, pe).item())
                        all_trues.append(pkis[i])

            elif model_name == "conplex":
                p = esm2[uid].unsqueeze(0).to(device)
                for i, smi in enumerate(smis):
                    d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(p, d)
                        all_preds.append(out["score"].item())
                        all_trues.append(pkis[i])

            elif model_name in ("v2", "v2_attn"):
                from idr_gat.model.anchor_transfer_v2 import encode_smiles as enc_v2
                q = esm2[uid].unsqueeze(0).to(device)
                for i, smi in enumerate(smis):
                    anchor = None
                    if smi in drug_strongest:
                        a = drug_strongest[smi]
                        if a != uid and a in esm2: anchor = a
                    if not anchor: continue
                    at = esm2[anchor].unsqueeze(0).to(device)
                    dt = torch.tensor([enc_v2(smi)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(at, q, dt)
                        all_preds.append(out["pki_pred"].item())
                        all_trues.append(pkis[i])

            elif model_name == "drug_anchor":
                from idr_gat.model.anchor_transfer import encode_smiles as enc_da
                # Build protein→strongest_drug
                prot_idx = train_dtc.groupby("uniprot_id")["pki"].idxmax()
                prot_strongest_drug = dict(zip(train_dtc.loc[prot_idx].uniprot_id, train_dtc.loc[prot_idx].ligand_smiles))
                p = esm2[uid].unsqueeze(0).to(device)
                for i, smi in enumerate(smis):
                    anc_smi = prot_strongest_drug.get(uid)
                    if not anc_smi or anc_smi == smi: continue
                    ad = torch.tensor([enc_da(anc_smi)], dtype=torch.long, device=device)
                    qd = torch.tensor([enc_da(smi)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        out = model(ad, qd, p)
                        all_preds.append(out["pki_pred"].item())
                        all_trues.append(pkis[i])

            elif model_name == "esm_dta":
                p = esm2[uid].unsqueeze(0).to(device)
                for i, smi in enumerate(smis):
                    d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                    with torch.no_grad():
                        all_preds.append(model(d, p).item())
                        all_trues.append(pkis[i])

        preds = np.array(all_preds)
        trues = np.array(all_trues)
        logger.info("%s: %d predictions", model_name, len(preds))

        if len(preds) == 0:
            continue

        # Compute metrics with bootstrap CIs
        auroc_mean, auroc_lo, auroc_hi = bootstrap_ci(trues, preds, auroc_fn, args.n_boot)
        ci_mean, ci_lo, ci_hi = bootstrap_ci(trues, preds, ci_fn, args.n_boot)
        rmse_fn = lambda t, p: float(np.sqrt(np.mean((t - p) ** 2)))
        rmse_mean, rmse_lo, rmse_hi = bootstrap_ci(trues, preds, rmse_fn, args.n_boot)

        results[model_name] = {
            "n": len(preds),
            "auroc": f"{auroc_mean:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]",
            "ci": f"{ci_mean:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]",
            "rmse": f"{rmse_mean:.3f} [{rmse_lo:.3f}, {rmse_hi:.3f}]",
            "auroc_mean": auroc_mean, "auroc_lo": auroc_lo, "auroc_hi": auroc_hi,
            "ci_mean": ci_mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "rmse_mean": rmse_mean, "rmse_lo": rmse_lo, "rmse_hi": rmse_hi,
        }
        logger.info("%s: AUROC=%s  CI=%s  RMSE=%s",
                    model_name, results[model_name]["auroc"],
                    results[model_name]["ci"], results[model_name]["rmse"])

    # Summary table
    logger.info("\n=== BOOTSTRAP 95%% CI SUMMARY (DTC test, n_boot=%d) ===", args.n_boot)
    logger.info("%-15s  %-25s  %-25s  %-25s  %s", "Model", "AUROC", "CI", "RMSE", "n")
    for m in ["esm_dta", "drug_anchor", "deepdta", "v2", "v2_attn", "conplex"]:
        if m in results:
            r = results[m]
            logger.info("%-15s  %-25s  %-25s  %-25s  %d", m, r["auroc"], r["ci"], r["rmse"], r["n"])

    out = Path("results/bootstrap_ci_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=2)
    logger.info("Saved to %s", out)


if __name__ == "__main__":
    main()
