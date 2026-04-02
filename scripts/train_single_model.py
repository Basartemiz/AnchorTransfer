"""Train a single model on a single dataset with 80/10/10 split + test eval.

Usage:
  PYTHONPATH=src python scripts/train_single_model.py \
    --model conplex --dataset dtc --device cuda
"""
import argparse, json, logging, random, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6, "F": 7, "I": 8,
    "H": 9, "K": 10, "M": 11, "L": 12, "O": 13, "N": 14, "Q": 15,
    "P": 16, "S": 17, "R": 18, "U": 19, "T": 20, "W": 21, "V": 22,
    "Y": 23, "X": 24, "Z": 25,
}
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
    "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "[": 50,
    "T": 17, "]": 51, "V": 18, "Y": 19, "c": 20, "e": 21, "l": 22,
    "n": 23, "o": 24, "r": 25, "s": 26, "t": 27, "u": 28,
}

DATASET_PATHS = {
    "dtc": ["data/processed/dtc_training_interactions.csv"],
    "bdb": ["data/processed/bindingdb_interactions.csv", "data/raw/bdb_ki_benchmark.csv"],
    "metz": [
        "data/processed/metz_interactions.csv",
        "data/raw/metz_benchmark.csv",
        "data/raw/metz.csv",
        "data/raw/metz/metz_benchmark.csv",
        "data/raw/metz/metz.csv",
    ],
}
SEQUENCE_PATHS = {
    "dtc": ["data/processed/dtc_sequences.json"],
    "bdb": ["data/processed/dtc_sequences.json", "data/raw/benchmark_proteins.csv"],
    "metz": [
        "data/processed/metz_sequences.json",
        "data/raw/metz_proteins.csv",
        "data/raw/benchmark_proteins.csv",
    ],
}
ESM_PATHS = {
    "dtc": ["data/processed/esm2_35m_dtc_proteins_full.pt", "data/processed/esm2_35m_dtc_proteins.pt"],
    "bdb": [
        "data/processed/esm2_35m_dtc_proteins_full.pt",
        "data/processed/esm2_35m_dtc_proteins.pt",
        "data/processed/esm2_35m_bindingdb.pt",
    ],
    "metz": [
        "data/processed/esm2_35m_metz.pt",
        "data/processed/esm2_35m_metz_proteins.pt",
        "data/processed/esm2_35m_benchmark.pt",
    ],
}


def resolve_default_path(dataset, path_map):
    for candidate in path_map[dataset]:
        if Path(candidate).exists():
            return candidate
    return path_map[dataset][0]


def normalize_interaction_frame(df):
    col_aliases = {
        "uniprot_id": ["uniprot_id", "protein_name", "protein_id", "target_id", "target", "Protein", "Target ID"],
        "ligand_smiles": ["ligand_smiles", "drug_smiles", "smiles", "SMILES", "compound_smiles"],
        "pki": ["pki", "pKd", "pKi", "pkd", "Y", "affinity", "Affinity"],
        "sequence": ["sequence", "target_sequence", "protein_sequence", "Target Sequence", "Protein Sequence"],
        "protein_type": ["protein_type"],
    }
    rename = {}
    for target, aliases in col_aliases.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = target
                break
    df = df.rename(columns=rename)
    missing = [c for c in ("uniprot_id", "ligand_smiles", "pki") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns {missing}. "
            "Expected normalized columns: uniprot_id, ligand_smiles, pki."
        )

    keep = [c for c in ("uniprot_id", "ligand_smiles", "pki", "sequence", "protein_type") if c in df.columns]
    df = df[keep].copy()
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["ligand_smiles"] = df["ligand_smiles"].astype(str).str.strip()
    df["pki"] = pd.to_numeric(df["pki"], errors="coerce")
    if "sequence" in df.columns:
        df["sequence"] = df["sequence"].astype(str).str.strip()
    return df.dropna(subset=["uniprot_id", "ligand_smiles", "pki"])


def load_interactions(path):
    df = pd.read_csv(path, sep=None, engine="python")
    return normalize_interaction_frame(df)


def load_sequences(path):
    if not path:
        return {}
    seq_path = Path(path)
    if not seq_path.exists():
        return {}
    if seq_path.suffix == ".json":
        with seq_path.open() as handle:
            return json.load(handle)

    seq_df = pd.read_csv(seq_path, sep=None, engine="python")
    rename = {}
    if "uniprot_id" not in seq_df.columns:
        for alias in ("protein_name", "protein_id", "target_id", "Protein", "Target ID"):
            if alias in seq_df.columns:
                rename[alias] = "uniprot_id"
                break
    if "sequence" not in seq_df.columns:
        for alias in ("target_sequence", "protein_sequence", "Target Sequence", "Protein Sequence"):
            if alias in seq_df.columns:
                rename[alias] = "sequence"
                break
    seq_df = seq_df.rename(columns=rename)
    if not {"uniprot_id", "sequence"} <= set(seq_df.columns):
        raise ValueError(f"Sequence file {path} must contain uniprot_id and sequence columns")
    seq_df = seq_df[["uniprot_id", "sequence"]].dropna().drop_duplicates("uniprot_id")
    seq_df["uniprot_id"] = seq_df["uniprot_id"].astype(str).str.strip()
    seq_df["sequence"] = seq_df["sequence"].astype(str).str.strip()
    return dict(zip(seq_df["uniprot_id"], seq_df["sequence"]))

def encode_prot(seq, ml=1000):
    return [CHARPROTSET.get(c, 0) for c in seq[:ml]] + [0] * max(0, ml - len(seq))

def encode_smi(smi, ml=100):
    return [CHARISOSMISET.get(c, 0) for c in smi[:ml]] + [0] * max(0, ml - len(smi))

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
        p = F.relu(self.pc1(p)); p = F.relu(self.pc2(p)); p = F.relu(self.pc3(p)); p = p.max(2)[0]
        x = torch.cat([s, p], 1)
        x = self.do(F.relu(self.fc1(x))); x = self.do(F.relu(self.fc2(x)))
        x = self.do(F.relu(self.fc3(x))); return self.out(x).squeeze(-1)


class DTADataset(Dataset):
    def __init__(self, df, esm2, seqs, all_smiles):
        self.uids = df.uniprot_id.values
        self.smiles = df.ligand_smiles.values
        self.pkis = df.pki.values.astype(np.float32)
        self.esm2 = esm2; self.seqs = seqs; self.all_smiles = all_smiles
        self.labels = np.full(len(df), -1, dtype=np.int64)
        self.labels[self.pkis >= 7.0] = 1; self.labels[self.pkis <= 5.0] = 0

    def __len__(self): return len(self.pkis)
    def __getitem__(self, i):
        uid = self.uids[i]; smi = self.smiles[i]
        return {
            "esm2": self.esm2.get(uid, torch.zeros(next(iter(self.esm2.values())).shape[0])),
            "prot": torch.tensor(encode_prot(self.seqs.get(uid, "A"*100)), dtype=torch.long),
            "drug": torch.tensor(encode_smi(smi), dtype=torch.long),
            "neg_drug": torch.tensor(encode_smi(random.choice(self.all_smiles)), dtype=torch.long),
            "pki": self.pkis[i], "label": self.labels[i],
        }


class DrugBANDataset(Dataset):
    def __init__(self, df, seqs):
        try:
            from idr_gat.model.drug_encoder import smiles_to_graph
        except ImportError:
            from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph as smiles_to_graph

        self.smiles_to_graph = smiles_to_graph
        self.uids = df.uniprot_id.values
        self.smiles = df.ligand_smiles.values
        self.pkis = df.pki.values.astype(np.float32)
        self.seqs = seqs
        self.graph_cache = {}

    def __len__(self): return len(self.pkis)

    def _graph(self, smiles):
        graph = self.graph_cache.get(smiles)
        if graph is None:
            graph = self.smiles_to_graph(smiles)
            self.graph_cache[smiles] = graph
        return graph.clone()

    def __getitem__(self, i):
        uid = self.uids[i]
        return {
            "graph": self._graph(self.smiles[i]),
            "prot": torch.tensor(encode_prot(self.seqs[uid]), dtype=torch.long),
            "pki": self.pkis[i],
        }


class AnchorDataset(Dataset):
    def __init__(self, df, esm2, drug_to_anchor, drug_to_second):
        from idr_gat.model.anchor_transfer_v2 import encode_smiles as enc
        self.enc = enc; self.esm2 = esm2
        q, s, p, l, a = [], [], [], [], []
        uids = df.uniprot_id.values; smis = df.ligand_smiles.values
        pkis = df.pki.values.astype(np.float32)
        for i in range(len(df)):
            smi = smis[i]
            if smi not in drug_to_anchor: continue
            anc = drug_to_anchor[smi]; query = uids[i]
            if anc == query:
                anc = drug_to_second.get(smi)
                if not anc or anc not in esm2: continue
            if anc not in esm2: continue
            pki = float(pkis[i])
            lb = 1 if pki >= 7 else (0 if pki <= 5 else -1)
            q.append(query); s.append(smi); p.append(pki); l.append(lb); a.append(anc)
        self.queries, self.smiles, self.pkis, self.labels, self.anchors = q, s, p, l, a
        logger.info("AnchorDataset: %d samples", len(q))

    def __len__(self): return len(self.queries)
    def __getitem__(self, i):
        return {
            "a": self.esm2[self.anchors[i]], "q": self.esm2[self.queries[i]],
            "d": torch.tensor(self.enc(self.smiles[i]), dtype=torch.long),
            "pki": self.pkis[i], "lbl": self.labels[i],
        }


class DrugAnchorDataset(Dataset):
    """Dataset for drug-anchor model: (anchor_drug, query_drug, protein) → pKi.

    For each protein, the anchor drug is the one with the highest pKi.
    If external_mapping is provided, it's used for anchor lookup (for val/test sets
    to use training-derived anchors or self-derived anchors without leakage).
    """
    def __init__(self, df, esm2, prot_to_anchor_drug=None, prot_to_second_drug=None, external_mapping=None):
        from idr_gat.model.anchor_transfer import encode_smiles as enc
        self.enc = enc; self.esm2 = esm2

        # If no external mapping, build from this df
        if prot_to_anchor_drug is None:
            df_v = df[df.uniprot_id.isin(esm2)].copy()
            idx = df_v.groupby("uniprot_id")["pki"].idxmax()
            prot_to_anchor_drug = dict(zip(df_v.loc[idx].uniprot_id, df_v.loc[idx].ligand_smiles))
            prot_to_second_drug = {}
            for uid, grp in df_v.groupby("uniprot_id"):
                v = grp.sort_values("pki", ascending=False)
                if len(v) > 1: prot_to_second_drug[uid] = v.iloc[1]["ligand_smiles"]

        prots, anchor_smis, query_smis, pkis, labels = [], [], [], [], []
        uids = df.uniprot_id.values; smis = df.ligand_smiles.values
        pki_vals = df.pki.values.astype(np.float32)
        for i in range(len(df)):
            uid = uids[i]; smi = smis[i]
            if uid not in esm2: continue
            anc_smi = prot_to_anchor_drug.get(uid)
            if not anc_smi: continue
            if anc_smi == smi:
                anc_smi = (prot_to_second_drug or {}).get(uid)
                if not anc_smi: continue
            pki = float(pki_vals[i])
            lb = 1 if pki >= 7 else (0 if pki <= 5 else -1)
            prots.append(uid); anchor_smis.append(anc_smi); query_smis.append(smi)
            pkis.append(pki); labels.append(lb)
        self.prots = prots; self.anchor_smis = anchor_smis; self.query_smis = query_smis
        self.pkis = pkis; self.labels = labels
        logger.info("DrugAnchorDataset: %d samples", len(prots))

    def __len__(self): return len(self.prots)
    def __getitem__(self, i):
        return {
            "p": self.esm2[self.prots[i]],
            "ad": torch.tensor(self.enc(self.anchor_smis[i]), dtype=torch.long),
            "qd": torch.tensor(self.enc(self.query_smis[i]), dtype=torch.long),
            "pki": self.pkis[i], "lbl": self.labels[i],
        }


def collate_drug_anchor(batch):
    return {
        "p": torch.stack([b["p"] for b in batch]),
        "ad": torch.stack([b["ad"] for b in batch]),
        "qd": torch.stack([b["qd"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
        "lbl": torch.tensor([b["lbl"] for b in batch], dtype=torch.long),
    }


def collate_dta(batch):
    return {k: torch.stack([b[k] for b in batch]) if isinstance(batch[0][k], torch.Tensor)
            else torch.tensor([b[k] for b in batch], dtype=torch.float if k == "pki" else torch.long)
            for k in batch[0]}

def collate_drugban(batch):
    return {
        "graph": Batch.from_data_list([b["graph"] for b in batch]),
        "prot": torch.stack([b["prot"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
    }

def collate_anchor(batch):
    return {
        "a": torch.stack([b["a"] for b in batch]), "q": torch.stack([b["q"] for b in batch]),
        "d": torch.stack([b["d"] for b in batch]),
        "pki": torch.tensor([b["pki"] for b in batch], dtype=torch.float),
        "lbl": torch.tensor([b["lbl"] for b in batch], dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["conplex", "deepdta", "v2", "drug_anchor", "esm_dta", "v2_attn", "drugban"])
    parser.add_argument("--dataset", required=True, choices=["dtc", "bdb", "metz"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--esm-dim", type=int, default=480, help="ESM-2 embedding dim (480 for 35M, 1280 for 650M)")
    parser.add_argument("--esm-path", type=str, default=None, help="Path to ESM-2 embeddings (overrides default)")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to interactions CSV/TSV (overrides dataset default)")
    parser.add_argument("--sequence-path", type=str, default=None, help="Path to protein sequences JSON/CSV (overrides dataset default)")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (overrides default models/{model}_{dataset})")
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    esm_path = args.esm_path or resolve_default_path(args.dataset, ESM_PATHS)
    data_path = args.dataset_path or resolve_default_path(args.dataset, DATASET_PATHS)
    seq_path = args.sequence_path or resolve_default_path(args.dataset, SEQUENCE_PATHS)
    esm2 = torch.load(esm_path, map_location="cpu", weights_only=False)
    esm2 = {k: v for k, v in esm2.items() if not torch.isnan(v).any()}
    seqs = load_sequences(seq_path)
    df = load_interactions(data_path)
    if "sequence" in df.columns:
        seqs.update(
            dict(
                zip(
                    df.loc[df.sequence.notna(), "uniprot_id"],
                    df.loc[df.sequence.notna(), "sequence"],
                )
            )
        )
    df = df[df.uniprot_id.isin(esm2)]
    if args.model in {"deepdta", "drugban"}:
        if not seqs:
            raise ValueError(
                f"{args.model} requires protein sequences. "
                f"Provide --sequence-path for dataset={args.dataset}."
            )
        df = df[df.uniprot_id.isin(seqs)]
    df = df[~df.uniprot_id.str.contains(",", na=False)]
    logger.info(
        "%s: %d interactions, %d proteins (data=%s, esm=%s, seqs=%s)",
        args.dataset.upper(), len(df), df.uniprot_id.nunique(), data_path, esm_path, seq_path,
    )

    # 80/10/10 split
    all_prots = sorted(set(df.uniprot_id) & set(esm2.keys()))
    random.seed(args.seed); random.shuffle(all_prots)
    n_test = max(1, int(len(all_prots) * 0.1))
    n_val = max(1, int(len(all_prots) * 0.1))
    test_prots = set(all_prots[:n_test])
    val_prots = set(all_prots[n_test:n_test + n_val])
    train_prots = set(all_prots[n_test + n_val:])
    train_df = df[df.uniprot_id.isin(train_prots)]
    val_df = df[df.uniprot_id.isin(val_prots)]
    test_df = df[df.uniprot_id.isin(test_prots)]
    logger.info("Split: train=%d (%d), val=%d (%d), test=%d (%d)",
                len(train_df), len(train_prots), len(val_df), len(val_prots), len(test_df), len(test_prots))

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"models/{args.model}_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    all_smiles = train_df.ligand_smiles.unique().tolist()

    # ── TRAIN ────────────────────────────────────────────────────────────────
    if args.model == "deepdta":
        tl = DataLoader(DTADataset(train_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_dta, num_workers=0)
        vl = DataLoader(DTADataset(val_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_dta, num_workers=0)
        model = DeepDTAModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); tl_loss = 0; nb = 0
            for b in tl:
                pred = model(b["drug"].to(device), b["prot"].to(device))
                loss = F.mse_loss(pred, b["pki"].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl_loss += loss.item(); nb += 1
            model.eval(); vl_loss = 0; vnb = 0; ap, at = [], []
            with torch.no_grad():
                for b in vl:
                    pred = model(b["drug"].to(device), b["prot"].to(device))
                    vl_loss += F.mse_loss(pred, b["pki"].to(device)).item(); vnb += 1
                    ap.append(pred.cpu().numpy()); at.append(b["pki"].numpy())
            vl_avg = vl_loss / max(vnb, 1); ap, at = np.concatenate(ap), np.concatenate(at)
            c = ci_fn(at, ap); r = float(np.corrcoef(at, ap)[0, 1]) if len(ap) > 1 else 0
            sched.step(); imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] Train=%.4f Val=%.4f CI=%.4f r=%.4f %s", ep, time.time()-t0, tl_loss/max(nb,1), vl_avg, c, r, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    elif args.model == "drugban":
        from idr_gat.model.drugban import DrugBANModel

        tl = DataLoader(DrugBANDataset(train_df, seqs), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_drugban, num_workers=0)
        vl = DataLoader(DrugBANDataset(val_df, seqs), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_drugban, num_workers=0)
        model = DrugBANModel().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); tl_loss = 0; nb = 0
            for b in tl:
                pred = model(b["graph"].to(device), b["prot"].to(device))
                loss = F.mse_loss(pred, b["pki"].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl_loss += loss.item(); nb += 1
            model.eval(); vl_loss = 0; vnb = 0; ap, at = [], []
            with torch.no_grad():
                for b in vl:
                    pred = model(b["graph"].to(device), b["prot"].to(device))
                    vl_loss += F.mse_loss(pred, b["pki"].to(device)).item(); vnb += 1
                    ap.append(pred.cpu().numpy()); at.append(b["pki"].numpy())
            vl_avg = vl_loss / max(vnb, 1); ap, at = np.concatenate(ap), np.concatenate(at)
            c = ci_fn(at, ap); r = float(np.corrcoef(at, ap)[0, 1]) if len(ap) > 1 else 0
            sched.step(); imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] Train=%.4f Val=%.4f CI=%.4f r=%.4f %s", ep, time.time()-t0, tl_loss/max(nb,1), vl_avg, c, r, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    elif args.model == "conplex":
        from idr_gat.model.conplex import ConPlex
        tl = DataLoader(DTADataset(train_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_dta, num_workers=0)
        vl = DataLoader(DTADataset(val_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_dta, num_workers=0)
        model = ConPlex(esm2_dim=args.esm_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); tl_loss = 0; nb = 0
            phase = "bce" if ep % 2 == 1 else "contrastive"
            for b in tl:
                mask = b["label"] >= 0
                if not mask.any(): continue
                p = b["esm2"][mask].to(device); d = b["drug"][mask].to(device)
                lbl = b["label"][mask].clamp(min=0).float().to(device)
                if phase == "contrastive":
                    out = model.compute_loss(p, d, lbl, phase="contrastive", neg_drug_indices=b["neg_drug"][mask].to(device))
                else:
                    out = model.compute_loss(p, d, lbl, phase="bce")
                opt.zero_grad(); out["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl_loss += out["loss"].item(); nb += 1
            model.eval(); vl_loss = 0; vnb = 0
            with torch.no_grad():
                for b in vl:
                    mask = b["label"] >= 0
                    if not mask.any(): continue
                    out = model.compute_loss(b["esm2"][mask].to(device), b["drug"][mask].to(device),
                                             b["label"][mask].clamp(min=0).float().to(device), phase="bce")
                    vl_loss += out["loss"].item(); vnb += 1
            vl_avg = vl_loss / max(vnb, 1); sched.step()
            imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] %s Train=%.4f Val=%.4f %s", ep, time.time()-t0, phase, tl_loss/max(nb,1), vl_avg, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    elif args.model in ("v2", "v2_attn"):
        if args.model == "v2":
            from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
            model_cls = AnchorTransferDTAv2
        else:
            from idr_gat.model.anchor_transfer_attn import AnchorTransferAttn
            model_cls = AnchorTransferAttn
        dtc_v = train_df[train_df.uniprot_id.isin(esm2)].copy()
        idx = dtc_v.groupby("ligand_smiles")["pki"].idxmax()
        drug_strongest = dict(zip(dtc_v.loc[idx].ligand_smiles, dtc_v.loc[idx].uniprot_id))
        drug_second = {}
        for smi, grp in dtc_v.groupby("ligand_smiles"):
            v = grp.sort_values("pki", ascending=False)
            if len(v) > 1: drug_second[smi] = v.iloc[1]["uniprot_id"]

        tl = DataLoader(AnchorDataset(train_df, esm2, drug_strongest, drug_second),
                        batch_size=args.batch_size, shuffle=True, collate_fn=collate_anchor, num_workers=0)
        vl = DataLoader(AnchorDataset(val_df, esm2, drug_strongest, drug_second),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate_anchor, num_workers=0)
        model = model_cls(esm2_dim=args.esm_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); trl, trb, trm, nb = 0, 0, 0, 0
            for b in tl:
                mask = b["lbl"] >= 0; lbl = b["lbl"].clamp(min=0)
                o = model.compute_loss(b["a"].to(device), b["q"].to(device), b["d"].to(device),
                                       b["pki"].to(device), lbl.to(device) if mask.any() else None, mask.to(device) if mask.any() else None)
                opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                trl += o["loss"].item(); trb += o["bce_loss"].item(); trm += o["mse_loss"].item(); nb += 1
            model.eval(); vlo, vnb = 0, 0; ap, at = [], []
            with torch.no_grad():
                for b in vl:
                    mask = b["lbl"] >= 0; lbl = b["lbl"].clamp(min=0)
                    o = model.compute_loss(b["a"].to(device), b["q"].to(device), b["d"].to(device),
                                           b["pki"].to(device), lbl.to(device) if mask.any() else None, mask.to(device) if mask.any() else None)
                    vlo += o["loss"].item(); vnb += 1
                    ap.append(o["pki_pred"].cpu().numpy()); at.append(b["pki"].numpy())
            vl_avg = vlo / max(vnb, 1); ap, at = np.concatenate(ap), np.concatenate(at)
            c = ci_fn(at, ap); r = float(np.corrcoef(at, ap)[0, 1]) if len(ap) > 1 else 0
            sched.step(); imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] Train=%.4f (bce=%.4f mse=%.4f) Val=%.4f CI=%.4f r=%.4f %s",
                        ep, time.time()-t0, trl/max(nb,1), trb/max(nb,1), trm/max(nb,1), vl_avg, c, r, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    elif args.model == "drug_anchor":
        from idr_gat.model.drug_anchor_dta import DrugAnchorDTA
        # Build protein→strongest_drug mapping from training set
        train_v = train_df[train_df.uniprot_id.isin(esm2)].copy()
        idx = train_v.groupby("uniprot_id")["pki"].idxmax()
        prot_strongest_drug = dict(zip(train_v.loc[idx].uniprot_id, train_v.loc[idx].ligand_smiles))
        prot_second_drug = {}
        for uid, grp in train_v.groupby("uniprot_id"):
            v = grp.sort_values("pki", ascending=False)
            if len(v) > 1: prot_second_drug[uid] = v.iloc[1]["ligand_smiles"]

        tl = DataLoader(DrugAnchorDataset(train_df, esm2, prot_strongest_drug, prot_second_drug),
                        batch_size=args.batch_size, shuffle=True, collate_fn=collate_drug_anchor, num_workers=0)
        # Val dataset builds its own per-protein anchor from val data (no leakage)
        vl = DataLoader(DrugAnchorDataset(val_df, esm2),
                        batch_size=args.batch_size, shuffle=False, collate_fn=collate_drug_anchor, num_workers=0)
        model = DrugAnchorDTA(esm2_dim=args.esm_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); trl, trb, trm, nb = 0, 0, 0, 0
            for b in tl:
                mask = b["lbl"] >= 0; lbl = b["lbl"].clamp(min=0)
                o = model.compute_loss(b["ad"].to(device), b["qd"].to(device), b["p"].to(device),
                                       b["pki"].to(device), lbl.to(device) if mask.any() else None, mask.to(device) if mask.any() else None)
                opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                trl += o["loss"].item(); trb += o["bce_loss"].item(); trm += o["mse_loss"].item(); nb += 1
            model.eval(); vlo, vnb = 0, 0; ap, at = [], []
            with torch.no_grad():
                for b in vl:
                    mask = b["lbl"] >= 0; lbl = b["lbl"].clamp(min=0)
                    o = model.compute_loss(b["ad"].to(device), b["qd"].to(device), b["p"].to(device),
                                           b["pki"].to(device), lbl.to(device) if mask.any() else None, mask.to(device) if mask.any() else None)
                    vlo += o["loss"].item(); vnb += 1
                    ap.append(o["pki_pred"].cpu().numpy()); at.append(b["pki"].numpy())
            vl_avg = vlo / max(vnb, 1); ap, at = np.concatenate(ap), np.concatenate(at)
            c = ci_fn(at, ap); r = float(np.corrcoef(at, ap)[0, 1]) if len(ap) > 1 else 0
            sched.step(); imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] Train=%.4f (bce=%.4f mse=%.4f) Val=%.4f CI=%.4f r=%.4f %s",
                        ep, time.time()-t0, trl/max(nb,1), trb/max(nb,1), trm/max(nb,1), vl_avg, c, r, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    elif args.model == "esm_dta":
        from idr_gat.model.esm_dta import EsmDTAModel
        tl = DataLoader(DTADataset(train_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=True, collate_fn=collate_dta, num_workers=0)
        vl = DataLoader(DTADataset(val_df, esm2, seqs, all_smiles), batch_size=args.batch_size,
                        shuffle=False, collate_fn=collate_dta, num_workers=0)
        model = EsmDTAModel(esm2_dim=args.esm_dim).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        best = float("inf"); pat = 0
        for ep in range(1, args.epochs + 1):
            t0 = time.time(); model.train(); tl_loss = 0; nb = 0
            for b in tl:
                pred = model(b["drug"].to(device), b["esm2"].to(device))
                loss = F.mse_loss(pred, b["pki"].to(device))
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
                tl_loss += loss.item(); nb += 1
            model.eval(); vl_loss = 0; vnb = 0; ap, at = [], []
            with torch.no_grad():
                for b in vl:
                    pred = model(b["drug"].to(device), b["esm2"].to(device))
                    vl_loss += F.mse_loss(pred, b["pki"].to(device)).item(); vnb += 1
                    ap.append(pred.cpu().numpy()); at.append(b["pki"].numpy())
            vl_avg = vl_loss / max(vnb, 1); ap, at = np.concatenate(ap), np.concatenate(at)
            c = ci_fn(at, ap); r = float(np.corrcoef(at, ap)[0, 1]) if len(ap) > 1 else 0
            sched.step(); imp = vl_avg < best
            if imp: best = vl_avg; pat = 0; torch.save({"epoch": ep, "model_state_dict": model.state_dict()}, out_dir / "best_model.pt")
            else: pat += 1
            logger.info("Ep %3d [%.0fs] Train=%.4f Val=%.4f CI=%.4f r=%.4f %s", ep, time.time()-t0, tl_loss/max(nb,1), vl_avg, c, r, "*" if imp else f"p={pat}")
            if pat >= args.patience: logger.info("Early stopping"); break

    # ── TEST EVAL ────────────────────────────────────────────────────────────
    logger.info("=== TEST SET EVALUATION ===")
    # Reload best model
    ckpt = torch.load(out_dir / "best_model.pt", map_location=device, weights_only=False)
    if args.model == "deepdta":
        model = DeepDTAModel().to(device)
    elif args.model == "drugban":
        from idr_gat.model.drugban import DrugBANModel
        model = DrugBANModel().to(device)
    elif args.model == "conplex":
        from idr_gat.model.conplex import ConPlex
        model = ConPlex(esm2_dim=args.esm_dim).to(device)
    elif args.model == "v2":
        from idr_gat.model.anchor_transfer_v2 import AnchorTransferDTAv2
        model = AnchorTransferDTAv2(esm2_dim=args.esm_dim).to(device)
    elif args.model == "v2_attn":
        from idr_gat.model.anchor_transfer_attn import AnchorTransferAttn
        model = AnchorTransferAttn(esm2_dim=args.esm_dim).to(device)
    elif args.model == "drug_anchor":
        from idr_gat.model.drug_anchor_dta import DrugAnchorDTA
        model = DrugAnchorDTA(esm2_dim=args.esm_dim).to(device)
    elif args.model == "esm_dta":
        from idr_gat.model.esm_dta import EsmDTAModel
        model = EsmDTAModel(esm2_dim=args.esm_dim).to(device)
    model.load_state_dict(ckpt["model_state_dict"]); model.eval()

    # Batched test eval
    test_preds_all, test_trues_all = [], []
    per_protein = []
    for uid, grp in test_df.groupby("uniprot_id"):
        if uid not in esm2: continue
        smis = grp.ligand_smiles.values; pkis = grp.pki.values
        preds = []

        if args.model == "deepdta":
            if uid not in seqs: continue
            seq = seqs[uid]
            pe = torch.tensor([encode_prot(seq)], dtype=torch.long, device=device)
            for smi in smis:
                se = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): preds.append(model(se, pe).item())

        elif args.model == "drugban":
            try:
                from idr_gat.model.drug_encoder import smiles_to_graph
            except ImportError:
                from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph as smiles_to_graph

            if uid not in seqs: continue
            prot_batch = torch.tensor([encode_prot(seqs[uid]) for _ in smis], dtype=torch.long, device=device)
            drug_batch = Batch.from_data_list([smiles_to_graph(smi) for smi in smis]).to(device)
            with torch.no_grad():
                preds = model(drug_batch, prot_batch).cpu().numpy().tolist()

        elif args.model == "conplex":
            p = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): out = model(p, d); preds.append(out["score"].item())

        elif args.model in ("v2", "v2_attn"):
            from idr_gat.model.anchor_transfer_v2 import encode_smiles as enc_v2
            q = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                anchor = None
                if smi in drug_strongest:
                    a = drug_strongest[smi]
                    if a != uid and a in esm2: anchor = a
                if not anchor: anchor = uid
                at = esm2[anchor].unsqueeze(0).to(device)
                dt = torch.tensor([enc_v2(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): out = model(at, q, dt); preds.append(out["pki_pred"].item())

        elif args.model == "drug_anchor":
            from idr_gat.model.anchor_transfer import encode_smiles as enc_da
            p = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                anc_smi = prot_strongest_drug.get(uid)
                if not anc_smi or anc_smi == smi:
                    anc_smi = prot_second_drug.get(uid)
                if not anc_smi: anc_smi = smi  # self-anchor fallback
                ad = torch.tensor([enc_da(anc_smi)], dtype=torch.long, device=device)
                qd = torch.tensor([enc_da(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): out = model(ad, qd, p); preds.append(out["pki_pred"].item())

        elif args.model == "esm_dta":
            p = esm2[uid].unsqueeze(0).to(device)
            for smi in smis:
                d = torch.tensor([encode_smi(smi)], dtype=torch.long, device=device)
                with torch.no_grad(): preds.append(model(d, p).item())

        if not preds: continue
        preds = np.array(preds)
        c = ci_fn(pkis, preds)
        r = float(np.corrcoef(pkis, preds)[0, 1]) if len(preds) > 1 and np.std(preds) > 0 else 0
        labels = (pkis >= 7.0).astype(int)
        auroc = roc_auc_score(labels, preds) if len(set(labels)) == 2 else float("nan")
        per_protein.append({"uid": uid, "ci": c, "r": r, "auroc": auroc, "n": len(preds)})
        test_preds_all.extend(preds); test_trues_all.extend(pkis)

    rdf = pd.DataFrame(per_protein)
    na = int(rdf.auroc.notna().sum())
    logger.info("TEST %s-%s (n=%d, auroc_valid=%d): CI=%.3f r=%.3f AUROC=%.3f",
                args.model.upper(), args.dataset.upper(), len(rdf), na, rdf.ci.mean(), rdf.r.mean(), rdf.auroc.mean())

    # Global CI
    if test_preds_all:
        g_ci = ci_fn(np.array(test_trues_all), np.array(test_preds_all))
        g_r = float(np.corrcoef(test_trues_all, test_preds_all)[0, 1])
        logger.info("TEST GLOBAL: CI=%.3f r=%.3f", g_ci, g_r)

    res_dir = Path(f"results/6model/{args.model}_{args.dataset}")
    res_dir.mkdir(parents=True, exist_ok=True)
    rdf.to_csv(res_dir / "test_results.csv", index=False)
    logger.info("Done")


if __name__ == "__main__":
    main()
