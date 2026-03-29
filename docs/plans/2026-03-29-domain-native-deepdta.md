# Domain-Native DeepDTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a DeepDTA from scratch on domain sequences (not full proteins) so anchor-based DTA predictions are native to the domain input distribution.

**Architecture:** Same DeepDTA CNN with PROTEIN_MAX_LEN=500, smaller kernels [3,4,6]. Data pipeline identifies binding domains via UniProt annotations (fallback: largest domain). Training on BindingDB pairs mapped to domain sequences. Inference uses single best anchor domain.

**Tech Stack:** PyTorch, existing `deepdta_model.py`/`deepdta_encoding.py`, UniProt REST API, Foldseek, existing eval infrastructure.

---

### Task 1: Binding Domain Identification Module

**Files:**
- Create: `src/idr_gat/data/binding_domain.py`
- Test: `tests/test_binding_domain.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_binding_domain.py
"""Tests for binding domain identification."""
import pytest


class TestDomainSequenceMapping:
    def test_find_domain_in_full_sequence(self):
        from idr_gat.data.binding_domain import find_domain_range
        full_seq = "AAAAGGGGGCCCCCTTTTTT"
        domain_seq = "GGGGG"
        start, end = find_domain_range(domain_seq, full_seq)
        assert start == 4
        assert end == 9

    def test_domain_not_found_returns_none(self):
        from idr_gat.data.binding_domain import find_domain_range
        result = find_domain_range("WWWWW", "AAAAA")
        assert result is None


class TestUniProtBindingSites:
    def test_parse_binding_features(self):
        from idr_gat.data.binding_domain import parse_uniprot_binding_sites
        # Minimal UniProt JSON features structure
        features = [
            {"type": "Binding site", "location": {"start": {"value": 100}, "end": {"value": 105}}},
            {"type": "Active site", "location": {"start": {"value": 200}, "end": {"value": 200}}},
            {"type": "Helix", "location": {"start": {"value": 50}, "end": {"value": 80}}},
        ]
        sites = parse_uniprot_binding_sites(features)
        assert len(sites) == 2
        assert sites[0] == (100, 105)
        assert sites[1] == (200, 200)


class TestSelectBindingDomain:
    def test_selects_domain_overlapping_binding_site(self):
        from idr_gat.data.binding_domain import select_binding_domain
        domains = {
            "P_d0": {"seq": "AAAA", "range": (1, 4), "length": 4},
            "P_d1": {"seq": "BBBB", "range": (90, 110), "length": 21},
        }
        binding_sites = [(95, 105)]
        best = select_binding_domain(domains, binding_sites)
        assert best == "P_d1"

    def test_falls_back_to_largest_domain(self):
        from idr_gat.data.binding_domain import select_binding_domain
        domains = {
            "P_d0": {"seq": "AAAA", "range": (1, 4), "length": 4},
            "P_d1": {"seq": "B" * 200, "range": (10, 210), "length": 200},
        }
        binding_sites = []  # no annotations
        best = select_binding_domain(domains, binding_sites)
        assert best == "P_d1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/IDP-work && PYTHONPATH=src:. python -m pytest tests/test_binding_domain.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Implement binding domain module**

```python
# src/idr_gat/data/binding_domain.py
"""Identify which domain of a protein is the drug-binding domain.

Priority: UniProt binding site annotations → largest domain fallback.
"""
from __future__ import annotations

import logging
import time
from typing import Optional
import urllib.request
import json

logger = logging.getLogger(__name__)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def pdb_to_seq(pdb_path: str) -> str:
    """Extract amino acid sequence from domain PDB CA atoms."""
    seq = []
    seen = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    resnum = int(line[22:26].strip())
                    if resnum not in seen:
                        seen.add(resnum)
                        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    except (FileNotFoundError, IOError):
        pass
    return "".join(seq)


def find_domain_range(domain_seq: str, full_seq: str) -> Optional[tuple[int, int]]:
    """Find where a domain sequence maps in the full protein sequence.

    Returns (start, end) 0-indexed, or None if not found.
    """
    idx = full_seq.find(domain_seq)
    if idx >= 0:
        return (idx, idx + len(domain_seq) - 1)
    # Approximate match (allow up to 3 mismatches)
    for start in range(len(full_seq) - len(domain_seq) + 1):
        sub = full_seq[start : start + len(domain_seq)]
        mismatches = sum(1 for a, b in zip(sub, domain_seq) if a != b)
        if mismatches <= 3:
            return (start, start + len(domain_seq) - 1)
    return None


def parse_uniprot_binding_sites(features: list[dict]) -> list[tuple[int, int]]:
    """Extract binding/active site residue ranges from UniProt feature list."""
    binding_types = {"Binding site", "Active site"}
    sites = []
    for feat in features:
        if feat.get("type") in binding_types:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start is not None and end is not None:
                sites.append((int(start), int(end)))
    return sites


def fetch_uniprot_binding_sites(uniprot_id: str) -> list[tuple[int, int]]:
    """Fetch binding site annotations from UniProt REST API.

    Returns list of (start, end) 1-indexed residue ranges.
    """
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        features = data.get("features", [])
        return parse_uniprot_binding_sites(features)
    except Exception as e:
        logger.debug("UniProt fetch failed for %s: %s", uniprot_id, e)
        return []


def select_binding_domain(
    domains: dict[str, dict],
    binding_sites: list[tuple[int, int]],
) -> str:
    """Select the binding domain from a dict of domain info.

    Args:
        domains: {domain_name: {"seq": str, "range": (start, end), "length": int}}
        binding_sites: [(start, end)] 1-indexed residue ranges from UniProt

    Returns:
        domain_name of the best binding domain.
    """
    if binding_sites:
        # Score each domain by overlap with binding sites
        best_domain = None
        best_overlap = -1
        for name, info in domains.items():
            d_start, d_end = info["range"]
            overlap = 0
            for bs_start, bs_end in binding_sites:
                # Convert 1-indexed UniProt to 0-indexed
                bs_s = bs_start - 1
                bs_e = bs_end - 1
                ov_start = max(d_start, bs_s)
                ov_end = min(d_end, bs_e)
                if ov_end >= ov_start:
                    overlap += ov_end - ov_start + 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = name
        if best_domain and best_overlap > 0:
            return best_domain

    # Fallback: largest domain
    return max(domains, key=lambda k: domains[k]["length"])


def identify_binding_domain(
    uniprot_id: str,
    domain_pdbs: dict[str, str],
    full_sequence: str,
) -> tuple[str, str, str]:
    """Identify the binding domain for a protein.

    Args:
        uniprot_id: UniProt accession
        domain_pdbs: {domain_name: pdb_path}
        full_sequence: full protein amino acid sequence

    Returns:
        (domain_name, domain_sequence, method) where method is "uniprot" or "largest"
    """
    # Build domain info
    domains = {}
    for name, pdb_path in domain_pdbs.items():
        dom_seq = pdb_to_seq(pdb_path)
        if not dom_seq:
            continue
        rng = find_domain_range(dom_seq, full_sequence)
        if rng is None:
            rng = (0, len(dom_seq) - 1)
        domains[name] = {"seq": dom_seq, "range": rng, "length": len(dom_seq)}

    if not domains:
        return ("", "", "none")

    # Try UniProt annotations
    binding_sites = fetch_uniprot_binding_sites(uniprot_id)
    method = "uniprot" if binding_sites else "largest"

    best = select_binding_domain(domains, binding_sites)
    return (best, domains[best]["seq"], method)
```

- [ ] **Step 4: Run tests**

Run: `cd /workspace/IDP-work && PYTHONPATH=src:. python -m pytest tests/test_binding_domain.py -v`
Expected: All 4 pass

- [ ] **Step 5: Commit**

```bash
git add src/idr_gat/data/binding_domain.py tests/test_binding_domain.py
git commit -m "feat: binding domain identification (UniProt + largest fallback)"
```

---

### Task 2: Build Domain Training Data

**Files:**
- Create: `scripts/build_domain_training_data.py`

- [ ] **Step 1: Write the data pipeline script**

```python
#!/usr/bin/env python3
"""Build domain-based training data for domain-native DeepDTA.

For each protein in benchmark_affinity.csv (training split only):
1. Identify the binding domain (UniProt annotations or largest)
2. Extract domain amino acid sequence from PDB
3. Output CSV: domain_sequence, ligand_smiles, pki

Usage:
    python scripts/build_domain_training_data.py \
        --benchmark data/raw/benchmark_affinity.csv \
        --domain-dir data/processed/model_organism_domains \
        --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
        --sequences data/processed/merged_sequences.json \
        --holdout-proteins data/raw/benchmark_affinity.csv \
        --output data/processed/domain_training_data.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions", required=True,
                        help="Training interactions CSV (uniprot_id, ligand_smiles, pki)")
    parser.add_argument("--domain-dir", required=True)
    parser.add_argument("--domain-metadata", required=True)
    parser.add_argument("--sequences", required=True)
    parser.add_argument("--holdout-proteins", required=True,
                        help="Benchmark CSV — these proteins excluded from training")
    parser.add_argument("--output", default="data/processed/domain_training_data.csv")
    parser.add_argument("--uniprot-cache", default="data/processed/uniprot_binding_cache.json",
                        help="Cache for UniProt API results")
    args = parser.parse_args()

    from idr_gat.data.binding_domain import identify_binding_domain

    # Load holdout proteins
    holdout_df = pd.read_csv(args.holdout_proteins)
    holdout_uids = set(holdout_df["uniprot_id"].unique())
    logger.info("Holdout: %d proteins", len(holdout_uids))

    # Load sequences
    with open(args.sequences) as f:
        sequences = json.load(f)
    with open(args.domain_metadata) as f:
        meta = json.load(f)
    protein_seqs = meta.get("protein_sequences", {})
    for k, v in sequences.items():
        if k not in protein_seqs:
            protein_seqs[k] = v
    domain_info = meta.get("domain_info", {})

    # Load interactions
    df = pd.read_csv(args.interactions)
    col_map = {"protein_id": "uniprot_id", "target_id": "uniprot_id",
               "canonical_smiles": "ligand_smiles", "smiles": "ligand_smiles"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "pki" not in df.columns and "binding_affinity" in df.columns:
        ba = pd.to_numeric(df["binding_affinity"], errors="coerce")
        df["pki"] = 9.0 - np.log10(ba)

    df = df.dropna(subset=["uniprot_id", "ligand_smiles", "pki"])
    df = df[(df["pki"] >= 3.0) & (df["pki"] <= 12.0)]
    df["uniprot_id"] = df["uniprot_id"].astype(str).str.strip()
    df["uniprot_id"] = df["uniprot_id"].str.replace(r"^(af_|pf_)", "", regex=True)

    # Exclude holdout
    df = df[~df["uniprot_id"].isin(holdout_uids)]
    logger.info("Training interactions: %d pairs, %d proteins",
                len(df), df["uniprot_id"].nunique())

    # Load UniProt cache
    uniprot_cache = {}
    if os.path.exists(args.uniprot_cache):
        with open(args.uniprot_cache) as f:
            uniprot_cache = json.load(f)

    # Identify binding domains per protein
    domain_dir = args.domain_dir
    binding_domains = {}  # uid -> (domain_name, domain_seq, method)
    unique_uids = sorted(df["uniprot_id"].unique())

    for i, uid in enumerate(unique_uids):
        # Find all domain PDBs for this protein
        pdbs = {}
        for dk in domain_info:
            if dk.startswith(uid + "_d"):
                pdb_path = os.path.join(domain_dir, dk + ".pdb")
                if os.path.exists(pdb_path):
                    pdbs[dk] = pdb_path

        if not pdbs:
            continue

        full_seq = protein_seqs.get(uid, "")
        if not full_seq:
            continue

        result = identify_binding_domain(uid, pdbs, full_seq)
        if result[0]:
            binding_domains[uid] = result
            if (i + 1) % 100 == 0:
                logger.info("  [%d/%d] %s: %s (%s, %d res)",
                            i + 1, len(unique_uids), uid, result[0], result[2], len(result[1]))
            # Rate limit UniProt API
            time.sleep(0.1)

    logger.info("Identified binding domains for %d/%d proteins", len(binding_domains), len(unique_uids))

    # Save UniProt cache
    os.makedirs(os.path.dirname(args.uniprot_cache), exist_ok=True)
    with open(args.uniprot_cache, "w") as f:
        json.dump(uniprot_cache, f)

    # Build output CSV
    output_rows = []
    for _, row in df.iterrows():
        uid = row["uniprot_id"]
        if uid not in binding_domains:
            continue
        domain_name, domain_seq, method = binding_domains[uid]
        output_rows.append({
            "uniprot_id": uid,
            "domain_name": domain_name,
            "domain_sequence": domain_seq,
            "ligand_smiles": row["ligand_smiles"],
            "pki": row["pki"],
            "method": method,
        })

    out_df = pd.DataFrame(output_rows)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Saved %d domain-drug pairs to %s (%d unique domains)",
                len(out_df), args.output, out_df["domain_name"].nunique())

    # Stats
    for m in ["uniprot", "largest"]:
        n = sum(1 for v in binding_domains.values() if v[2] == m)
        logger.info("  Method %s: %d proteins", m, n)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test data pipeline locally**

Run: `cd /workspace/IDP-work && PYTHONPATH=src:. python scripts/build_domain_training_data.py --interactions data/raw/benchmark_affinity.csv --domain-dir data/processed/model_organism_domains --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json --sequences data/processed/merged_sequences.json --holdout-proteins data/raw/benchmark_affinity.csv --output /tmp/test_domain_data.csv --max-proteins 5`

Expected: Creates CSV with domain_sequence, ligand_smiles, pki columns

- [ ] **Step 3: Commit**

```bash
git add scripts/build_domain_training_data.py
git commit -m "feat: data pipeline for domain-based DeepDTA training"
```

---

### Task 3: Domain-Native DeepDTA Model

**Files:**
- Create: `scripts/domain_deepdta_model.py`
- Test: `tests/test_domain_deepdta.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_domain_deepdta.py
"""Tests for domain-adapted DeepDTA model."""
import torch
import pytest


class TestDomainDeepDTA:
    def test_forward_shape(self):
        from scripts.domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        smiles = torch.randint(0, 52, (4, 100))
        protein = torch.randint(0, 26, (4, 500))
        out = model(smiles, protein)
        assert out.shape == (4,)

    def test_shorter_protein_max_len(self):
        from scripts.domain_deepdta_model import DomainDeepDTA, DOMAIN_MAX_LEN
        assert DOMAIN_MAX_LEN == 500

    def test_smaller_kernels(self):
        from scripts.domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        # Protein kernels should be 3, 4, 6
        assert model.protein_conv1.kernel_size == (3,)
        assert model.protein_conv2.kernel_size == (4,)
        assert model.protein_conv3.kernel_size == (6,)

    def test_gradient_flows(self):
        from scripts.domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        smiles = torch.randint(0, 52, (2, 100))
        protein = torch.randint(0, 26, (2, 500))
        out = model(smiles, protein)
        loss = out.sum()
        loss.backward()
        assert model.protein_embed.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /workspace/IDP-work && PYTHONPATH=src:scripts:. python -m pytest tests/test_domain_deepdta.py -v`
Expected: FAIL

- [ ] **Step 3: Implement domain-adapted model**

```python
# scripts/domain_deepdta_model.py
"""Domain-adapted DeepDTA architecture for shorter domain sequences.

Changes from standard DeepDTA:
- DOMAIN_MAX_LEN = 500 (was 1000)
- Protein CNN kernels: [3, 4, 6] (was [8, 8, 8])
- Drug CNN unchanged
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DOMAIN_MAX_LEN = 500
SMILES_MAX_LEN = 100


class DomainDeepDTA(nn.Module):
    """DeepDTA adapted for domain-length protein sequences."""

    def __init__(self, smiles_vocab=66, protein_vocab=26, embed_dim=128, num_filters=32):
        super().__init__()
        self.smiles_embed = nn.Embedding(smiles_vocab, embed_dim, padding_idx=0)
        self.protein_embed = nn.Embedding(protein_vocab, embed_dim, padding_idx=0)

        # Drug CNN: same as original (kernels 8)
        self.smiles_conv1 = nn.Conv1d(embed_dim, num_filters, 8)
        self.smiles_conv2 = nn.Conv1d(num_filters, num_filters * 2, 8)
        self.smiles_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, 8)

        # Protein CNN: smaller kernels for shorter domain sequences
        self.protein_conv1 = nn.Conv1d(embed_dim, num_filters, 3)
        self.protein_conv2 = nn.Conv1d(num_filters, num_filters * 2, 4)
        self.protein_conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, 6)

        self.fc1 = nn.Linear(num_filters * 3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, smiles_seq, protein_seq):
        smiles = self.smiles_embed(smiles_seq).permute(0, 2, 1)
        smiles = F.relu(self.smiles_conv1(smiles))
        smiles = F.relu(self.smiles_conv2(smiles))
        smiles = F.relu(self.smiles_conv3(smiles))
        smiles = smiles.max(dim=2)[0]

        protein = self.protein_embed(protein_seq).permute(0, 2, 1)
        protein = F.relu(self.protein_conv1(protein))
        protein = F.relu(self.protein_conv2(protein))
        protein = F.relu(self.protein_conv3(protein))
        protein = protein.max(dim=2)[0]

        x = torch.cat([smiles, protein], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.out(x).squeeze(-1)


def encode_domain(seq: str, max_len: int = DOMAIN_MAX_LEN) -> list[int]:
    """Encode domain sequence using DeepDTA CHARPROTSET, padded to DOMAIN_MAX_LEN."""
    from deepdta_encoding import CHARPROTSET
    return [CHARPROTSET.get(c, 0) for c in seq[:max_len]] + [0] * max(0, max_len - len(seq))
```

- [ ] **Step 4: Run tests**

Run: `cd /workspace/IDP-work && PYTHONPATH=src:scripts:. python -m pytest tests/test_domain_deepdta.py -v`
Expected: All 4 pass

- [ ] **Step 5: Commit**

```bash
git add scripts/domain_deepdta_model.py tests/test_domain_deepdta.py
git commit -m "feat: domain-adapted DeepDTA model (shorter kernels, max_len=500)"
```

---

### Task 4: Training Script

**Files:**
- Create: `scripts/train_domain_deepdta.py`

- [ ] **Step 1: Write training script**

Based on existing `train_deepdta_fair.py` but reads domain training data CSV and uses `DomainDeepDTA`.

```python
#!/usr/bin/env python3
"""Train domain-native DeepDTA on domain-drug pairs.

Usage:
    python scripts/train_domain_deepdta.py \
        --training-data data/processed/domain_training_data.csv \
        --epochs 100 --batch-size 256 --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from domain_deepdta_model import DomainDeepDTA, encode_domain, DOMAIN_MAX_LEN
from deepdta_encoding import encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class DomainDTADataset(Dataset):
    def __init__(self, smiles_encoded, domain_encoded, pkis):
        self.smiles = smiles_encoded
        self.domains = domain_encoded
        self.pkis = pkis

    def __len__(self):
        return len(self.pkis)

    def __getitem__(self, idx):
        return (torch.tensor(self.smiles[idx], dtype=torch.long),
                torch.tensor(self.domains[idx], dtype=torch.long),
                torch.tensor(self.pkis[idx], dtype=torch.float32))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-data", required=True,
                        help="CSV with domain_sequence, ligand_smiles, pki")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models/domain_deepdta")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load domain training data
    df = pd.read_csv(args.training_data)
    logger.info("Loaded %d domain-drug pairs (%d unique domains)",
                len(df), df["domain_name"].nunique())

    # Encode
    smiles_encoded = [encode_smiles(smi) for smi in df["ligand_smiles"]]
    domain_encoded = [encode_domain(seq) for seq in df["domain_sequence"]]
    pkis = df["pki"].values.astype(np.float32)

    # 90/10 split (by protein to avoid leakage)
    uids = df["uniprot_id"].unique()
    np.random.RandomState(args.seed).shuffle(uids)
    n_train_uids = int(0.9 * len(uids))
    train_uids = set(uids[:n_train_uids])

    train_mask = df["uniprot_id"].isin(train_uids).values
    val_mask = ~train_mask

    train_ds = DomainDTADataset(
        [smiles_encoded[i] for i in range(len(df)) if train_mask[i]],
        [domain_encoded[i] for i in range(len(df)) if train_mask[i]],
        pkis[train_mask],
    )
    val_ds = DomainDTADataset(
        [smiles_encoded[i] for i in range(len(df)) if val_mask[i]],
        [domain_encoded[i] for i in range(len(df)) if val_mask[i]],
        pkis[val_mask],
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    logger.info("Train: %d (%d proteins), Val: %d (%d proteins)",
                len(train_ds), n_train_uids, len(val_ds), len(uids) - n_train_uids)

    model = DomainDeepDTA().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("DomainDeepDTA parameters: %s", f"{n_params:,}")

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for s, d, y in train_loader:
            s, d, y = s.to(device), d.to(device), y.to(device)
            pred = model(s, d)
            loss = F.mse_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
        train_loss = total_loss / len(train_ds)

        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for s, d, y in val_loader:
                s, d, y = s.to(device), d.to(device), y.to(device)
                val_total += F.mse_loss(model(s, d), y).item() * len(y)
        val_loss = val_total / len(val_ds)

        scheduler.step()
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "domain_max_len": DOMAIN_MAX_LEN,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            logger.info("Epoch %d/%d train=%.4f val=%.4f best=%.4f patience=%d/%d%s",
                        epoch, args.epochs, train_loss, val_loss, best_val_loss,
                        patience_counter, args.patience, " *" if improved else "")

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Done. Best val=%.4f. Model at %s", best_val_loss, output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test with tiny data**

```bash
cd /workspace/IDP-work
# Create a tiny test CSV
python3 -c "
import csv
with open('/tmp/tiny_domain_data.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['uniprot_id','domain_name','domain_sequence','ligand_smiles','pki','method'])
    for i in range(100):
        w.writerow([f'P{i:05d}', f'P{i:05d}_d0', 'ACDEFGHIKLMNPQRSTVWY' * 5, f'CCO{i}', 7.0 + i*0.01, 'largest'])
"
PYTHONPATH=scripts:. python scripts/train_domain_deepdta.py --training-data /tmp/tiny_domain_data.csv --epochs 3 --batch-size 16 --device cpu --output-dir /tmp/test_domain_model
```

Expected: Trains 3 epochs, saves best_model.pt

- [ ] **Step 3: Commit**

```bash
git add scripts/train_domain_deepdta.py
git commit -m "feat: training script for domain-native DeepDTA"
```

---

### Task 5: Evaluation Script

**Files:**
- Create: `scripts/evaluate_anchor_dta_domain.py`

This is a copy of `evaluate_anchor_dta_type2_domseq.py` modified to:
1. Load `DomainDeepDTA` instead of `DeepDTAFair`
2. Use `encode_domain` with `DOMAIN_MAX_LEN=500`
3. Use single best anchor (highest TM) instead of TM-weighted average
4. Use `qtmscore` for proper TM values

- [ ] **Step 1: Create eval script from existing**

Copy `evaluate_anchor_dta_type2_domseq.py` and make these changes:

1. Replace `DeepDTAFair` import with `DomainDeepDTA`
2. Replace `encode_protein` with `encode_domain` (max_len=500)
3. Replace `predict_anchor_based` (TM-weighted average) with single best anchor
4. Keep the rest (conformation selection, Foldseek search, metrics) identical

Key changes in the predict function:

```python
def predict_single_anchor(
    model: DomainDeepDTA,
    anchors: list[dict],
    smiles_list: list[str],
    device: str = "cpu",
) -> np.ndarray | None:
    """Run DomainDeepDTA with the single best anchor's domain sequence."""
    if not anchors:
        return None

    # Pick highest TM anchor
    best = max(anchors, key=lambda a: a["anchor_tm"])
    seq = best["anchor_sequence"]
    if not seq:
        return None

    domain_enc = encode_domain(seq)
    domain_tensor = torch.tensor([domain_enc], dtype=torch.long, device=device)

    all_preds = []
    for i in range(0, len(smiles_list), 256):
        batch = smiles_list[i:i+256]
        smi_encs = [encode_smiles(s) for s in batch]
        smi_tensor = torch.tensor(smi_encs, dtype=torch.long, device=device)
        domain_batch = domain_tensor.expand(len(batch), -1)
        with torch.no_grad():
            preds = model(smi_tensor, domain_batch)
        all_preds.append(preds.cpu().numpy().flatten())

    return np.concatenate(all_preds)
```

- [ ] **Step 2: Test eval script runs**

```bash
cd /workspace/IDP-work
PYTHONPATH=src/idr_gat:scripts:. python scripts/evaluate_anchor_dta_domain.py \
  --domain-dir data/processed/model_organism_domains \
  --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
  --benchmark data/raw/benchmark_affinity.csv \
  --deepdta-model models/domain_deepdta/best_model.pt \
  --sequences data/processed/merged_sequences.json \
  --foldseek-bin /workspace/foldseek \
  --threads 16 --device cuda \
  --anchor-tm-threshold 0.6 \
  --max-proteins 3 \
  --output-dir results/domain_dta_test
```

Expected: Processes 3 proteins, prints per-protein metrics

- [ ] **Step 3: Commit**

```bash
git add scripts/evaluate_anchor_dta_domain.py
git commit -m "feat: eval script for domain-native DeepDTA with single best anchor"
```

---

### Task 6: End-to-End Run

- [ ] **Step 1: Build full domain training data**

```bash
cd /workspace/IDP-work
PYTHONPATH=src:. python scripts/build_domain_training_data.py \
  --interactions data/processed/training_interactions_affinity.csv \
  --domain-dir data/processed/model_organism_domains \
  --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
  --sequences data/processed/merged_sequences.json \
  --holdout-proteins data/raw/benchmark_affinity.csv \
  --output data/processed/domain_training_data.csv
```

Expected: Creates CSV with thousands of domain-drug pairs

- [ ] **Step 2: Train domain-native DeepDTA**

```bash
PYTHONPATH=scripts:. python scripts/train_domain_deepdta.py \
  --training-data data/processed/domain_training_data.csv \
  --epochs 100 --batch-size 256 --device cuda \
  --output-dir models/domain_deepdta
```

Expected: Trains until early stopping, saves best_model.pt

- [ ] **Step 3: Run full eval**

```bash
PYTHONPATH=src/idr_gat:scripts:. python scripts/evaluate_anchor_dta_domain.py \
  --domain-dir data/processed/model_organism_domains \
  --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
  --benchmark data/raw/benchmark_affinity.csv \
  --deepdta-model models/domain_deepdta/best_model.pt \
  --sequences data/processed/merged_sequences.json \
  --foldseek-bin /workspace/foldseek \
  --threads 16 --device cuda \
  --anchor-tm-threshold 0.6 \
  --output-dir results/anchor_dta_domain_native
```

Expected: Full 203-protein eval with domain-native model

- [ ] **Step 4: Run disorder analysis**

```bash
python scripts/analyze_disorder_entropy.py \
  --eval-log results/anchor_dta_domain_native/*.log \
  --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
  --sequences data/processed/merged_sequences.json \
  --output-dir results/disorder_analysis_domain_native
```

Expected: TM quartiles + disorder quartiles + heatmap for domain-native model
