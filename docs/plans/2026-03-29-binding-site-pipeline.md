# Binding Site Anchor Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace domain-level graph nodes with binding-site-level nodes using P2Rank pocket prediction, enabling binding-site-specific anchor matching at TM ≥ 0.9.

**Architecture:** P2Rank predicts pockets on AlphaFold domain PDBs → extract one PDB per pocket (score > 0.5, ≥ 10 residues) → build Foldseek DB + graph from binding site PDBs. New script `build_binding_site_graph.py` reuses existing Foldseek/graph infrastructure.

**Tech Stack:** P2Rank (Java binary), Foldseek, PyTorch Geometric, existing graph builder functions.

---

## File Structure

```
src/idr_gat/data/p2rank.py                  — Run P2Rank, parse predictions, extract pocket PDBs
scripts/build_binding_site_graph.py          — Full pipeline: P2Rank → Foldseek → binding site graph
tests/test_p2rank.py                         — Tests for P2Rank integration
```

---

### Task 1: P2Rank Module — Pocket Prediction + PDB Extraction

**Files:**
- Create: `src/idr_gat/data/p2rank.py`
- Create: `tests/test_p2rank.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_p2rank.py`:

```python
"""Tests for P2Rank pocket prediction and binding site extraction."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pytest


def _make_fake_pdb(path: Path, n_residues: int = 50):
    """Write a minimal Cα-only PDB file."""
    with open(path, "w") as f:
        for i in range(n_residues):
            x, y, z = i * 3.8, 0.0, 0.0
            f.write(
                f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 90.00           C\n"
            )
        f.write("END\n")


def _make_fake_predictions_csv(output_dir: Path, pdb_name: str):
    """Create a fake P2Rank predictions CSV."""
    pred_dir = output_dir / f"{pdb_name}_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    csv_path = pred_dir / f"{pdb_name}.pdb_predictions.csv"
    with open(csv_path, "w") as f:
        f.write("name,rank,score,probability,sas_points,surf_atoms,center_x,center_y,center_z,residue_ids\n")
        f.write("pocket1,1,12.5,0.85,50,30,10.0,0.0,0.0,A_1 A_2 A_3 A_4 A_5 A_6 A_7 A_8 A_9 A_10 A_11 A_12\n")
        f.write("pocket2,2,8.2,0.62,35,20,30.0,0.0,0.0,A_20 A_21 A_22 A_23 A_24 A_25 A_26 A_27 A_28 A_29 A_30\n")
        f.write("pocket3,3,2.1,0.30,15,10,45.0,0.0,0.0,A_40 A_41 A_42 A_43 A_44\n")
    return csv_path


class TestParsePredictions:
    def test_parse_basic(self):
        from idr_gat.data.p2rank import parse_predictions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = _make_fake_predictions_csv(tmp, "test")
            pockets = parse_predictions(csv)
        assert len(pockets) == 3
        assert pockets[0]["name"] == "pocket1"
        assert pockets[0]["score"] == 12.5
        assert pockets[0]["probability"] == 0.85
        assert len(pockets[0]["residue_ids"]) == 12

    def test_filter_by_score(self):
        from idr_gat.data.p2rank import parse_predictions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = _make_fake_predictions_csv(tmp, "test")
            pockets = parse_predictions(csv, min_score=0.5)
        # pocket1 (0.85) and pocket2 (0.62) pass, pocket3 (0.30) filtered
        assert len(pockets) == 2

    def test_filter_by_residues(self):
        from idr_gat.data.p2rank import parse_predictions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = _make_fake_predictions_csv(tmp, "test")
            pockets = parse_predictions(csv, min_score=0.5, min_residues=12)
        # Only pocket1 has 12 residues
        assert len(pockets) == 1


class TestExtractPocketPDB:
    def test_extract(self):
        from idr_gat.data.p2rank import extract_pocket_pdb
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb_path = tmp / "domain.pdb"
            _make_fake_pdb(pdb_path, n_residues=50)

            out_path = tmp / "pocket1.pdb"
            residue_ids = ["A_1", "A_2", "A_3", "A_5", "A_10"]
            n = extract_pocket_pdb(pdb_path, residue_ids, out_path)

        assert out_path.exists()
        assert n == 5
        # Verify only those residues are in output
        lines = [l for l in open(out_path) if l.startswith("ATOM")]
        assert len(lines) == 5

    def test_extract_returns_zero_for_missing_residues(self):
        from idr_gat.data.p2rank import extract_pocket_pdb
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb_path = tmp / "domain.pdb"
            _make_fake_pdb(pdb_path, n_residues=5)

            out_path = tmp / "pocket.pdb"
            n = extract_pocket_pdb(pdb_path, ["A_99", "A_100"], out_path)
        assert n == 0


class TestProcessDomain:
    def test_process_domain_produces_pocket_pdbs(self):
        from idr_gat.data.p2rank import process_domain_pockets
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pdb_path = tmp / "A0A5B7_d0.pdb"
            _make_fake_pdb(pdb_path, n_residues=50)

            # Create fake P2Rank output
            _make_fake_predictions_csv(tmp / "p2rank_output", "A0A5B7_d0")

            pocket_dir = tmp / "pockets"
            pocket_dir.mkdir()
            results = process_domain_pockets(
                pdb_path,
                p2rank_output_dir=tmp / "p2rank_output",
                pocket_pdb_dir=pocket_dir,
                min_score=0.5,
                min_residues=10,
            )

        assert len(results) == 2  # pocket1 (12 res) and pocket2 (11 res) pass
        assert (pocket_dir / "A0A5B7_d0_pocket1.pdb").exists()
        assert (pocket_dir / "A0A5B7_d0_pocket2.pdb").exists()
        assert results[0]["parent_domain"] == "A0A5B7_d0"
        assert results[0]["n_residues"] >= 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_p2rank.py -v`

Expected: FAIL — `idr_gat.data.p2rank` doesn't exist.

- [ ] **Step 3: Implement P2Rank module**

Create `src/idr_gat/data/p2rank.py`:

```python
"""P2Rank pocket prediction and binding site PDB extraction.

Runs P2Rank on AlphaFold domain PDBs, parses predicted pockets,
and extracts pocket residues into separate PDB files.
"""
from __future__ import annotations

import csv
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_p2rank(
    pdb_path: Path,
    output_dir: Path,
    p2rank_bin: str = "prank",
    threads: int = 1,
    use_alphafold_config: bool = True,
) -> Path:
    """Run P2Rank on a single PDB file.

    Returns path to the predictions CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [p2rank_bin, "predict", "-f", str(pdb_path), "-o", str(output_dir), "-threads", str(threads)]
    if use_alphafold_config:
        cmd.extend(["-c", "alphafold"])

    subprocess.run(cmd, check=True, capture_output=True)

    # P2Rank outputs to {output_dir}/{pdb_name}_predictions/{pdb_name}.pdb_predictions.csv
    pdb_name = pdb_path.stem
    pred_csv = output_dir / f"{pdb_name}_predictions" / f"{pdb_name}.pdb_predictions.csv"
    if not pred_csv.exists():
        # Try alternative path format
        pred_csv = output_dir / f"{pdb_path.name}_predictions.csv"
    return pred_csv


def parse_predictions(
    csv_path: Path,
    min_score: float = 0.5,
    min_residues: int = 10,
    max_pockets: int = 5,
) -> list[dict]:
    """Parse P2Rank predictions CSV and filter pockets.

    Returns list of dicts with keys: name, score, probability, residue_ids, center.
    """
    pockets = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("name", "").strip()
            score_str = row.get("score", "0").strip()
            prob_str = row.get("probability", "0").strip()
            residues_str = row.get("residue_ids", "").strip()

            try:
                score = float(score_str)
                probability = float(prob_str)
            except ValueError:
                continue

            residue_ids = [r.strip() for r in residues_str.split() if r.strip()]

            if probability < min_score:
                continue
            if len(residue_ids) < min_residues:
                continue

            pockets.append({
                "name": name,
                "score": score,
                "probability": probability,
                "residue_ids": residue_ids,
                "center": (
                    float(row.get("center_x", 0)),
                    float(row.get("center_y", 0)),
                    float(row.get("center_z", 0)),
                ),
            })

    # Sort by score descending, limit to max_pockets
    pockets.sort(key=lambda p: p["score"], reverse=True)
    return pockets[:max_pockets]


def extract_pocket_pdb(
    domain_pdb: Path,
    residue_ids: list[str],
    output_pdb: Path,
) -> int:
    """Extract pocket residues from a domain PDB into a new PDB file.

    Args:
        domain_pdb: source PDB (full domain)
        residue_ids: list of "chain_resnum" strings (e.g., ["A_42", "A_43"])
        output_pdb: where to write the pocket PDB

    Returns number of residues written.
    """
    # Build lookup set: (chain, resnum)
    target_residues = set()
    for rid in residue_ids:
        parts = rid.split("_")
        if len(parts) == 2:
            target_residues.add((parts[0], int(parts[1])))

    written = 0
    with open(domain_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                chain = line[21].strip() or "A"
                try:
                    resnum = int(line[22:26].strip())
                except ValueError:
                    continue
                if (chain, resnum) in target_residues:
                    fout.write(line)
                    if line[12:16].strip() == "CA":
                        written += 1
        fout.write("END\n")

    return written


def process_domain_pockets(
    pdb_path: Path,
    p2rank_output_dir: Path,
    pocket_pdb_dir: Path,
    min_score: float = 0.5,
    min_residues: int = 10,
    max_pockets: int = 5,
) -> list[dict]:
    """Process a single domain: parse P2Rank output and extract pocket PDBs.

    Assumes P2Rank has already been run. Reads predictions from p2rank_output_dir.

    Returns list of pocket metadata dicts.
    """
    domain_name = pdb_path.stem
    pred_csv = p2rank_output_dir / f"{domain_name}_predictions" / f"{domain_name}.pdb_predictions.csv"
    if not pred_csv.exists():
        pred_csv = p2rank_output_dir / f"{pdb_path.name}_predictions.csv"
    if not pred_csv.exists():
        return []

    pockets = parse_predictions(pred_csv, min_score=min_score,
                                 min_residues=min_residues, max_pockets=max_pockets)

    pocket_pdb_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pocket in pockets:
        pocket_name = f"{domain_name}_{pocket['name']}"
        out_pdb = pocket_pdb_dir / f"{pocket_name}.pdb"
        n_written = extract_pocket_pdb(pdb_path, pocket["residue_ids"], out_pdb)

        if n_written >= min_residues:
            results.append({
                "pocket_name": pocket_name,
                "parent_domain": domain_name,
                "score": pocket["score"],
                "probability": pocket["probability"],
                "n_residues": n_written,
                "residue_ids": pocket["residue_ids"],
            })
        else:
            out_pdb.unlink(missing_ok=True)

    return results
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_p2rank.py -v`

Expected: All 6 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add src/idr_gat/data/p2rank.py tests/test_p2rank.py
git commit -m "feat: add P2Rank pocket prediction and binding site PDB extraction"
```

---

### Task 2: Binding Site Graph Builder Script

**Files:**
- Create: `scripts/build_binding_site_graph.py`

- [ ] **Step 1: Create the graph builder script**

Create `scripts/build_binding_site_graph.py`:

```python
#!/usr/bin/env python3
"""Build binding-site-level protein graph from AlphaFold domains + P2Rank.

Pipeline:
  1. Run P2Rank on all domain PDBs → predicted pockets
  2. Extract pocket residues → one PDB per pocket (score > 0.5, ≥ 10 residues)
  3. Foldseek 3Di + all-vs-all search on binding site PDBs
  4. Union-Find clustering at TM ≥ 0.9
  5. ESM-2 encoding of binding site sequences
  6. Build PyG graph

Usage:
  PYTHONPATH=src:. python scripts/build_binding_site_graph.py \
    --domain-dir data/processed/alphafold_human_domains \
    --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
    --p2rank-bin p2rank_2.4.2/prank \
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from idr_gat.data.p2rank import run_p2rank, process_domain_pockets

# Reuse graph building functions from the domain graph builder
from scripts.build_alphafold_graph import (
    foldseek_cluster_and_search,
    build_graph,
    AA3TO1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def extract_sequence_from_pdb(pdb_path: Path) -> str:
    """Extract amino acid sequence from Cα atoms in a PDB file."""
    seq = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                seq.append(AA3TO1.get(resname, "X"))
    return "".join(seq)


def main():
    parser = argparse.ArgumentParser(description="Build binding site graph")
    parser.add_argument("--domain-dir", type=str, required=True,
                        help="Directory with AlphaFold domain PDBs")
    parser.add_argument("--domain-metadata", type=str, required=True)
    parser.add_argument("--p2rank-bin", type=str, default="prank")
    parser.add_argument("--foldseek-bin", type=str, default="foldseek")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cluster-tm", type=float, default=0.9)
    parser.add_argument("--edge-tm-low", type=float, default=0.4)
    parser.add_argument("--edge-tm-high", type=float, default=0.9)
    parser.add_argument("--foldseek-cluster-tm", type=float, default=0.7)
    parser.add_argument("--min-pocket-score", type=float, default=0.5)
    parser.add_argument("--min-pocket-residues", type=int, default=10)
    parser.add_argument("--max-pockets-per-domain", type=int, default=5)
    parser.add_argument("--esm2-model", type=str, default="esm2_t12_35M_UR50D")
    parser.add_argument("--output-dir", type=str, default="data/graphs/binding_sites_tm09")
    parser.add_argument("--no-esm2", action="store_true")
    args = parser.parse_args()

    domain_dir = Path(args.domain_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pocket_pdb_dir = output_dir / "pocket_pdbs"
    pocket_pdb_dir.mkdir(exist_ok=True)
    p2rank_output_dir = output_dir / "p2rank_output"
    p2rank_output_dir.mkdir(exist_ok=True)

    # Load domain metadata
    with open(args.domain_metadata) as f:
        domain_metadata = json.load(f)
    conformation_to_protein = domain_metadata["conformation_to_protein"]
    protein_sequences = domain_metadata["protein_sequences"]

    # Step 1: Run P2Rank on all domains
    domain_pdbs = sorted(domain_dir.glob("*.pdb"))
    logger.info("Step 1: Running P2Rank on %d domain PDBs...", len(domain_pdbs))

    pocket_metadata = {}
    pocket_to_protein = {}
    n_total_pockets = 0

    for i, pdb_path in enumerate(domain_pdbs):
        domain_name = pdb_path.stem

        # Check if P2Rank output already exists (cached)
        pred_dir = p2rank_output_dir / f"{domain_name}_predictions"
        if not pred_dir.exists():
            try:
                run_p2rank(pdb_path, p2rank_output_dir,
                          p2rank_bin=args.p2rank_bin, threads=1,
                          use_alphafold_config=True)
            except Exception as e:
                if i < 5:
                    logger.warning("P2Rank failed for %s: %s", domain_name, e)
                continue

        # Extract pocket PDBs
        results = process_domain_pockets(
            pdb_path, p2rank_output_dir, pocket_pdb_dir,
            min_score=args.min_pocket_score,
            min_residues=args.min_pocket_residues,
            max_pockets=args.max_pockets_per_domain,
        )

        protein_id = conformation_to_protein.get(domain_name, domain_name)
        for r in results:
            pocket_metadata[r["pocket_name"]] = r
            pocket_to_protein[r["pocket_name"]] = protein_id
            n_total_pockets += 1

        if (i + 1) % 1000 == 0:
            logger.info("  %d/%d domains processed, %d pockets extracted",
                       i + 1, len(domain_pdbs), n_total_pockets)

    logger.info("P2Rank done: %d pockets from %d domains", n_total_pockets, len(domain_pdbs))

    if n_total_pockets == 0:
        logger.error("No pockets found!")
        return

    # Save pocket metadata
    with open(output_dir / "pocket_metadata.json", "w") as f:
        json.dump({
            "pocket_to_protein": pocket_to_protein,
            "pocket_metadata": pocket_metadata,
            "protein_sequences": protein_sequences,
        }, f)

    # Step 2: Foldseek pipeline on pocket PDBs
    logger.info("Step 2: Foldseek pipeline on %d pocket PDBs...", n_total_pockets)
    pocket_names = sorted(pocket_to_protein.keys())
    foldseek_work = output_dir / "foldseek_work"

    all_names, similarities, threedi_seqs = foldseek_cluster_and_search(
        pocket_names, pocket_pdb_dir, foldseek_work,
        foldseek_bin=args.foldseek_bin,
        cluster_tm=args.foldseek_cluster_tm,
        threads=args.threads,
    )

    # Step 3: Build graph
    logger.info("Step 3: Building binding site graph...")
    from idr_gat.data.foldseek import parse_3di_to_indices

    all_3di = [parse_3di_to_indices(threedi_seqs[name]) for name in all_names]
    all_protein_ids = [pocket_to_protein[name] for name in all_names]

    # ESM-2 embeddings per binding site
    esm2_embeddings = None
    if not args.no_esm2:
        pocket_sequences = {}
        for name in all_names:
            pdb_path = pocket_pdb_dir / f"{name}.pdb"
            if pdb_path.exists():
                seq = extract_sequence_from_pdb(pdb_path)
                if seq:
                    pocket_sequences[name] = seq

        logger.info("Computing ESM-2 embeddings for %d binding sites...", len(pocket_sequences))
        from idr_gat.data.esm_encoder import encode_sequences
        esm2_embeddings = encode_sequences(
            pocket_sequences,
            model_name=args.esm2_model,
            device=args.device,
        )

    from idr_gat.graph.builder import build_conformation_graph
    graph = build_conformation_graph(
        threedi_sequences=all_3di,
        similarities=similarities,
        threshold=args.edge_tm_low,
        threshold_high=args.edge_tm_high,
        protein_ids=all_protein_ids,
        conformation_ids=all_names,
        esm2_embeddings=esm2_embeddings,
    )

    # Node ranges (pocket-level, mapped to parent protein)
    node_ranges = {}
    for i, name in enumerate(all_names):
        pid = pocket_to_protein[name]
        if pid not in node_ranges:
            node_ranges[pid] = (i, i + 1)
        else:
            node_ranges[pid] = (node_ranges[pid][0], i + 1)

    logger.info("Binding site graph: %d nodes, %d edges, %d proteins",
                graph.num_nodes, graph.edge_index.shape[1], len(node_ranges))

    torch.save(graph, output_dir / "global_graph.pt")
    torch.save(node_ranges, output_dir / "protein_node_ranges.pt")

    with open(output_dir / "graph_metadata.json", "w") as f:
        json.dump({
            "n_nodes": graph.num_nodes,
            "n_edges": int(graph.edge_index.shape[1]),
            "n_proteins": len(node_ranges),
            "n_pockets": n_total_pockets,
            "cluster_tm": args.cluster_tm,
            "edge_tm_range": [args.edge_tm_low, args.edge_tm_high],
            "min_pocket_score": args.min_pocket_score,
            "min_pocket_residues": args.min_pocket_residues,
        }, f, indent=2)

    logger.info("DONE — Saved to %s", output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is syntactically valid**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -c "import ast; ast.parse(open('scripts/build_binding_site_graph.py').read()); print('OK')"`

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add scripts/build_binding_site_graph.py
git commit -m "feat: add binding site graph builder script (P2Rank → Foldseek → PyG)"
```

---

### Task 3: Install P2Rank on Remote + Test Run

- [ ] **Step 1: Download and install P2Rank on A6000**

```bash
ssh root@194.68.245.215 -p 22078 -i ~/.ssh/id_ed25519 "
cd /workspace && \
wget -q https://github.com/rdk/p2rank/releases/download/2.4.2/p2rank_2.4.2.tar.gz && \
tar xzf p2rank_2.4.2.tar.gz && \
ln -sf /workspace/p2rank_2.4.2/prank /usr/local/bin/prank && \
prank predict -h 2>&1 | head -3
"
```

Expected: P2Rank help output.

- [ ] **Step 2: Test on a single domain PDB**

```bash
ssh root@194.68.245.215 -p 22078 -i ~/.ssh/id_ed25519 "
cd /workspace/IDP-work && \
prank predict -f data/processed/alphafold_human_domains/A0A5B7_d0.pdb \
  -o /tmp/p2rank_test -c alphafold -threads 4 && \
cat /tmp/p2rank_test/A0A5B7_d0.pdb_predictions.csv 2>/dev/null || \
find /tmp/p2rank_test -name '*predictions*' -exec cat {} \;
"
```

Expected: Predictions CSV with pocket scores and residue IDs.

- [ ] **Step 3: Commit test results and push**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git push origin alphafold
```

---

### Task 4: Run Full Test Suite + Push

- [ ] **Step 1: Run all tests**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/ -v --tb=short`

Expected: All tests PASS.

- [ ] **Step 2: Push**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git push origin alphafold
```
