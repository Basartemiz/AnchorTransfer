# DTC Dataset + Disorder Entropy Evaluation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DrugTargetCommons data loader for training and disorder entropy binned evaluation with TM×disorder heatmap analysis.

**Architecture:** DTC loader downloads/parses bulk CSV, filters Ki/Kd, excludes benchmark proteins, outputs training CSV. Disorder score module computes 3Di entropy + (1-pLDDT) per IDP. Eval script adds quartile reporting and 2D heatmap (TM bins × disorder bins).

**Tech Stack:** pandas, numpy, Foldseek CLI (3Di encoding), matplotlib (heatmap), existing AffinityGAT pipeline.

---

## File Structure

```
src/idr_gat/data/dtc_loader.py           — Download, filter, deduplicate DTC data
src/idr_gat/evaluation/disorder_score.py  — Compute 3Di entropy + pLDDT disorder score
scripts/evaluate_anchor_dta.py            — Add disorder quartile + heatmap reporting
tests/test_dtc_loader.py                  — Tests for DTC loader
tests/test_disorder_score.py              — Tests for disorder score
```

---

### Task 1: DTC Data Loader

**Files:**
- Create: `src/idr_gat/data/dtc_loader.py`
- Create: `tests/test_dtc_loader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dtc_loader.py`:

```python
"""Tests for DrugTargetCommons data loader."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest


def _make_fake_dtc_csv(path: Path):
    """Create a minimal DTC-format CSV for testing."""
    rows = [
        # Ki measurements — should be kept
        {"compound_id": "C1", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 10.0, "standard_units": "nM",
         "canonical_smiles": "CCO"},
        {"compound_id": "C2", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 50000.0, "standard_units": "nM",
         "canonical_smiles": "c1ccccc1"},
        # Kd measurement — should be kept
        {"compound_id": "C3", "target_id": "Q67890", "gene_name": "CDK2",
         "standard_type": "Kd", "standard_value": 100.0, "standard_units": "nM",
         "canonical_smiles": "CCN"},
        # IC50 — should be filtered out
        {"compound_id": "C4", "target_id": "Q67890", "gene_name": "CDK2",
         "standard_type": "IC50", "standard_value": 200.0, "standard_units": "nM",
         "canonical_smiles": "CCC"},
        # Duplicate (same target + smiles, different value) — should be median
        {"compound_id": "C1", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 20.0, "standard_units": "nM",
         "canonical_smiles": "CCO"},
        # Missing SMILES — should be filtered
        {"compound_id": "C5", "target_id": "P99999", "gene_name": "TP53",
         "standard_type": "Ki", "standard_value": 5.0, "standard_units": "nM",
         "canonical_smiles": ""},
        # Zero value — should be filtered
        {"compound_id": "C6", "target_id": "P99999", "gene_name": "TP53",
         "standard_type": "Ki", "standard_value": 0.0, "standard_units": "nM",
         "canonical_smiles": "CCCC"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


class TestFilterDTC:
    def test_filters_ki_kd_only(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name))
        # Only Ki and Kd kept (IC50 dropped), empty SMILES dropped, zero value dropped
        assert len(result) == 3  # C1-P12345 (deduped), C2-P12345, C3-Q67890
        assert set(result["uniprot_id"]) == {"P12345", "Q67890"}

    def test_pki_conversion(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name))
        # C1: Ki=10nM (deduped with 20nM → median 15nM) → pKi = 9 - log10(15) ≈ 7.82
        row = result[result["ligand_smiles"] == "CCO"].iloc[0]
        assert 7.5 < row["pki"] < 8.5

    def test_excludes_benchmark_proteins(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name), exclude_proteins={"P12345"})
        assert "P12345" not in result["uniprot_id"].values
        assert len(result) == 1  # only Q67890

    def test_pki_clipping(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name))
        assert result["pki"].min() >= 3.0
        assert result["pki"].max() <= 12.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_dtc_loader.py -v`

Expected: FAIL — `idr_gat.data.dtc_loader` doesn't exist.

- [ ] **Step 3: Implement DTC loader**

Create `src/idr_gat/data/dtc_loader.py`:

```python
"""DrugTargetCommons data loader.

Downloads and filters DTC bulk data to Ki/Kd interactions with valid
UniProt IDs and SMILES, converts to pKi, deduplicates.

DTC data format (CSV columns):
  compound_id, target_id, gene_name, standard_type, standard_value,
  standard_units, canonical_smiles, ...

The target_id column contains UniProt accessions.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def filter_dtc(
    csv_path: Path,
    exclude_proteins: set[str] | None = None,
    pki_min: float = 3.0,
    pki_max: float = 12.0,
) -> pd.DataFrame:
    """Load and filter DTC CSV to clean Ki/Kd interactions.

    Args:
        csv_path: path to DTC bulk CSV
        exclude_proteins: UniProt IDs to exclude (e.g., benchmark test set)
        pki_min: minimum pKi to keep
        pki_max: maximum pKi to keep

    Returns:
        DataFrame with columns: uniprot_id, ligand_smiles, pki
    """
    df = pd.read_csv(csv_path)

    # Normalize column names (DTC may use different casing)
    df.columns = df.columns.str.strip().str.lower()

    # Map to standard names
    col_map = {}
    for c in df.columns:
        if "target_id" in c or "uniprot" in c:
            col_map[c] = "uniprot_id"
        elif "smiles" in c:
            col_map[c] = "smiles"
        elif "standard_type" in c or "activity_type" in c:
            col_map[c] = "activity_type"
        elif "standard_value" in c or "activity_value" in c:
            col_map[c] = "activity_value"
        elif "standard_units" in c:
            col_map[c] = "units"
    df = df.rename(columns=col_map)

    # Filter to Ki/Kd only
    if "activity_type" in df.columns:
        df = df[df["activity_type"].str.upper().isin(["KI", "KD"])].copy()

    # Require valid SMILES and UniProt
    df = df.dropna(subset=["smiles", "uniprot_id", "activity_value"])
    df = df[df["smiles"].str.len() > 0]
    df = df[df["activity_value"] > 0]

    # Convert to pKi (assuming nM units)
    df["pki"] = 9.0 - np.log10(df["activity_value"].clip(lower=1e-3))
    df["pki"] = df["pki"].clip(pki_min, pki_max)

    # Deduplicate: median pKi per (uniprot_id, smiles) pair
    df = df.groupby(["uniprot_id", "smiles"]).agg(pki=("pki", "median")).reset_index()

    # Exclude benchmark proteins
    if exclude_proteins:
        df = df[~df["uniprot_id"].isin(exclude_proteins)]

    # Rename to match pipeline format
    df = df.rename(columns={"smiles": "ligand_smiles"})

    logger.info("DTC filtered: %d interactions, %d proteins, %d drugs",
                len(df), df["uniprot_id"].nunique(), df["ligand_smiles"].nunique())
    return df[["uniprot_id", "ligand_smiles", "pki"]]
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_dtc_loader.py -v`

Expected: All 4 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add src/idr_gat/data/dtc_loader.py tests/test_dtc_loader.py
git commit -m "feat: add DTC data loader with Ki/Kd filtering and pKi conversion"
```

---

### Task 2: Disorder Score Module

**Files:**
- Create: `src/idr_gat/evaluation/disorder_score.py`
- Create: `tests/test_disorder_score.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_disorder_score.py`:

```python
"""Tests for disorder entropy score computation."""
from __future__ import annotations

import numpy as np
import pytest


class TestThreeDiEntropy:
    def test_identical_conformations_zero_entropy(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        # All conformations have same 3Di sequence
        seqs = ["AAAAAA", "AAAAAA", "AAAAAA"]
        entropy = threedi_entropy(seqs)
        assert entropy == 0.0

    def test_max_entropy_all_different(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        # Each position has maximum diversity (20 tokens, 20 conformations)
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [c * 5 for c in alphabet]  # 20 seqs, each all same char
        entropy = threedi_entropy(seqs)
        assert entropy > 0.9  # near max

    def test_partial_entropy(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        # First position varies (A/C), second position constant (A)
        seqs = ["AA", "CA", "AA", "CA"]
        entropy = threedi_entropy(seqs)
        assert 0.0 < entropy < 1.0


class TestDisorderScore:
    def test_score_range(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        score = compute_disorder_score(threedi_entropy=0.5, mean_plddt=0.7)
        assert 0.0 <= score <= 1.0

    def test_high_disorder(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        # High 3Di entropy + low pLDDT = high disorder
        score = compute_disorder_score(threedi_entropy=0.9, mean_plddt=0.3)
        assert score > 0.7

    def test_low_disorder(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        # Low 3Di entropy + high pLDDT = low disorder
        score = compute_disorder_score(threedi_entropy=0.1, mean_plddt=0.9)
        assert score < 0.3


class TestBinByDisorder:
    def test_quartile_binning(self):
        from idr_gat.evaluation.disorder_score import bin_by_disorder
        scores = {"P1": 0.1, "P2": 0.3, "P3": 0.5, "P4": 0.7,
                  "P5": 0.2, "P6": 0.4, "P7": 0.6, "P8": 0.8}
        bins = bin_by_disorder(scores, n_bins=4)
        assert len(bins) == 4
        # Each bin should have 2 proteins
        for label, uids in bins.items():
            assert len(uids) == 2

    def test_bin_labels_ordered(self):
        from idr_gat.evaluation.disorder_score import bin_by_disorder
        scores = {"P1": 0.1, "P2": 0.9}
        bins = bin_by_disorder(scores, n_bins=2)
        labels = list(bins.keys())
        assert "Q1" in labels[0]
        assert "Q2" in labels[1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_disorder_score.py -v`

Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement disorder score**

Create `src/idr_gat/evaluation/disorder_score.py`:

```python
"""Disorder intensity score for IDPs.

Combines 3Di structural alphabet entropy (conformational variability)
with AlphaFold pLDDT (predicted disorder) into a single [0,1] score.

Higher score = more disordered = more likely induced-fit binding mechanism.
Lower score = more residual structure = more likely conformational selection.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np


# 3Di alphabet (20 structural tokens)
THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
MAX_ENTROPY = math.log2(len(THREEDI_ALPHABET))


def threedi_entropy(seqs: list[str]) -> float:
    """Compute mean per-residue Shannon entropy across 3Di sequences.

    Each sequence represents one conformation's 3Di encoding.
    Returns normalized entropy in [0, 1].
    """
    if not seqs or len(seqs) < 2:
        return 0.0

    seq_len = min(len(s) for s in seqs)
    if seq_len == 0:
        return 0.0

    entropies = []
    for pos in range(seq_len):
        counts = Counter(s[pos] for s in seqs if pos < len(s))
        total = sum(counts.values())
        if total <= 1:
            entropies.append(0.0)
            continue
        h = 0.0
        for c, n in counts.items():
            p = n / total
            if p > 0:
                h -= p * math.log2(p)
        entropies.append(h / MAX_ENTROPY)  # normalize to [0, 1]

    return float(np.mean(entropies))


def compute_disorder_score(
    threedi_entropy: float,
    mean_plddt: float,
    w_entropy: float = 0.5,
    w_plddt: float = 0.5,
) -> float:
    """Compute composite disorder score.

    Args:
        threedi_entropy: normalized 3Di entropy [0, 1]
        mean_plddt: mean pLDDT [0, 1] (will be inverted)
        w_entropy: weight for 3Di entropy
        w_plddt: weight for (1 - pLDDT)
    Returns:
        score in [0, 1], higher = more disordered
    """
    return w_entropy * threedi_entropy + w_plddt * (1.0 - mean_plddt)


def bin_by_disorder(
    scores: dict[str, float],
    n_bins: int = 4,
) -> dict[str, list[str]]:
    """Bin proteins into quantile groups by disorder score.

    Returns: {"Q1 (lowest)": [uid, ...], "Q2": [...], ...}
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1])
    uids = [uid for uid, _ in sorted_items]
    n = len(uids)

    bins = {}
    for i in range(n_bins):
        start = i * n // n_bins
        end = (i + 1) * n // n_bins
        label = f"Q{i+1}" if i > 0 and i < n_bins - 1 else (
            f"Q1 (lowest)" if i == 0 else f"Q{n_bins} (highest)")
        bins[label] = uids[start:end]

    return bins
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/test_disorder_score.py -v`

Expected: All 7 PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add src/idr_gat/evaluation/disorder_score.py tests/test_disorder_score.py
git commit -m "feat: add disorder entropy score with 3Di entropy + pLDDT"
```

---

### Task 3: Add Disorder Quartile + Heatmap to Eval Script

**Files:**
- Modify: `scripts/evaluate_anchor_dta.py`

- [ ] **Step 1: Add `--disorder-analysis` flag and imports**

Add to the argparse section (after `--max-drugs`):

```python
parser.add_argument("--disorder-analysis", action="store_true",
                    help="Compute disorder score and report per-quartile metrics + TM×disorder heatmap")
parser.add_argument("--foldseek-3di-cache", default=None,
                    help="Path to cached 3Di sequences per protein (JSON: {uid: [seq1, seq2, ...]})")
```

- [ ] **Step 2: Add disorder computation after main eval loop**

After the line `summary_df.to_csv(output_dir / "protein_summary.csv", index=False)`, add:

```python
    # Disorder entropy analysis
    if args.disorder_analysis:
        from idr_gat.evaluation.disorder_score import (
            threedi_entropy, compute_disorder_score, bin_by_disorder,
        )
        from idr_gat.data.foldseek import encode_3di
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("Computing disorder scores...")
        idp_summary = summary_df[summary_df["protein_type"] == "idp"].copy()

        # Compute 3Di entropy per IDP from conformations
        disorder_scores = {}
        for _, row in idp_summary.iterrows():
            uid = row["uniprot_id"]
            conf_dirs = idrome_index.get(uid, [])
            if not conf_dirs:
                continue

            # Collect conformation PDB paths
            from pathlib import Path as P
            pdb_paths = []
            for cd in conf_dirs:
                cd_path = P(cd)
                if cd_path.is_dir():
                    if not (cd_path / "traj.xtc").exists():
                        pdb_paths.extend(sorted(cd_path.glob("*.pdb")))
                    elif (cd_path / "top.pdb").exists():
                        pdb_paths.append(cd_path / "top.pdb")

            if len(pdb_paths) < 2:
                continue

            # Encode each PDB with Foldseek 3Di
            import tempfile, shutil
            with tempfile.TemporaryDirectory() as tmp:
                tmp_pdb_dir = P(tmp) / "pdbs"
                tmp_pdb_dir.mkdir()
                for i, p in enumerate(pdb_paths[:20]):  # cap at 20 conformations
                    shutil.copy2(p, tmp_pdb_dir / f"conf_{i:03d}.pdb")
                threedi_seqs = encode_3di(tmp_pdb_dir, foldseek_bin=args.foldseek_bin)
                if len(threedi_seqs) >= 2:
                    ent = threedi_entropy(list(threedi_seqs.values()))
                    # Use mean_anchor_tm as proxy for pLDDT (inverted)
                    # or fetch from AlphaFold if available
                    plddt_proxy = max(0.0, 1.0 - row.get("mean_anchor_tm", 0.5))
                    disorder_scores[uid] = compute_disorder_score(ent, 1.0 - plddt_proxy)

        if disorder_scores:
            idp_summary["disorder_score"] = idp_summary["uniprot_id"].map(disorder_scores)
            scored = idp_summary.dropna(subset=["disorder_score", "ci_anchor", "ci_seq"])

            # Quartile analysis
            bins = bin_by_disorder(
                {row["uniprot_id"]: row["disorder_score"] for _, row in scored.iterrows()},
                n_bins=4,
            )
            logger.info("=" * 60)
            logger.info("DISORDER QUARTILE ANALYSIS (IDPs)")
            logger.info("=" * 60)
            for label, uids in bins.items():
                sub = scored[scored["uniprot_id"].isin(uids)]
                if len(sub) == 0:
                    continue
                sci = sub["ci_seq"].mean()
                aci = sub["ci_anchor"].mean()
                smse = sub["mse_seq"].mean()
                amse = sub["mse_anchor"].mean()
                dm = (amse - smse) / smse * 100 if smse > 0 else 0
                ds_range = (sub["disorder_score"].min(), sub["disorder_score"].max())
                logger.info("  %s (n=%d, ds=%.2f-%.2f): dCI=%+.3f  dMSE=%+.1f%%",
                           label, len(sub), ds_range[0], ds_range[1], aci - sci, dm)

            # 2D Heatmap: TM bins × Disorder bins
            scored_with_tm = scored[scored["mean_anchor_tm"] > 0]
            if len(scored_with_tm) >= 8:
                tm_vals = scored_with_tm["mean_anchor_tm"].values
                ds_vals = scored_with_tm["disorder_score"].values
                dmse_vals = ((scored_with_tm["mse_anchor"] - scored_with_tm["mse_seq"]) /
                             scored_with_tm["mse_seq"].clip(lower=0.01) * 100).values

                # 4×4 grid
                tm_edges = np.percentile(tm_vals, [0, 25, 50, 75, 100])
                ds_edges = np.percentile(ds_vals, [0, 25, 50, 75, 100])

                heatmap = np.full((4, 4), np.nan)
                counts = np.zeros((4, 4), dtype=int)

                for tm, ds, dmse in zip(tm_vals, ds_vals, dmse_vals):
                    ti = min(np.searchsorted(tm_edges[1:], tm, side="right"), 3)
                    di = min(np.searchsorted(ds_edges[1:], ds, side="right"), 3)
                    if np.isnan(heatmap[di, ti]):
                        heatmap[di, ti] = 0.0
                    heatmap[di, ti] += dmse
                    counts[di, ti] += 1

                mask = counts > 0
                heatmap[mask] /= counts[mask]

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto", origin="lower")
                ax.set_xticks(range(4))
                ax.set_xticklabels([f"TM Q{i+1}" for i in range(4)])
                ax.set_yticks(range(4))
                ax.set_yticklabels([f"DS Q{i+1}" for i in range(4)])
                ax.set_xlabel("Anchor TM-score Quartile")
                ax.set_ylabel("Disorder Score Quartile")
                ax.set_title("dMSE% by Anchor TM × Disorder Score (IDPs)")
                for i in range(4):
                    for j in range(4):
                        if counts[i, j] > 0:
                            ax.text(j, i, f"{heatmap[i,j]:+.0f}%\nn={counts[i,j]}",
                                   ha="center", va="center", fontsize=9)
                plt.colorbar(im, label="dMSE%")
                plt.tight_layout()
                fig.savefig(output_dir / "tm_disorder_heatmap.pdf", dpi=150)
                fig.savefig(output_dir / "tm_disorder_heatmap.png", dpi=150)
                plt.close()
                logger.info("Saved heatmap: %s", output_dir / "tm_disorder_heatmap.png")

            # Save scores
            scored.to_csv(output_dir / "disorder_scores.csv", index=False)
            logger.info("Saved disorder scores: %s", output_dir / "disorder_scores.csv")
```

- [ ] **Step 3: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add scripts/evaluate_anchor_dta.py
git commit -m "feat: add disorder quartile analysis + TM×disorder heatmap to eval"
```

---

### Task 4: Run Tests + Push

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python3 -m pytest tests/ -v --tb=short`

Expected: All tests PASS.

- [ ] **Step 2: Push**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git push origin alphafold
```
