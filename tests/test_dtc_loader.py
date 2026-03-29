"""Tests for DrugTargetCommons data loader."""
from __future__ import annotations

import tempfile
from pathlib import Path
import pandas as pd
import pytest


def _make_fake_dtc_csv(path: Path):
    """Create a minimal DTC-format CSV for testing."""
    rows = [
        {"compound_id": "C1", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 10.0, "standard_units": "nM",
         "canonical_smiles": "CCO"},
        {"compound_id": "C2", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 50000.0, "standard_units": "nM",
         "canonical_smiles": "c1ccccc1"},
        {"compound_id": "C3", "target_id": "Q67890", "gene_name": "CDK2",
         "standard_type": "Kd", "standard_value": 100.0, "standard_units": "nM",
         "canonical_smiles": "CCN"},
        {"compound_id": "C4", "target_id": "Q67890", "gene_name": "CDK2",
         "standard_type": "IC50", "standard_value": 200.0, "standard_units": "nM",
         "canonical_smiles": "CCC"},
        {"compound_id": "C1", "target_id": "P12345", "gene_name": "BRAF",
         "standard_type": "Ki", "standard_value": 20.0, "standard_units": "nM",
         "canonical_smiles": "CCO"},
        {"compound_id": "C5", "target_id": "P99999", "gene_name": "TP53",
         "standard_type": "Ki", "standard_value": 5.0, "standard_units": "nM",
         "canonical_smiles": ""},
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
        assert len(result) == 3
        assert set(result["uniprot_id"]) == {"P12345", "Q67890"}

    def test_pki_conversion(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name))
        row = result[result["ligand_smiles"] == "CCO"].iloc[0]
        assert 7.5 < row["pki"] < 8.5

    def test_excludes_benchmark_proteins(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name), exclude_proteins={"P12345"})
        assert "P12345" not in result["uniprot_id"].values
        assert len(result) == 1

    def test_pki_clipping(self):
        from idr_gat.data.dtc_loader import filter_dtc
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            _make_fake_dtc_csv(Path(f.name))
            result = filter_dtc(Path(f.name))
        assert result["pki"].min() >= 3.0
        assert result["pki"].max() <= 12.0
