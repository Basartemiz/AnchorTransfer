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
        assert len(pockets) == 2

    def test_filter_by_residues(self):
        from idr_gat.data.p2rank import parse_predictions
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            csv = _make_fake_predictions_csv(tmp, "test")
            pockets = parse_predictions(csv, min_score=0.5, min_residues=12)
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

            assert len(results) == 2
            assert (pocket_dir / "A0A5B7_d0_pocket1.pdb").exists()
            assert (pocket_dir / "A0A5B7_d0_pocket2.pdb").exists()
            assert results[0]["parent_domain"] == "A0A5B7_d0"
            assert results[0]["n_residues"] >= 10
