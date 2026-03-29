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

    pdb_name = pdb_path.stem
    pred_csv = output_dir / f"{pdb_name}_predictions" / f"{pdb_name}.pdb_predictions.csv"
    if not pred_csv.exists():
        pred_csv = output_dir / f"{pdb_path.name}_predictions.csv"
    return pred_csv


def parse_predictions(
    csv_path: Path,
    min_score: float = 0.0,
    min_residues: int = 0,
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

    Returns number of Cα residues written.
    """
    target_residues = set()
    for rid in residue_ids:
        parts = rid.split("_")
        if len(parts) == 2:
            try:
                target_residues.add((parts[0], int(parts[1])))
            except ValueError:
                continue

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

    Assumes P2Rank has already been run.

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
