"""AlphaFold structure download, pLDDT filtering, and Cα extraction."""

import logging
from pathlib import Path

import numpy as np
import requests

logger = logging.getLogger(__name__)

AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def fetch_alphafold_metadata(uniprot_id: str) -> dict | None:
    """Fetch AlphaFold entry metadata including PDB URL."""
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                return data[0]
    except Exception as e:
        logger.debug("AlphaFold API failed for %s: %s", uniprot_id, e)
    return None


def download_alphafold_pdb(uniprot_id: str, output_path: Path) -> bool:
    """Download AlphaFold PDB structure."""
    meta = fetch_alphafold_metadata(uniprot_id)
    if not meta or "pdbUrl" not in meta:
        return False
    try:
        resp = requests.get(meta["pdbUrl"], timeout=30)
        if resp.status_code == 200:
            output_path.write_text(resp.text)
            return True
    except Exception as e:
        logger.debug("Download failed for %s: %s", uniprot_id, e)
    return False


def extract_ca_coords_and_plddt(pdb_path: Path) -> tuple[np.ndarray, list[float], str]:
    """Extract Cα coordinates, pLDDT scores (B-factor column), and sequence.

    In AlphaFold PDBs, the B-factor column contains pLDDT confidence scores.

    Returns:
        coords: (N, 3) Cα coordinates
        plddt: list of per-residue pLDDT scores
        sequence: one-letter amino acid sequence
    """
    coords = []
    plddt = []
    sequence = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                bfactor = float(line[60:66])
                resname = line[17:20].strip()

                coords.append([x, y, z])
                plddt.append(bfactor)
                sequence.append(AA3TO1.get(resname, "X"))

    return np.array(coords, dtype=np.float32), plddt, "".join(sequence)


def filter_by_plddt(plddt_scores: list[float], threshold: float = 70.0) -> bool:
    """Return True if mean pLDDT is above threshold."""
    if not plddt_scores:
        return False
    return float(np.mean(plddt_scores)) >= threshold
