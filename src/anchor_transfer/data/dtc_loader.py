"""DrugTargetCommons data loader.

Downloads and filters DTC bulk data to Ki/Kd interactions with valid
UniProt IDs and SMILES, converts to pKi, deduplicates.
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
    df.columns = df.columns.str.strip().str.lower()

    # Map to standard names (DTC column names vary across versions)
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

    df = df.rename(columns={"smiles": "ligand_smiles"})

    logger.info("DTC filtered: %d interactions, %d proteins, %d drugs",
                len(df), df["uniprot_id"].nunique(), df["ligand_smiles"].nunique())
    return df[["uniprot_id", "ligand_smiles", "pki"]]
