#!/usr/bin/env python3
"""Build domain-based training data for domain-native DeepDTA.

For each protein in the interactions file:
1. Identify the binding domain (UniProt annotations or largest)
2. Extract domain amino acid sequence from PDB
3. Output CSV: domain_sequence, ligand_smiles, pki

Usage:
    python scripts/build_domain_training_data.py \
        --interactions data/processed/training_interactions_affinity.csv \
        --domain-dir data/processed/model_organism_domains \
        --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
        --sequences data/processed/merged_sequences.json \
        --holdout-proteins data/raw/benchmark_affinity.csv \
        --output data/processed/domain_training_data.csv
"""
from __future__ import annotations

import argparse
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
    parser.add_argument("--uniprot-cache", default="data/processed/uniprot_binding_cache.json")
    parser.add_argument("--max-proteins", type=int, default=0,
                        help="Limit proteins for testing (0=all)")
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

    # Identify binding domains per protein
    domain_dir = args.domain_dir
    binding_domains = {}
    unique_uids = sorted(df["uniprot_id"].unique())
    if args.max_proteins > 0:
        unique_uids = unique_uids[:args.max_proteins]

    for i, uid in enumerate(unique_uids):
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
            time.sleep(0.05)  # rate limit UniProt API

    logger.info("Identified binding domains for %d/%d proteins",
                len(binding_domains), len(unique_uids))

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
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out_df.to_csv(args.output, index=False)
    logger.info("Saved %d domain-drug pairs to %s (%d unique domains)",
                len(out_df), args.output, out_df["domain_name"].nunique())

    for m in ["uniprot", "largest"]:
        n = sum(1 for v in binding_domains.values() if v[2] == m)
        logger.info("  Method %s: %d proteins", m, n)


if __name__ == "__main__":
    main()
