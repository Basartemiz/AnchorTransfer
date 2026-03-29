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
from pathlib import Path

import numpy as np
import torch

from idr_gat.data.p2rank import run_p2rank, process_domain_pockets

# Reuse graph building functions from the domain graph builder
from scripts.build_alphafold_graph import (
    foldseek_cluster_and_search,
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

    # Step 1: Run P2Rank on all domains and extract pocket PDBs
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
            "pocket_metadata": {k: {kk: vv for kk, vv in v.items() if kk != "residue_ids"}
                                for k, v in pocket_metadata.items()},
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
