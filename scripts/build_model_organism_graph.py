#!/usr/bin/env python3
"""Build extended protein graph from AlphaFold reference proteomes (model organisms).

Downloads individual AlphaFold proteome archives one at a time to fit in limited
disk space (~60GB). Streams: download tar → extract domains → delete tar → next.

Same pipeline as build_alphafold_graph.py but with ~48 reference proteomes (~300K proteins).

Usage:
  PYTHONPATH=src:. python scripts/build_model_organism_graph.py \
    --local-scratch /usr/local/scratch \
    --device cuda --threads 128
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import shutil
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AlphaFold reference proteomes (48 model organisms)
# Format: (proteome_id, taxid, name, approx_proteins)
# From: https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/
# ---------------------------------------------------------------------------

REFERENCE_PROTEOMES = [
    ("UP000005640", "9606", "HUMAN", 23391),
    ("UP000000589", "10090", "MOUSE", 21615),
    ("UP000002494", "10116", "RAT", 21272),
    ("UP000000437", "7955", "DANRE", 24664),       # Zebrafish
    ("UP000000803", "7227", "DROME", 13458),        # Drosophila
    ("UP000001940", "6239", "CAEEL", 19694),        # C. elegans
    ("UP000006548", "3702", "ARATH", 27281),        # Arabidopsis
    ("UP000002311", "559292", "YEAST", 6049),       # S. cerevisiae
    ("UP000002296", "284812", "SCHPO", 5128),       # S. pombe
    ("UP000000625", "83333", "ECOLI", 4392),        # E. coli K12
    ("UP000000318", "224308", "BACSU", 4119),       # B. subtilis
    ("UP000001584", "99287", "SALTY", 4445),        # Salmonella
    ("UP000000535", "85962", "HELPY", 1553),        # H. pylori
    ("UP000001014", "122586", "NEIM8", 2038),       # N. meningitidis
    ("UP000000429", "85963", "BUCAI", 564),         # Buchnera
    ("UP000059680", "39947", "ORYSJ", 37244),       # Rice
    ("UP000007305", "4577", "MAIZE", 39299),        # Maize
    ("UP000002195", "44689", "DICDI", 12504),       # Dictyostelium
    ("UP000008153", "5671", "LEIIN", 7924),         # Leishmania
    ("UP000001450", "36329", "PLAF7", 5187),        # P. falciparum
    ("UP000005225", "237561", "CANAL", 6035),       # C. albicans
    ("UP000000579", "71421", "HAEIN", 1709),        # H. influenzae
    ("UP000000586", "171101", "STRPN", 1965),       # S. pneumoniae
    ("UP000001570", "243274", "TREPA", 1028),       # T. pallidum
    ("UP000002485", "284813", "CANGA", 5228),       # C. glabrata
    ("UP000007841", "1280", "STAAU", 2888),         # S. aureus
    ("UP000000609", "243273", "MYCGE", 476),        # M. genitalium
    ("UP000001584", "99287", "SALTY", 4445),        # Salmonella
    ("UP000000432", "208964", "PSEAE", 5563),       # P. aeruginosa
    ("UP000002438", "208963", "PSEPK", 5350),       # P. putida
    ("UP000000803", "7227", "DROME", 13458),        # duplicate removed below
    ("UP000186698", "694009", "SARS2", 14),         # SARS-CoV-2
    ("UP000001631", "9913", "BOVIN", 23847),        # Bovine
    ("UP000002254", "9615", "CANLF", 20456),        # Dog
    ("UP000002277", "9823", "PIG", 22168),          # Pig
    ("UP000001811", "9031", "CHICK", 16736),        # Chicken
    ("UP000008827", "9544", "MACMU", 21210),        # Macaque
    ("UP000002279", "9796", "HORSE", 20363),        # Horse
    ("UP000002254", "9615", "CANLF", 20456),        # duplicate
    ("UP000001940", "6239", "CAEEL", 19694),        # duplicate
]

# Deduplicate by proteome_id
_seen = set()
UNIQUE_PROTEOMES = []
for p in REFERENCE_PROTEOMES:
    if p[0] not in _seen:
        _seen.add(p[0])
        UNIQUE_PROTEOMES.append(p)
REFERENCE_PROTEOMES = UNIQUE_PROTEOMES

# Reuse functions from build_alphafold_graph.py
from scripts.build_alphafold_graph import (
    parse_alphafold_pdb,
    extract_ordered_domains,
    write_ca_pdb,
    _process_single_structure,
    foldseek_cluster_and_search,
    union_find_cluster,
    build_graph,
    AA3TO1,
)


def download_and_extract_proteome(
    proteome_id: str,
    taxid: str,
    name: str,
    download_dir: Path,
    domain_dir: Path,
    plddt_threshold: float = 70.0,
    min_domain_length: int = 30,
    gap_tolerance: int = 5,
    n_workers: int = 0,
) -> dict:
    """Download one proteome tar, extract domains, delete tar.

    Returns partial metadata dict for this proteome.
    """
    import multiprocessing as mp

    url = f"https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/{proteome_id}_{taxid}_{name}_v6.tar"
    tar_path = download_dir / f"{proteome_id}_{name}.tar"
    extract_dir = download_dir / f"tmp_{name}"

    metadata = {
        "conformation_to_protein": {},
        "protein_sequences": {},
        "protein_aliases": {},
        "domain_info": {},
    }

    # Download
    if not tar_path.exists():
        logger.info("Downloading %s (%s)...", name, url)
        result = subprocess.run(
            ["wget", "-q", "-c", "-O", str(tar_path), url],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning("Failed to download %s: %s", name, result.stderr[:200])
            tar_path.unlink(missing_ok=True)
            return metadata
    else:
        logger.info("[CACHED] Tar for %s already exists", name)

    # Extract PDB files
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(tar_path, "r") as tf:
            pdb_members = [m for m in tf.getmembers()
                           if m.name.endswith(".pdb.gz") or m.name.endswith(".pdb")]
            logger.info("  Extracting %d PDB files from %s...", len(pdb_members), name)
            tf.extractall(extract_dir, members=pdb_members)
    except Exception as e:
        logger.warning("Failed to extract %s: %s", name, e)
        shutil.rmtree(extract_dir, ignore_errors=True)
        tar_path.unlink(missing_ok=True)
        return metadata

    # Flatten nested directories
    for nested in extract_dir.rglob("*.pdb.gz"):
        target = extract_dir / nested.name
        if nested != target:
            shutil.move(str(nested), str(target))
    for nested in extract_dir.rglob("*.pdb"):
        target = extract_dir / nested.name
        if nested != target:
            shutil.move(str(nested), str(target))

    # Process domains in parallel
    pdb_files = sorted(list(extract_dir.glob("*.pdb.gz")) + list(extract_dir.glob("*.pdb")))
    logger.info("  Processing %d structures for %s...", len(pdb_files), name)

    domain_dir.mkdir(parents=True, exist_ok=True)

    if n_workers == 0:
        n_workers = mp.cpu_count()

    args_list = [
        (pf, plddt_threshold, min_domain_length, gap_tolerance, str(domain_dir))
        for pf in pdb_files
    ]

    n_domains = 0
    with mp.Pool(n_workers) as pool:
        for result in pool.imap_unordered(_process_single_structure, args_list, chunksize=100):
            if result is None or not result["domains"]:
                continue
            uid = result["uniprot_id"]
            metadata["protein_sequences"][uid] = result["sequence"]
            metadata["protein_aliases"][uid] = [uid]
            for d in result["domains"]:
                metadata["conformation_to_protein"][d["name"]] = uid
                metadata["domain_info"][d["name"]] = {
                    "length": d["length"],
                    "mean_plddt": d["mean_plddt"],
                    "min_plddt": d["min_plddt"],
                }
                n_domains += 1

    logger.info("  %s: %d domains from %d proteins", name, n_domains, len(metadata["protein_sequences"]))

    # Delete tar and extracted PDBs to save space
    tar_path.unlink(missing_ok=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    return metadata


def merge_metadata(all_meta: list[dict]) -> dict:
    """Merge metadata dicts from multiple proteomes."""
    merged = {
        "conformation_to_protein": {},
        "protein_sequences": {},
        "protein_aliases": {},
        "domain_info": {},
    }
    for m in all_meta:
        merged["conformation_to_protein"].update(m["conformation_to_protein"])
        merged["protein_sequences"].update(m["protein_sequences"])
        merged["protein_aliases"].update(m["protein_aliases"])
        merged["domain_info"].update(m["domain_info"])
    return merged


def main():
    parser = argparse.ArgumentParser(description="Build model organism graph from AlphaFold reference proteomes")
    parser.add_argument("--plddt-threshold", type=float, default=70.0)
    parser.add_argument("--min-domain-length", type=int, default=30)
    parser.add_argument("--gap-tolerance", type=int, default=5)
    parser.add_argument("--cluster-tm", type=float, default=0.9,
                        help="Union-Find clustering threshold (default 0.9)")
    parser.add_argument("--edge-tm-low", type=float, default=0.4)
    parser.add_argument("--edge-tm-high", type=float, default=0.9)
    parser.add_argument("--foldseek-cluster-tm", type=float, default=0.7)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--threads", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--no-esm2", action="store_true")
    parser.add_argument("--esm2-model", default="esm2_t12_35M_UR50D")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--local-scratch", type=str, default=None,
                        help="Fast local NVMe for temp work")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, use existing domain PDBs")
    parser.add_argument("--proteomes", type=str, default=None,
                        help="Comma-separated list of proteome names to include (default: all)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    download_dir = data_dir / "raw" / "alphafold_proteomes"
    download_dir.mkdir(parents=True, exist_ok=True)

    # Use scratch for domain PDBs (they accumulate and are needed for Foldseek)
    if args.local_scratch:
        domain_dir = Path(args.local_scratch) / "model_organism_domains"
    else:
        domain_dir = data_dir / "processed" / "model_organism_domains"
    domain_dir.mkdir(parents=True, exist_ok=True)

    meta_path = domain_dir / "domain_metadata.json"
    tm_tag = f"tm{args.cluster_tm:.1f}".replace(".", "")
    edge_tag = f"e{args.edge_tm_low:.1f}-{args.edge_tm_high:.1f}".replace(".", "")
    graph_dir = data_dir / "graphs" / f"model_organisms_{tm_tag}_{edge_tag}"

    # Filter proteomes if specified
    proteomes = REFERENCE_PROTEOMES
    if args.proteomes:
        wanted = set(args.proteomes.upper().split(","))
        proteomes = [p for p in proteomes if p[2] in wanted]
        logger.info("Filtered to %d proteomes: %s", len(proteomes), [p[2] for p in proteomes])

    total_proteins = sum(p[3] for p in proteomes)
    logger.info("=" * 60)
    logger.info("Model Organism Graph Builder")
    logger.info("  Proteomes: %d organisms, ~%dk estimated proteins", len(proteomes), total_proteins // 1000)
    logger.info("  pLDDT >= %.1f, min domain %d residues", args.plddt_threshold, args.min_domain_length)
    logger.info("  Cluster TM: %.2f, Edge TM: [%.2f, %.2f)", args.cluster_tm,
                args.edge_tm_low, args.edge_tm_high)
    logger.info("  Domain PDBs: %s", domain_dir)
    logger.info("  Output: %s", graph_dir)
    logger.info("=" * 60)

    # Step 1: Download and extract domains one proteome at a time
    if not args.skip_download:
        if meta_path.exists():
            logger.info("[CACHED] Loading existing domain metadata from %s", meta_path)
            with open(meta_path) as f:
                metadata = json.load(f)
            logger.info("Cached: %d domains, %d proteins",
                        len(metadata["conformation_to_protein"]),
                        len(metadata["protein_sequences"]))
        else:
            all_meta = []
            for i, (pid, taxid, name, est_n) in enumerate(proteomes):
                logger.info("[%d/%d] %s (%s, ~%d proteins)...",
                            i + 1, len(proteomes), name, pid, est_n)
                meta = download_and_extract_proteome(
                    pid, taxid, name, download_dir, domain_dir,
                    plddt_threshold=args.plddt_threshold,
                    min_domain_length=args.min_domain_length,
                    gap_tolerance=args.gap_tolerance,
                    n_workers=args.workers,
                )
                all_meta.append(meta)
                # Save progress after each proteome
                merged_so_far = merge_metadata(all_meta)
                logger.info("  Running total: %d domains, %d proteins",
                            len(merged_so_far["conformation_to_protein"]),
                            len(merged_so_far["protein_sequences"]))

            metadata = merge_metadata(all_meta)
            with open(meta_path, "w") as f:
                json.dump(metadata, f)
            logger.info("Saved metadata: %d domains, %d proteins",
                        len(metadata["conformation_to_protein"]),
                        len(metadata["protein_sequences"]))
    else:
        with open(meta_path) as f:
            metadata = json.load(f)

    n_domains = len(metadata["conformation_to_protein"])
    n_proteins = len(metadata["protein_sequences"])
    logger.info("Total domains: %d from %d proteins", n_domains, n_proteins)

    if n_domains == 0:
        logger.error("No domains extracted!")
        return

    # Step 2: Foldseek cluster + search
    if args.local_scratch:
        foldseek_work = Path(args.local_scratch) / "foldseek_model_organisms"
    else:
        foldseek_work = data_dir / "processed" / "model_organism_foldseek"
    foldseek_work.mkdir(parents=True, exist_ok=True)

    sim_cache = data_dir / "processed" / "model_organism_similarity_cache.npz"
    threedi_cache = data_dir / "processed" / "model_organism_3di_seqs.json"

    if sim_cache.exists() and threedi_cache.exists():
        logger.info("[CACHED] Loading Foldseek results from cache")
        cached = np.load(sim_cache, allow_pickle=True)
        all_names = cached["names"].tolist()
        rows, cols, scores = cached["rows"], cached["cols"], cached["scores"]
        similarities = {(int(r), int(c)): float(s) for r, c, s in zip(rows, cols, scores)}
        with open(threedi_cache) as f:
            threedi_seqs = json.load(f)
        logger.info("Loaded %d names, %d pairs, %d 3Di seqs",
                    len(all_names), len(similarities), len(threedi_seqs))
    else:
        all_names, similarities, threedi_seqs = foldseek_cluster_and_search(
            domain_dir, foldseek_work,
            foldseek_bin=args.foldseek_bin,
            cluster_tm=args.foldseek_cluster_tm,
            threads=args.threads,
        )
        # Save cache
        rows_arr = np.array([k[0] for k in similarities], dtype=np.int32)
        cols_arr = np.array([k[1] for k in similarities], dtype=np.int32)
        scores_arr = np.array([v for v in similarities.values()], dtype=np.float32)
        np.savez(sim_cache, names=np.array(all_names),
                 rows=rows_arr, cols=cols_arr, scores=scores_arr)
        with open(threedi_cache, "w") as f:
            json.dump(threedi_seqs, f)
        logger.info("Saved Foldseek cache")

    # Step 3: Union-Find clustering
    names, similarities = union_find_cluster(
        all_names, similarities,
        metadata["conformation_to_protein"],
        threedi_seqs,
        cluster_tm_threshold=args.cluster_tm,
    )

    # Step 4: Build graph
    graph_dir.mkdir(parents=True, exist_ok=True)
    graph, node_ranges = build_graph(
        names, similarities, threedi_seqs,
        metadata["conformation_to_protein"],
        metadata["protein_sequences"],
        edge_tm_low=args.edge_tm_low,
        edge_tm_high=args.edge_tm_high,
        use_esm2=not args.no_esm2,
        esm2_model_name=args.esm2_model,
        device=args.device,
    )

    # Save
    torch.save(graph, graph_dir / "global_graph.pt")
    torch.save(node_ranges, graph_dir / "protein_node_ranges.pt")

    graph_meta = {
        "type": "model_organisms",
        "n_proteomes": len(proteomes),
        "organisms": [p[2] for p in proteomes],
        "plddt_threshold": args.plddt_threshold,
        "min_domain_length": args.min_domain_length,
        "cluster_tm": args.cluster_tm,
        "edge_tm_range": [args.edge_tm_low, args.edge_tm_high],
        "foldseek_cluster_tm": args.foldseek_cluster_tm,
        "n_nodes": graph.num_nodes,
        "n_edges": int(graph.edge_index.shape[1]),
        "n_proteins": len(node_ranges),
        "n_domains_before_clustering": n_domains,
    }
    with open(graph_dir / "graph_metadata.json", "w") as f:
        json.dump(graph_meta, f, indent=2)

    # Also save domain metadata to graph dir for easy access
    shutil.copy2(meta_path, graph_dir / "domain_metadata.json")

    logger.info("=" * 60)
    logger.info("DONE — Graph saved to %s", graph_dir)
    logger.info("  Nodes: %d", graph.num_nodes)
    logger.info("  Edges: %d", int(graph.edge_index.shape[1]))
    logger.info("  Proteins: %d", len(node_ranges))
    logger.info("  Organisms: %d", len(proteomes))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
