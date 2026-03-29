#!/usr/bin/env python3
"""Build a protein graph from AlphaFold Swiss-Prot predicted structures.

Pipeline:
  1. Download bulk AlphaFold proteome (human first, then full Swiss-Prot)
  2. Extract ordered domains by per-residue pLDDT filtering
  3. Foldseek cluster → representative all-vs-all search
  4. Union-Find global clustering + graph construction
  5. ESM-2 embeddings + final graph output

Usage:
  python scripts/build_alphafold_graph.py --proteome human --plddt-threshold 70
  python scripts/build_alphafold_graph.py --proteome swissprot --plddt-threshold 70
"""

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
# Config
# ---------------------------------------------------------------------------

PROTEOME_URLS = {
    "human": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/UP000005640_9606_HUMAN_v6.tar",
    "swissprot": "https://ftp.ebi.ac.uk/pub/databases/alphafold/latest/swissprot_pdb_v6.tar",
}

DEFAULT_PLDDT_THRESHOLD = 70.0
DEFAULT_MIN_DOMAIN_LENGTH = 30
DEFAULT_GAP_TOLERANCE = 5
DEFAULT_CLUSTER_TM = 0.8
DEFAULT_EDGE_TM_LOW = 0.3
DEFAULT_EDGE_TM_HIGH = 0.8
DEFAULT_FOLDSEEK_CLUSTER_TM = 0.7  # pre-clustering threshold for scalability


AA3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ---------------------------------------------------------------------------
# Step 1: Bulk download
# ---------------------------------------------------------------------------

def download_proteome(proteome: str, download_dir: Path) -> Path:
    """Download AlphaFold proteome archive. Returns path to extracted dir."""
    download_dir.mkdir(parents=True, exist_ok=True)
    url = PROTEOME_URLS[proteome]
    tar_name = url.split("/")[-1]
    tar_path = download_dir / tar_name
    extract_dir = download_dir / f"alphafold_{proteome}"

    if extract_dir.exists() and any(extract_dir.glob("*.pdb.gz")) or any(extract_dir.glob("*.pdb")):
        n = len(list(extract_dir.glob("*.pdb*")))
        logger.info("[CACHED] %d AlphaFold structures already extracted in %s", n, extract_dir)
        return extract_dir

    if not tar_path.exists():
        logger.info("Downloading %s (~%s)...", url,
                     "23GB" if proteome == "human" else "~2TB")
        subprocess.run(
            ["wget", "-c", "--progress=dot:giga", "-O", str(tar_path), url],
            check=True,
        )
    else:
        logger.info("[CACHED] Tar archive already at %s", tar_path)

    logger.info("Extracting %s...", tar_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    # Extract only PDB files (skip CIF and PAE JSON to save space)
    with tarfile.open(tar_path, "r") as tf:
        pdb_members = [m for m in tf.getmembers()
                       if m.name.endswith(".pdb.gz") or m.name.endswith(".pdb")]
        logger.info("Extracting %d PDB files...", len(pdb_members))
        tf.extractall(extract_dir, members=pdb_members)

    # Flatten any nested directories
    for nested in extract_dir.rglob("*.pdb.gz"):
        target = extract_dir / nested.name
        if nested != target:
            shutil.move(str(nested), str(target))
    for nested in extract_dir.rglob("*.pdb"):
        target = extract_dir / nested.name
        if nested != target:
            shutil.move(str(nested), str(target))

    logger.info("Extracted %d PDB files to %s",
                len(list(extract_dir.glob("*.pdb*"))), extract_dir)
    return extract_dir


# ---------------------------------------------------------------------------
# Step 2: Domain extraction by pLDDT
# ---------------------------------------------------------------------------

def parse_alphafold_pdb(pdb_path: Path) -> tuple[list[dict], str]:
    """Parse an AlphaFold PDB (possibly gzipped). Returns list of residue dicts and UniProt ID."""
    open_fn = gzip.open if str(pdb_path).endswith(".gz") else open
    residues = []
    uniprot_id = pdb_path.stem.replace(".pdb", "")
    # Extract UniProt from filename: AF-P04637-F1-model_v4.pdb.gz -> P04637
    parts = uniprot_id.split("-")
    if len(parts) >= 2 and parts[0] == "AF":
        uniprot_id = parts[1]

    with open_fn(pdb_path, "rt") as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                resname = line[17:20].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                plddt = float(line[60:66])
                residues.append({
                    "resname": resname,
                    "aa": AA3TO1.get(resname, "X"),
                    "coords": [x, y, z],
                    "plddt": plddt,
                })
    return residues, uniprot_id


def extract_ordered_domains(
    residues: list[dict],
    plddt_threshold: float = 70.0,
    min_domain_length: int = 30,
    gap_tolerance: int = 5,
) -> list[list[dict]]:
    """Extract contiguous ordered domains from pLDDT-scored residues.

    A domain is a contiguous stretch where most residues have pLDDT >= threshold.
    Short gaps (up to gap_tolerance low-pLDDT residues) within a domain are tolerated.

    Returns list of domains, each a list of residue dicts.
    """
    if not residues:
        return []

    # Mark each residue as ordered or not
    is_ordered = [r["plddt"] >= plddt_threshold for r in residues]

    # Find contiguous ordered regions with gap tolerance
    domains = []
    current_domain_start = None
    gap_count = 0

    for i, ordered in enumerate(is_ordered):
        if ordered:
            if current_domain_start is None:
                current_domain_start = i
            gap_count = 0
        else:
            if current_domain_start is not None:
                gap_count += 1
                if gap_count > gap_tolerance:
                    # End current domain (exclude trailing gap)
                    domain_end = i - gap_count
                    domain = residues[current_domain_start:domain_end + 1]
                    if len(domain) >= min_domain_length:
                        domains.append(domain)
                    current_domain_start = None
                    gap_count = 0

    # Handle last domain
    if current_domain_start is not None:
        # Trim trailing gap
        domain_end = len(residues) - 1
        while domain_end >= current_domain_start and not is_ordered[domain_end]:
            domain_end -= 1
        domain = residues[current_domain_start:domain_end + 1]
        if len(domain) >= min_domain_length:
            domains.append(domain)

    return domains


def write_ca_pdb(residues: list[dict], output_path: Path) -> None:
    """Write Cα-only PDB for a domain."""
    with open(output_path, "w") as f:
        for i, r in enumerate(residues):
            x, y, z = r["coords"]
            aa3 = next((k for k, v in AA3TO1.items() if v == r["aa"]), "ALA")
            f.write(
                f"ATOM  {i+1:5d}  CA  {aa3:3s} A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{r['plddt']:6.2f}\n"
            )
        f.write("END\n")


def process_alphafold_structures(
    af_dir: Path,
    output_dir: Path,
    plddt_threshold: float = 70.0,
    min_domain_length: int = 30,
    gap_tolerance: int = 5,
    local_scratch: Path | None = None,
) -> dict:
    """Process all AlphaFold structures → extract ordered domains as Cα PDBs.

    If local_scratch is set, writes PDBs to fast local disk first, then bulk
    copies to output_dir (avoids slow NFS per-file writes).

    Returns metadata dict with conformation_to_protein, protein_sequences, protein_aliases.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "conformation_to_protein": {},
        "protein_sequences": {},
        "protein_aliases": {},
        "domain_info": {},  # domain_name -> {start, end, mean_plddt, length}
    }

    # Check for cached metadata
    meta_path = output_dir / "domain_metadata.json"
    if meta_path.exists():
        existing_pdbs = list(output_dir.glob("*.pdb"))
        if existing_pdbs:
            logger.info("[CACHED] Loading domain metadata (%d domains)", len(existing_pdbs))
            with open(meta_path) as f:
                metadata = json.load(f)
            return metadata

    # Use local scratch for fast writes, bulk copy at the end
    write_dir = output_dir
    if local_scratch is not None:
        write_dir = local_scratch / "alphafold_domains"
        if write_dir.exists():
            shutil.rmtree(write_dir)
        write_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using local scratch: %s (will bulk copy to %s)", write_dir, output_dir)

    pdb_files = sorted(list(af_dir.glob("*.pdb.gz")) + list(af_dir.glob("*.pdb")))
    logger.info("Processing %d AlphaFold structures (pLDDT >= %.1f, min_len=%d)...",
                len(pdb_files), plddt_threshold, min_domain_length)

    n_domains_total = 0
    n_skipped = 0
    n_no_domain = 0

    for idx, pdb_file in enumerate(pdb_files):
        if (idx + 1) % 5000 == 0:
            logger.info("  Processed %d/%d structures (%d domains extracted)",
                        idx + 1, len(pdb_files), n_domains_total)

        try:
            residues, uniprot_id = parse_alphafold_pdb(pdb_file)
        except Exception as e:
            logger.debug("Failed to parse %s: %s", pdb_file.name, e)
            n_skipped += 1
            continue

        if not residues:
            n_skipped += 1
            continue

        domains = extract_ordered_domains(
            residues,
            plddt_threshold=plddt_threshold,
            min_domain_length=min_domain_length,
            gap_tolerance=gap_tolerance,
        )

        if not domains:
            n_no_domain += 1
            continue

        # Full sequence for ESM-2
        full_seq = "".join(r["aa"] for r in residues)
        metadata["protein_sequences"][uniprot_id] = full_seq
        metadata["protein_aliases"][uniprot_id] = [uniprot_id]

        for d_idx, domain in enumerate(domains):
            domain_name = f"{uniprot_id}_d{d_idx}"
            domain_pdb_path = write_dir / f"{domain_name}.pdb"

            write_ca_pdb(domain, domain_pdb_path)

            metadata["conformation_to_protein"][domain_name] = uniprot_id
            plddt_scores = [r["plddt"] for r in domain]
            metadata["domain_info"][domain_name] = {
                "length": len(domain),
                "mean_plddt": float(np.mean(plddt_scores)),
                "min_plddt": float(np.min(plddt_scores)),
            }
            n_domains_total += 1

    logger.info("Domain extraction complete: %d domains from %d proteins "
                "(skipped %d, no-domain %d)",
                n_domains_total, len(metadata["protein_sequences"]),
                n_skipped, n_no_domain)

    # Bulk copy from local scratch to NFS output_dir
    if local_scratch is not None and write_dir != output_dir:
        logger.info("Keeping %d domain PDBs on local scratch at %s (skipping NFS copy)",
                    n_domains_total, write_dir)
        # Return write_dir as the actual domain PDB location for downstream steps
        metadata["_domain_pdb_dir"] = str(write_dir)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


def _process_single_structure(args):
    """Worker function for parallel domain extraction."""
    pdb_file, plddt_threshold, min_domain_length, gap_tolerance, write_dir = args
    try:
        residues, uniprot_id = parse_alphafold_pdb(pdb_file)
    except Exception:
        return None

    if not residues:
        return None

    domains = extract_ordered_domains(
        residues,
        plddt_threshold=plddt_threshold,
        min_domain_length=min_domain_length,
        gap_tolerance=gap_tolerance,
    )

    if not domains:
        return {"uniprot_id": uniprot_id, "domains": []}

    full_seq = "".join(r["aa"] for r in residues)
    result = {
        "uniprot_id": uniprot_id,
        "sequence": full_seq,
        "domains": [],
    }

    for d_idx, domain in enumerate(domains):
        domain_name = f"{uniprot_id}_d{d_idx}"
        domain_pdb_path = Path(write_dir) / f"{domain_name}.pdb"
        write_ca_pdb(domain, domain_pdb_path)

        plddt_scores = [r["plddt"] for r in domain]
        result["domains"].append({
            "name": domain_name,
            "length": len(domain),
            "mean_plddt": float(np.mean(plddt_scores)),
            "min_plddt": float(np.min(plddt_scores)),
        })

    return result


def process_alphafold_structures_parallel(
    af_dir: Path,
    output_dir: Path,
    plddt_threshold: float = 70.0,
    min_domain_length: int = 30,
    gap_tolerance: int = 5,
    local_scratch: Path | None = None,
    n_workers: int = 0,
) -> dict:
    """Parallel version of process_alphafold_structures using multiprocessing.

    Uses all CPU cores by default for maximum throughput.
    """
    import multiprocessing as mp

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "conformation_to_protein": {},
        "protein_sequences": {},
        "protein_aliases": {},
        "domain_info": {},
    }

    meta_path = output_dir / "domain_metadata.json"
    if meta_path.exists():
        existing_pdbs = list(output_dir.glob("*.pdb"))
        if existing_pdbs:
            logger.info("[CACHED] Loading domain metadata (%d domains)", len(existing_pdbs))
            with open(meta_path) as f:
                metadata = json.load(f)
            return metadata

    write_dir = output_dir
    if local_scratch is not None:
        write_dir = local_scratch / "alphafold_domains"
        if write_dir.exists():
            shutil.rmtree(write_dir)
        write_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Using local scratch: %s", write_dir)

    pdb_files = sorted(list(af_dir.glob("*.pdb.gz")) + list(af_dir.glob("*.pdb")))
    logger.info("Processing %d AlphaFold structures with %d workers (pLDDT >= %.1f, min_len=%d)...",
                len(pdb_files), n_workers or mp.cpu_count(), plddt_threshold, min_domain_length)

    args_list = [
        (pf, plddt_threshold, min_domain_length, gap_tolerance, str(write_dir))
        for pf in pdb_files
    ]

    if n_workers == 0:
        n_workers = mp.cpu_count()

    n_domains_total = 0
    n_skipped = 0
    n_no_domain = 0

    with mp.Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_process_single_structure, args_list, chunksize=100)):
            if (i + 1) % 5000 == 0:
                logger.info("  Processed %d/%d structures (%d domains extracted)",
                            i + 1, len(pdb_files), n_domains_total)

            if result is None:
                n_skipped += 1
                continue

            if not result["domains"]:
                n_no_domain += 1
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
                n_domains_total += 1

    logger.info("Domain extraction complete: %d domains from %d proteins "
                "(skipped %d, no-domain %d)",
                n_domains_total, len(metadata["protein_sequences"]),
                n_skipped, n_no_domain)

    if local_scratch is not None and write_dir != output_dir:
        logger.info("Keeping %d domain PDBs on local scratch at %s (skipping NFS copy)",
                    n_domains_total, write_dir)
        # Return write_dir as the actual domain PDB location for downstream steps
        metadata["_domain_pdb_dir"] = str(write_dir)

    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Step 3: Foldseek cluster-then-search
# ---------------------------------------------------------------------------

def foldseek_cluster_and_search(
    pdb_dir: Path,
    work_dir: Path,
    foldseek_bin: str = "foldseek",
    cluster_tm: float = 0.7,
    search_max_accept: int = 300,
    threads: int = 8,
) -> tuple[list[str], dict[tuple[int, int], float], dict[str, str]]:
    """Foldseek cluster → representative all-vs-all → sparse similarities.

    Returns:
        all_names: list of ALL domain names (not just reps)
        similarities: sparse (i, j) -> TM-score for all domains (propagated from reps)
        threedi_seqs: dict of domain_name -> 3Di sequence
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    db_path = work_dir / "domaindb"
    cluster_path = work_dir / "cluster"
    rep_dir = work_dir / "rep_pdbs"
    tsv_3di = work_dir / "3di_seqs.tsv"

    all_names = sorted(p.stem for p in pdb_dir.glob("*.pdb"))
    logger.info("Foldseek pipeline: %d domain structures", len(all_names))

    # --- Create database ---
    db_marker = Path(str(db_path) + ".dbtype")
    if not db_marker.exists():
        logger.info("Creating Foldseek database...")
        subprocess.run([
            foldseek_bin, "createdb", str(pdb_dir), str(db_path),
        ], check=True, capture_output=True)

    # --- 3Di encoding ---
    if tsv_3di.exists() and tsv_3di.stat().st_size > 0:
        logger.info("[CACHED] Loading 3Di sequences from %s", tsv_3di)
    else:
        logger.info("Extracting 3Di sequences...")
        subprocess.run([
            foldseek_bin, "lndb", str(db_path) + "_h", str(db_path) + "_ss_h",
        ], check=True, capture_output=True)
        subprocess.run([
            foldseek_bin, "convert2fasta", str(db_path) + "_ss", str(tsv_3di),
        ], check=True, capture_output=True)

    # Parse 3Di
    threedi_seqs = {}
    current_name = None
    current_seq = []
    with open(tsv_3di) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    threedi_seqs[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        threedi_seqs[current_name] = "".join(current_seq)

    # --- Clustering ---
    cluster_tsv = work_dir / "cluster_members.tsv"
    if cluster_tsv.exists() and cluster_tsv.stat().st_size > 0:
        logger.info("[CACHED] Loading clusters from %s", cluster_tsv)
    else:
        logger.info("Clustering at TM >= %.2f...", cluster_tm)
        subprocess.run([
            foldseek_bin, "cluster",
            str(db_path), str(cluster_path), str(work_dir / "tmp_cluster"),
            "--min-seq-id", str(cluster_tm),
            "--alignment-type", "2",  # TM-score based
            "--threads", str(threads),
        ], check=True, capture_output=True)

        subprocess.run([
            foldseek_bin, "createtsv",
            str(db_path), str(db_path), str(cluster_path), str(cluster_tsv),
        ], check=True, capture_output=True)

    # Parse clusters: rep -> [members]
    import pandas as pd
    cluster_df = pd.read_csv(cluster_tsv, sep="\t", header=None, names=["rep", "member"])
    clusters = {}
    for _, row in cluster_df.iterrows():
        clusters.setdefault(row["rep"], []).append(row["member"])

    reps = sorted(clusters.keys())
    n_clusters = len(reps)
    logger.info("Foldseek clustering: %d domains → %d clusters (%.1f%% reduction)",
                len(all_names), n_clusters,
                100 * (1 - n_clusters / max(len(all_names), 1)))

    # --- Extract representative PDBs for all-vs-all search ---
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_set = set(reps)
    n_linked = 0
    for pdb_file in pdb_dir.glob("*.pdb"):
        if pdb_file.stem in rep_set:
            dst = rep_dir / pdb_file.name
            if not dst.exists():
                shutil.copy2(pdb_file, dst)
                n_linked += 1
    logger.info("Copied %d representative PDBs to %s", n_linked, rep_dir)

    # --- All-vs-all search among representatives ---
    rep_db = work_dir / "repdb"
    rep_result = work_dir / "rep_result"
    rep_aln_tsv = work_dir / "rep_alignment.tsv"

    if rep_aln_tsv.exists() and rep_aln_tsv.stat().st_size > 0:
        logger.info("[CACHED] Loading representative alignments from %s", rep_aln_tsv)
    else:
        rep_db_marker = Path(str(rep_db) + ".dbtype")
        if not rep_db_marker.exists():
            logger.info("Creating representative database (%d structures)...", len(reps))
            subprocess.run([
                foldseek_bin, "createdb", str(rep_dir), str(rep_db),
            ], check=True, capture_output=True)

        logger.info("All-vs-all search among %d representatives...", len(reps))
        subprocess.run([
            foldseek_bin, "search",
            str(rep_db), str(rep_db), str(rep_result),
            str(work_dir / "tmp_search"),
            "-a",
            "--alignment-type", "2",  # 3Di+AA prefilter + TM-score rescore (fast)
            "--max-accept", str(search_max_accept),
            "--threads", str(threads),
        ], check=True, capture_output=True)

        subprocess.run([
            foldseek_bin, "convertalis",
            str(rep_db), str(rep_db), str(rep_result), str(rep_aln_tsv),
            "--format-output", "query,target,alntmscore",
        ], check=True, capture_output=True)

    # Parse representative similarities
    rep_name_to_idx = {n: i for i, n in enumerate(all_names)}
    similarities: dict[tuple[int, int], float] = {}

    if rep_aln_tsv.stat().st_size > 0:
        rep_df = pd.read_csv(rep_aln_tsv, sep="\t", header=None,
                             names=["query", "target", "tmscore"])

        # Build rep -> members mapping with indices
        rep_member_indices: dict[str, list[int]] = {}
        for rep_name, members in clusters.items():
            rep_member_indices[rep_name] = [
                rep_name_to_idx[m] for m in members if m in rep_name_to_idx
            ]

        for _, row in rep_df.iterrows():
            q_rep = row["query"]
            t_rep = row["target"]
            score = float(row["tmscore"])
            if q_rep == t_rep:
                continue

            # Propagate score: all members of q_rep cluster get edges to all members of t_rep cluster
            q_members = rep_member_indices.get(q_rep, [])
            t_members = rep_member_indices.get(t_rep, [])

            for qi in q_members:
                for ti in t_members:
                    if qi == ti:
                        continue
                    key = (min(qi, ti), max(qi, ti))
                    similarities[key] = max(similarities.get(key, 0.0), score)

    logger.info("Sparse similarities: %d pairs from %d representative alignments",
                len(similarities), len(reps))

    return all_names, similarities, threedi_seqs


# ---------------------------------------------------------------------------
# Step 4: Global clustering + graph construction
# ---------------------------------------------------------------------------

def union_find_cluster(
    names: list[str],
    similarities: dict[tuple[int, int], float],
    conformation_to_protein: dict[str, str],
    threedi_seqs: dict[str, str],
    cluster_tm_threshold: float = 0.8,
) -> tuple[list[str], dict[tuple[int, int], float]]:
    """Union-Find global clustering (identical to existing pipeline)."""
    n = len(names)
    logger.info("Global clustering %d nodes (TM >= %.2f)...", n, cluster_tm_threshold)

    if n <= 1:
        return names, similarities

    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j), score in similarities.items():
        if score >= cluster_tm_threshold:
            union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    # Pick medoid per cluster
    keep_indices = []
    for members in clusters.values():
        if len(members) == 1:
            keep_indices.append(members[0])
        else:
            best_idx, best_avg = members[0], -1.0
            for m in members:
                total = sum(
                    similarities.get((min(m, o), max(m, o)), 0.0)
                    for o in members if o != m
                )
                avg = total / (len(members) - 1)
                if avg > best_avg:
                    best_avg = avg
                    best_idx = m
            keep_indices.append(best_idx)

    keep_indices.sort()
    new_names = [names[i] for i in keep_indices]

    old_to_new = {old: new for new, old in enumerate(keep_indices)}
    kept_set = set(keep_indices)
    new_similarities: dict[tuple[int, int], float] = {}
    for (i, j), score in similarities.items():
        if i in kept_set and j in kept_set:
            ni, nj = old_to_new[i], old_to_new[j]
            key = (min(ni, nj), max(ni, nj))
            new_similarities[key] = max(new_similarities.get(key, 0.0), score)

    n_proteins_before = len(set(conformation_to_protein.get(nm, nm) for nm in names))
    n_proteins_after = len(set(conformation_to_protein.get(nm, nm) for nm in new_names))
    logger.info("Global clustering: %d → %d nodes (%d clusters), proteins %d → %d",
                n, len(keep_indices), len(clusters), n_proteins_before, n_proteins_after)

    return new_names, new_similarities


def build_graph(
    names: list[str],
    similarities: dict[tuple[int, int], float],
    threedi_seqs: dict[str, str],
    conformation_to_protein: dict[str, str],
    protein_sequences: dict[str, str],
    edge_tm_low: float = 0.3,
    edge_tm_high: float = 0.8,
    use_esm2: bool = True,
    esm2_model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
) -> tuple:
    """Build PyG graph from clustered domains."""
    from idr_gat.data.foldseek import parse_3di_to_indices
    from idr_gat.graph.builder import build_conformation_graph

    all_3di = [parse_3di_to_indices(threedi_seqs[name]) for name in names]
    all_protein_ids = [conformation_to_protein[name] for name in names]

    # Protein node ranges
    node_ranges = {}
    current_start = 0
    while current_start < len(all_protein_ids):
        pid = all_protein_ids[current_start]
        current_end = current_start + 1
        while current_end < len(all_protein_ids) and all_protein_ids[current_end] == pid:
            current_end += 1
        node_ranges[pid] = (current_start, current_end)
        current_start = current_end

    # ESM-2 embeddings
    esm2_embeddings = None
    if use_esm2:
        logger.info("Computing ESM-2 embeddings for %d proteins...", len(protein_sequences))
        from idr_gat.data.esm_encoder import encode_sequences
        esm2_embeddings = encode_sequences(
            protein_sequences,
            model_name=esm2_model_name,
            device=device,
        )
        logger.info("ESM-2 embeddings computed for %d proteins", len(esm2_embeddings))

    # Build graph
    graph = build_conformation_graph(
        threedi_sequences=all_3di,
        similarities=similarities,
        threshold=edge_tm_low,
        threshold_high=edge_tm_high,
        protein_ids=all_protein_ids,
        conformation_ids=names,
        esm2_embeddings=esm2_embeddings,
    )

    logger.info("Graph: %d nodes, %d edges, %d proteins",
                graph.num_nodes, graph.edge_index.shape[1], len(node_ranges))

    return graph, node_ranges


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build AlphaFold domain graph")
    parser.add_argument("--proteome", choices=["human", "swissprot"], default="human")
    parser.add_argument("--plddt-threshold", type=float, default=DEFAULT_PLDDT_THRESHOLD)
    parser.add_argument("--min-domain-length", type=int, default=DEFAULT_MIN_DOMAIN_LENGTH)
    parser.add_argument("--gap-tolerance", type=int, default=DEFAULT_GAP_TOLERANCE)
    parser.add_argument("--cluster-tm", type=float, default=DEFAULT_CLUSTER_TM,
                        help="Union-Find clustering threshold")
    parser.add_argument("--edge-tm-low", type=float, default=DEFAULT_EDGE_TM_LOW)
    parser.add_argument("--edge-tm-high", type=float, default=DEFAULT_EDGE_TM_HIGH)
    parser.add_argument("--foldseek-cluster-tm", type=float, default=DEFAULT_FOLDSEEK_CLUSTER_TM,
                        help="Pre-clustering TM for Foldseek scalability")
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0,
                        help="CPU workers for domain extraction (0=all cores)")
    parser.add_argument("--no-esm2", action="store_true")
    parser.add_argument("--esm2-model", default="esm2_t12_35M_UR50D",
                        help="ESM-2 model name (default: 35M for speed)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--local-scratch", type=str, default=None,
                        help="Local NVMe path for fast temp writes (e.g. /usr/local/scratch)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    download_dir = data_dir / "raw" / "alphafold"
    tm_tag = f"tm{args.cluster_tm:.1f}".replace(".", "")
    edge_tag = f"e{args.edge_tm_low:.1f}-{args.edge_tm_high:.1f}".replace(".", "")
    output_tag = f"alphafold_{args.proteome}_{tm_tag}_{edge_tag}"
    domain_dir = data_dir / "processed" / f"alphafold_{args.proteome}_domains"
    foldseek_work = data_dir / "processed" / f"alphafold_{args.proteome}_foldseek"
    graph_dir = data_dir / "graphs" / output_tag

    logger.info("=" * 60)
    logger.info("AlphaFold Domain Graph Builder")
    logger.info("  Proteome: %s", args.proteome)
    logger.info("  pLDDT threshold: %.1f", args.plddt_threshold)
    logger.info("  Min domain length: %d", args.min_domain_length)
    logger.info("  Cluster TM: %.2f, Edge TM: [%.2f, %.2f)", args.cluster_tm,
                args.edge_tm_low, args.edge_tm_high)
    logger.info("  Output: %s", graph_dir)
    logger.info("=" * 60)

    # Step 1: Download
    af_dir = download_proteome(args.proteome, download_dir)

    # Step 2: Domain extraction (parallel, local scratch for speed)
    local_scratch = Path(args.local_scratch) if args.local_scratch else None
    metadata = process_alphafold_structures_parallel(
        af_dir, domain_dir,
        plddt_threshold=args.plddt_threshold,
        min_domain_length=args.min_domain_length,
        gap_tolerance=args.gap_tolerance,
        local_scratch=local_scratch,
        n_workers=args.workers,
    )

    n_domains = len(metadata["conformation_to_protein"])
    n_proteins = len(metadata["protein_sequences"])
    logger.info("Domains: %d from %d proteins", n_domains, n_proteins)

    if n_domains == 0:
        logger.error("No domains extracted — check pLDDT threshold and data")
        return

    # Use local scratch PDB dir if available (avoids NFS for Foldseek)
    actual_domain_dir = Path(metadata.get("_domain_pdb_dir", str(domain_dir)))
    logger.info("Domain PDBs at: %s", actual_domain_dir)

    # Foldseek work dir also on local scratch if available
    if local_scratch is not None:
        foldseek_work = local_scratch / "foldseek_work"
    foldseek_work.mkdir(parents=True, exist_ok=True)

    # Step 3: Foldseek cluster + search (with persistent NFS cache)
    sim_cache_path = data_dir / "processed" / f"alphafold_{args.proteome}_similarity_cache.npz"
    threedi_cache_path = data_dir / "processed" / f"alphafold_{args.proteome}_3di_seqs.json"

    if sim_cache_path.exists() and threedi_cache_path.exists():
        logger.info("[CACHED] Loading Foldseek results from NFS cache")
        cached = np.load(sim_cache_path, allow_pickle=True)
        all_names = cached["names"].tolist()
        rows, cols, scores = cached["rows"], cached["cols"], cached["scores"]
        similarities = {(int(r), int(c)): float(s) for r, c, s in zip(rows, cols, scores)}
        with open(threedi_cache_path) as f:
            threedi_seqs = json.load(f)
        logger.info("Loaded %d names, %d similarity pairs, %d 3Di seqs from cache",
                    len(all_names), len(similarities), len(threedi_seqs))
    else:
        all_names, similarities, threedi_seqs = foldseek_cluster_and_search(
            actual_domain_dir, foldseek_work,
            foldseek_bin=args.foldseek_bin,
            cluster_tm=args.foldseek_cluster_tm,
            threads=args.threads,
        )
        # Persist to NFS for future runs
        rows_arr = np.array([k[0] for k in similarities], dtype=np.int32)
        cols_arr = np.array([k[1] for k in similarities], dtype=np.int32)
        scores_arr = np.array([v for v in similarities.values()], dtype=np.float32)
        np.savez(sim_cache_path, names=np.array(all_names),
                 rows=rows_arr, cols=cols_arr, scores=scores_arr)
        with open(threedi_cache_path, "w") as f:
            json.dump(threedi_seqs, f)
        logger.info("Saved Foldseek cache to NFS: %s, %s", sim_cache_path, threedi_cache_path)

    # Step 4: Global clustering
    names, similarities = union_find_cluster(
        all_names, similarities,
        metadata["conformation_to_protein"],
        threedi_seqs,
        cluster_tm_threshold=args.cluster_tm,
    )

    # Step 5: Build graph
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

    # Save graph metadata
    graph_meta = {
        "proteome": args.proteome,
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

    logger.info("=" * 60)
    logger.info("DONE — Graph saved to %s", graph_dir)
    logger.info("  Nodes: %d", graph.num_nodes)
    logger.info("  Edges: %d", int(graph.edge_index.shape[1]))
    logger.info("  Proteins: %d", len(node_ranges))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
