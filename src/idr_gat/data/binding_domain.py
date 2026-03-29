"""Identify which domain of a protein is the drug-binding domain.

Priority: UniProt binding site annotations → largest domain fallback.
"""
from __future__ import annotations

import json
import logging
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def pdb_to_seq(pdb_path: str) -> str:
    """Extract amino acid sequence from domain PDB CA atoms."""
    seq = []
    seen = set()
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") and line[12:16].strip() == "CA":
                    resnum = int(line[22:26].strip())
                    if resnum not in seen:
                        seen.add(resnum)
                        seq.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    except (FileNotFoundError, IOError):
        pass
    return "".join(seq)


def find_domain_range(domain_seq: str, full_seq: str) -> Optional[tuple[int, int]]:
    """Find where a domain sequence maps in the full protein sequence.

    Returns (start, end) 0-indexed inclusive, or None if not found.
    """
    idx = full_seq.find(domain_seq)
    if idx >= 0:
        return (idx, idx + len(domain_seq) - 1)
    # Approximate match (allow up to 3 mismatches)
    for start in range(len(full_seq) - len(domain_seq) + 1):
        sub = full_seq[start : start + len(domain_seq)]
        mismatches = sum(1 for a, b in zip(sub, domain_seq) if a != b)
        if mismatches <= 3:
            return (start, start + len(domain_seq) - 1)
    return None


def parse_uniprot_binding_sites(features: list[dict]) -> list[tuple[int, int]]:
    """Extract binding/active site residue ranges from UniProt feature list."""
    binding_types = {"Binding site", "Active site"}
    sites = []
    for feat in features:
        if feat.get("type") in binding_types:
            loc = feat.get("location", {})
            start = loc.get("start", {}).get("value")
            end = loc.get("end", {}).get("value")
            if start is not None and end is not None:
                sites.append((int(start), int(end)))
    return sites


def fetch_uniprot_binding_sites(uniprot_id: str) -> list[tuple[int, int]]:
    """Fetch binding site annotations from UniProt REST API."""
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        features = data.get("features", [])
        return parse_uniprot_binding_sites(features)
    except Exception as e:
        logger.debug("UniProt fetch failed for %s: %s", uniprot_id, e)
        return []


def select_binding_domain(
    domains: dict[str, dict],
    binding_sites: list[tuple[int, int]],
) -> str:
    """Select the binding domain from a dict of domain info.

    Args:
        domains: {domain_name: {"seq": str, "range": (start, end), "length": int}}
        binding_sites: [(start, end)] 1-indexed residue ranges from UniProt
    """
    if binding_sites:
        best_domain = None
        best_overlap = -1
        for name, info in domains.items():
            d_start, d_end = info["range"]
            overlap = 0
            for bs_start, bs_end in binding_sites:
                bs_s = bs_start - 1  # convert 1-indexed to 0-indexed
                bs_e = bs_end - 1
                ov_start = max(d_start, bs_s)
                ov_end = min(d_end, bs_e)
                if ov_end >= ov_start:
                    overlap += ov_end - ov_start + 1
            if overlap > best_overlap:
                best_overlap = overlap
                best_domain = name
        if best_domain and best_overlap > 0:
            return best_domain

    return max(domains, key=lambda k: domains[k]["length"])


def identify_binding_domain(
    uniprot_id: str,
    domain_pdbs: dict[str, str],
    full_sequence: str,
) -> tuple[str, str, str]:
    """Identify the binding domain for a protein.

    Returns (domain_name, domain_sequence, method).
    """
    domains = {}
    for name, pdb_path in domain_pdbs.items():
        dom_seq = pdb_to_seq(pdb_path)
        if not dom_seq:
            continue
        rng = find_domain_range(dom_seq, full_sequence)
        if rng is None:
            rng = (0, len(dom_seq) - 1)
        domains[name] = {"seq": dom_seq, "range": rng, "length": len(dom_seq)}

    if not domains:
        return ("", "", "none")

    binding_sites = fetch_uniprot_binding_sites(uniprot_id)
    method = "uniprot" if binding_sites else "largest"

    best = select_binding_domain(domains, binding_sites)
    return (best, domains[best]["seq"], method)
