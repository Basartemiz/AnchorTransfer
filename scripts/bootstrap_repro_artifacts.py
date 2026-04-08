#!/usr/bin/env python3
"""Bootstrap raw reproduction artifacts and optionally reuse cached files.

This helper makes the paper-reproduction prerequisites explicit:
1. Reuse exact artifact filenames from local search roots when available.
2. Download the public DrugTargetCommons bulk export when missing.
3. Derive protein-sequence CSVs needed by `reproduce/01_prepare_data.sh`.

The generated CSVs use the column contract expected by the numbered flow:
`uniprot_id,sequence`. For benchmark proteins, `uniprot_id` is the benchmark
identifier used by the benchmark CSV itself, which may be a UniProt accession
or a dataset-specific protein label such as Davis kinase names.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import ssl
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

csv.field_size_limit(sys.maxsize)

DTC_URL = "https://drugtargetcommons.fimm.fi/static/Excell_files/DTC_data.csv"
ZENODO_DTC_URL = "https://zenodo.org/records/15108794/files/DTC_data.csv.gz?download=1"
ZENODO_ARTIFACTS_URL = "https://zenodo.org/records/19442952/files"
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"

RAW_ARTIFACTS = {
    "DTC_data.csv": Path("data/raw/DTC_data.csv"),
    "dtc_proteins.csv": Path("data/raw/dtc_proteins.csv"),
    "benchmark_proteins.csv": Path("data/raw/benchmark_proteins.csv"),
}

OPTIONAL_PRECOMPUTED = {
    "dtc_training_interactions.csv": Path("data/processed/dtc_training_interactions.csv"),
    "esm2_35m_benchmark.pt": Path("data/processed/esm2_35m_benchmark.pt"),
    "esm2_35m_dtc.pt": Path("data/processed/esm2_35m_dtc.pt"),
    "esm2_650m_benchmark.pt": Path("data/processed/esm2_650m_benchmark.pt"),
    # Legacy filename — copy as esm2_35m_dtc.pt if found
    "esm2_35m_dtc_proteins_full.pt": Path("data/processed/esm2_35m_dtc.pt"),
    # BDB artifacts
    "bindingdb_interactions.csv": Path("data/processed/bindingdb_interactions.csv"),
    "merged_sequences.json": Path("data/processed/merged_sequences.json"),
}

BENCHMARK_PATHS = [
    Path("data/raw/davis_benchmark.csv"),
    Path("data/raw/metz_benchmark.csv"),
    Path("data/raw/glass_benchmark.csv"),
    Path("data/raw/bdb_ki_benchmark.csv"),
    Path("data/raw/davis/davis_benchmark.csv"),
    Path("data/raw/metz/metz_benchmark.csv"),
]

ID_COLUMNS = [
    "uniprot_id",
    "protein_name",
    "target_uniprot_id",
    "Target_ID",
    "target_id",
]
SEQUENCE_COLUMNS = [
    "sequence",
    "protein_sequence",
    "target_sequence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap reproduction artifacts")
    parser.add_argument("--repo-root", default=".", help="Repository root")
    parser.add_argument(
        "--search-root",
        action="append",
        default=[],
        help="Directory to search recursively for exact artifact filenames",
    )
    parser.add_argument(
        "--dtc-url",
        default=DTC_URL,
        help="Public URL for the DrugTargetCommons bulk export",
    )
    parser.add_argument(
        "--skip-search",
        action="store_true",
        help="Do not search local roots for existing artifact files",
    )
    parser.add_argument(
        "--skip-dtc-download",
        action="store_true",
        help="Do not download DTC_data.csv when it is missing",
    )
    parser.add_argument(
        "--skip-uniprot",
        action="store_true",
        help="Do not fetch missing protein sequences from UniProt",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def build_ssl_context(verify: bool) -> ssl.SSLContext:
    if not verify:
        return ssl._create_unverified_context()
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


def read_url_bytes(url: str, timeout: int = 60) -> bytes:
    last_error: Exception | None = None
    for verify in (True, False):
        try:
            with urlopen(url, timeout=timeout, context=build_ssl_context(verify)) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if verify:
                logger.warning("Verified TLS fetch failed for %s; retrying without certificate verification", url)
    if last_error is None:
        raise RuntimeError(f"Failed to fetch {url}")
    raise last_error


def copy_exact_artifact(destination: Path, search_roots: list[Path]) -> Path | None:
    for root in search_roots:
        if not root.exists():
            continue
        try:
            matches = sorted(root.rglob(destination.name))
        except OSError as exc:
            logger.warning("Skipping search root %s: %s", root, exc)
            continue
        for candidate in matches:
            if not candidate.is_file():
                continue
            try:
                if candidate.resolve() == destination.resolve():
                    continue
            except OSError:
                pass
            ensure_parent(destination)
            shutil.copy2(candidate, destination)
            logger.info("Copied %s from %s", destination.name, candidate)
            return candidate
    return None


def maybe_reuse_exact_files(repo_root: Path, search_roots: list[Path]) -> None:
    for relative_path in [*RAW_ARTIFACTS.values(), *OPTIONAL_PRECOMPUTED.values()]:
        destination = repo_root / relative_path
        if destination.exists():
            continue
        copy_exact_artifact(destination, search_roots)


def maybe_decompress_gzip(payload: bytes) -> bytes:
    if payload[:2] == b"\x1f\x8b":
        return gzip.decompress(payload)
    return payload


def download_dtc_if_needed(destination: Path, url: str) -> None:
    if destination.exists():
        return
    ensure_parent(destination)
    candidate_urls = [url]
    if url == DTC_URL:
        candidate_urls.append(ZENODO_DTC_URL)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    last_error: Exception | None = None
    for candidate_url in candidate_urls:
        try:
            logger.info("Downloading %s from %s", destination.name, candidate_url)
            tmp_path.write_bytes(maybe_decompress_gzip(read_url_bytes(candidate_url)))
            tmp_path.replace(destination)
            logger.info("Saved %s", destination)
            return
        except Exception as exc:
            last_error = exc
            logger.warning("Download failed from %s: %s", candidate_url, exc)
    if last_error is None:
        raise RuntimeError(f"Failed to download {destination.name}")
    raise last_error


def infer_target_column(fieldnames: list[str]) -> str | None:
    lowered = {name.lower(): name for name in fieldnames}
    for candidate in ("target_id", "uniprot_id"):
        if candidate in lowered:
            return lowered[candidate]
    for field in fieldnames:
        lower = field.lower()
        if "target_id" in lower or "uniprot" in lower:
            return field
    return None


def looks_like_uniprot(value: str) -> bool:
    value = value.strip()
    return bool(
        re.fullmatch(r"(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9])(?:-\d+)?", value)
    )


def split_identifier_field(value: str) -> list[str]:
    return [part for part in re.split(r"[\s,;]+", value.strip()) if part]


def unique_ids_from_dtc(dtc_csv: Path) -> list[str]:
    with dtc_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{dtc_csv} has no header row")
        target_column = infer_target_column(reader.fieldnames)
        if target_column is None:
            raise ValueError(f"Could not infer target column from {dtc_csv}")
        ids: set[str] = set()
        for row in reader:
            value = (row.get(target_column) or "").strip()
            if value:
                for candidate in split_identifier_field(value):
                    if looks_like_uniprot(candidate):
                        ids.add(candidate)
    result = sorted(ids)
    logger.info("Collected %d unique DTC protein identifiers", len(result))
    return result


def extract_sequences_from_benchmark_csv(path: Path) -> tuple[dict[str, str], set[str]]:
    sequences: dict[str, str] = {}
    unresolved: set[str] = set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return sequences, unresolved
        id_column = next((name for name in ID_COLUMNS if name in reader.fieldnames), None)
        seq_column = next((name for name in SEQUENCE_COLUMNS if name in reader.fieldnames), None)
        if id_column is None:
            logger.warning("Skipping %s: no supported protein identifier column", path)
            return sequences, unresolved
        for row in reader:
            protein_id = (row.get(id_column) or "").strip()
            if not protein_id:
                continue
            sequence = (row.get(seq_column) or "").strip() if seq_column else ""
            if sequence:
                sequences.setdefault(protein_id, sequence)
            else:
                for candidate in split_identifier_field(protein_id):
                    if looks_like_uniprot(candidate):
                        unresolved.add(candidate)
    logger.info(
        "Parsed %s: %d embedded benchmark sequences, %d unresolved accessions",
        path,
        len(sequences),
        len(unresolved),
    )
    return sequences, unresolved


def extract_glass_sequences(repo_root: Path) -> dict[str, str]:
    path = repo_root / "data/raw/glass/glass2_sequences.json"
    if not path.exists():
        return {}
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        logger.warning("Skipping %s: expected a JSON object", path)
        return {}
    sequences = {
        str(protein_id): sequence.strip()
        for protein_id, sequence in payload.items()
        if isinstance(sequence, str) and sequence.strip()
    }
    logger.info("Loaded %d GLASS benchmark sequences from %s", len(sequences), path)
    return sequences


def fetch_uniprot_sequences(accessions: Iterable[str], batch_size: int = 50) -> dict[str, str]:
    accession_list = sorted({acc for acc in accessions if acc})
    fetched: dict[str, str] = {}
    for start in range(0, len(accession_list), batch_size):
        batch = accession_list[start:start + batch_size]
        query = " OR ".join(f"(accession:{acc})" for acc in batch)
        params = urlencode(
            {
                "fields": "accession,sequence",
                "format": "tsv",
                "size": str(len(batch)),
                "query": query,
            }
        )
        url = f"{UNIPROT_SEARCH_URL}?{params}"
        for attempt in range(3):
            try:
                payload = read_url_bytes(url, timeout=60).decode("utf-8")
                break
            except (HTTPError, URLError) as exc:
                if attempt == 2:
                    raise RuntimeError(f"UniProt fetch failed for batch starting at {start}: {exc}") from exc
                time.sleep(1.5 * (attempt + 1))
        rows = list(csv.DictReader(payload.splitlines(), delimiter="\t"))
        for row in rows:
            accession = (row.get("Entry") or "").strip()
            sequence = (row.get("Sequence") or "").strip()
            if accession and sequence:
                fetched[accession] = sequence
        logger.info("Fetched %d/%d UniProt sequences", len(fetched), len(accession_list))
        time.sleep(0.1)
    return fetched


def write_sequence_csv(destination: Path, sequences: dict[str, str]) -> None:
    ensure_parent(destination)
    frame = pd.DataFrame(
        sorted(sequences.items()),
        columns=["uniprot_id", "sequence"],
    )
    frame.to_csv(destination, index=False)
    logger.info("Wrote %d sequences to %s", len(frame), destination)


def build_dtc_proteins(repo_root: Path, skip_uniprot: bool) -> None:
    destination = repo_root / RAW_ARTIFACTS["dtc_proteins.csv"]
    if destination.exists():
        return
    dtc_csv = repo_root / RAW_ARTIFACTS["DTC_data.csv"]
    protein_ids = unique_ids_from_dtc(dtc_csv)
    sequences: dict[str, str] = {}
    if not skip_uniprot:
        sequences = fetch_uniprot_sequences(protein_ids)
    missing = len(protein_ids) - len(sequences)
    if missing:
        logger.warning("Could not resolve %d DTC protein identifiers to sequences", missing)
    if not sequences:
        raise RuntimeError("Unable to build data/raw/dtc_proteins.csv without any resolved sequences")
    write_sequence_csv(destination, sequences)


def build_benchmark_proteins(repo_root: Path, skip_uniprot: bool) -> None:
    destination = repo_root / RAW_ARTIFACTS["benchmark_proteins.csv"]
    if destination.exists():
        return

    sequences = extract_glass_sequences(repo_root)
    unresolved: set[str] = set()

    for relative_path in BENCHMARK_PATHS:
        path = repo_root / relative_path
        if not path.exists():
            continue
        csv_sequences, csv_unresolved = extract_sequences_from_benchmark_csv(path)
        for protein_id, sequence in csv_sequences.items():
            sequences.setdefault(protein_id, sequence)
        unresolved.update(csv_unresolved - set(sequences))

    if unresolved and not skip_uniprot:
        fetched = fetch_uniprot_sequences(unresolved)
        for protein_id, sequence in fetched.items():
            sequences.setdefault(protein_id, sequence)
        unresolved -= set(fetched)

    if unresolved:
        logger.warning("Could not resolve %d benchmark protein identifiers", len(unresolved))
    if not sequences:
        raise RuntimeError("Unable to build data/raw/benchmark_proteins.csv from available benchmark sources")
    write_sequence_csv(destination, sequences)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    search_roots = [Path(root).expanduser().resolve() for root in args.search_root]

    if not args.skip_search and search_roots:
        maybe_reuse_exact_files(repo_root, search_roots)

    dtc_csv = repo_root / RAW_ARTIFACTS["DTC_data.csv"]
    if not dtc_csv.exists():
        if args.skip_dtc_download:
            raise RuntimeError("DTC_data.csv is missing and --skip-dtc-download was set")
        download_dtc_if_needed(dtc_csv, args.dtc_url)

    build_dtc_proteins(repo_root, skip_uniprot=args.skip_uniprot)
    build_benchmark_proteins(repo_root, skip_uniprot=args.skip_uniprot)

    missing_precomputed = [
        str(relative_path)
        for relative_path in OPTIONAL_PRECOMPUTED.values()
        if not (repo_root / relative_path).exists()
    ]
    if missing_precomputed:
        logger.info(
            "Precomputed embeddings still missing: %s",
            ", ".join(missing_precomputed),
        )
        logger.info("Run bash reproduce/01_prepare_data.sh to generate them.")

    logger.info("Artifact bootstrap complete")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - CLI error path
        logger.error("%s", exc)
        raise SystemExit(1)
