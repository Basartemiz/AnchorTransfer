# src/idr_gat/data/foldseek.py
import subprocess
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Foldseek 3Di structural alphabet (20 states)
THREEDI_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
THREEDI_TO_IDX = {c: i for i, c in enumerate(THREEDI_ALPHABET)}


PADDING_IDX = 20  # index used for padding in sequences


def parse_3di_to_indices(seq_3di: str) -> np.ndarray:
    """Convert a 3Di string to integer index array.

    Each character maps to 0-19 (the 20 3Di structural states).
    Unknown characters default to 0.

    Returns:
        np.ndarray of shape (seq_len,) with dtype int64
    """
    return np.array(
        [THREEDI_TO_IDX.get(c.upper(), 0) for c in seq_3di],
        dtype=np.int64,
    )


def parse_3di_to_features(seq_3di: str) -> np.ndarray:
    """Convert a 3Di string to one-hot encoded feature matrix.

    Kept for backward compatibility. Prefer parse_3di_to_indices() with
    learned embeddings for better representational quality.

    Returns:
        np.ndarray of shape (seq_len, 20) -- one-hot encoded
    """
    n = len(seq_3di)
    features = np.zeros((n, 20), dtype=np.float32)
    for i, c in enumerate(seq_3di):
        idx = THREEDI_TO_IDX.get(c.upper(), 0)
        features[i, idx] = 1.0
    return features


def encode_3di(
    pdb_dir: Path,
    foldseek_bin: str = "foldseek",
    output_dir: Path | None = None,
) -> dict[str, str]:
    """Run Foldseek to extract 3Di sequences for all PDBs in a directory.

    Args:
        pdb_dir: directory containing PDB files
        foldseek_bin: path to foldseek binary
        output_dir: where to write the Foldseek database

    Returns:
        dict mapping filename -> 3Di sequence string
    """
    if output_dir is None:
        output_dir = pdb_dir.parent / (pdb_dir.name + "_foldseek_db")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "structdb"
    tsv_path = output_dir / "3di_seqs.tsv"

    if tsv_path.exists() and tsv_path.stat().st_size > 0:
        results = {}
        current_name = None
        current_seq = []
        with open(tsv_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name is not None:
                        results[current_name] = "".join(current_seq)
                    current_name = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)
        if current_name is not None:
            results[current_name] = "".join(current_seq)
        if results:
            return results

    subprocess.run([
        foldseek_bin, "createdb", str(pdb_dir), str(db_path),
    ], check=True, capture_output=True)

    # Link header DB so convert2fasta can read sequence names
    subprocess.run([
        foldseek_bin, "lndb", str(db_path) + "_h", str(db_path) + "_ss_h",
    ], check=True, capture_output=True)

    subprocess.run([
        foldseek_bin, "convert2fasta", str(db_path) + "_ss", str(tsv_path),
    ], check=True, capture_output=True)

    results = {}
    current_name = None
    current_seq = []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name is not None:
                    results[current_name] = "".join(current_seq)
                current_name = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_name is not None:
        results[current_name] = "".join(current_seq)

    return results


def compute_pairwise_similarity(
    pdb_dir: Path,
    foldseek_bin: str = "foldseek",
    output_dir: Path | None = None,
) -> tuple[list[str], dict[tuple[int, int], float]]:
    """Compute all-vs-all structural similarity using Foldseek (sparse).

    Returns:
        names: list of structure names
        similarities: sparse dict mapping (i, j) -> TM-score (i < j only)
    """
    if output_dir is None:
        output_dir = pdb_dir.parent / (pdb_dir.name + "_foldseek_aln")
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / "structdb"
    result_path = output_dir / "result"
    tsv_path = output_dir / "alignment.tsv"

    subprocess.run([
        foldseek_bin, "createdb", str(pdb_dir), str(db_path),
    ], check=True, capture_output=True)

    subprocess.run([
        foldseek_bin, "search",
        str(db_path), str(db_path), str(result_path),
        str(output_dir / "tmp"),
        "-a",
        "--max-accept", "300",
    ], check=True, capture_output=True)

    subprocess.run([
        foldseek_bin, "convertalis",
        str(db_path), str(db_path), str(result_path), str(tsv_path),
        "--format-output", "query,target,alntmscore",
    ], check=True, capture_output=True)

    import pandas as pd

    all_pdb_files = sorted(pdb_dir.glob("*.pdb"))
    all_names = sorted(p.stem for p in all_pdb_files)

    if tsv_path.stat().st_size == 0:
        logger.warning("Foldseek alignment file is empty — no hits found")
        return all_names, {}

    df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "tmscore"])

    hit_names = set(df["query"].tolist() + df["target"].tolist())
    missing = set(all_names) - hit_names
    if missing:
        logger.warning(
            f"{len(missing)} structures had no Foldseek alignments and will "
            f"appear as isolated nodes: {sorted(missing)[:5]}..."
        )
    names = sorted(set(all_names) | hit_names)
    name_to_idx = {n: i for i, n in enumerate(names)}

    similarities: dict[tuple[int, int], float] = {}
    for _, row in df.iterrows():
        qi = name_to_idx[row["query"]]
        ti = name_to_idx[row["target"]]
        if qi == ti:
            continue
        key = (min(qi, ti), max(qi, ti))
        score = float(row["tmscore"])
        similarities[key] = max(similarities.get(key, 0.0), score)

    return names, similarities


def compute_cross_similarity(
    query_dir: Path,
    target_dir: Path,
    foldseek_bin: str = "foldseek",
    output_dir: Path | None = None,
    threads: int | None = None,
) -> tuple[list[str], list[str], np.ndarray]:
    """Compute query-vs-target structural similarity using Foldseek.

    Much faster than all-vs-all when only cross-similarities are needed
    (e.g. 10 IDP conformations vs 1,898 global graph nodes).

    Returns:
        query_names: list of query structure names
        target_names: list of target structure names
        cross_matrix: (n_query, n_target) matrix of TM-scores
    """
    if output_dir is None:
        output_dir = query_dir.parent / f"{query_dir.name}_cross_aln_{target_dir.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    query_db = output_dir / "querydb"
    target_db_dir = target_dir.parent / (target_dir.name + "_foldseek_db")
    target_db_dir.mkdir(parents=True, exist_ok=True)
    target_db = target_db_dir / "structdb"
    result_path = output_dir / "result"
    tsv_path = output_dir / "alignment.tsv"

    query_names = sorted(p.stem for p in query_dir.glob("*.pdb"))
    target_names = sorted(p.stem for p in target_dir.glob("*.pdb"))
    q_idx = {n: i for i, n in enumerate(query_names)}
    t_idx = {n: i for i, n in enumerate(target_names)}

    if tsv_path.exists():
        cross_matrix = np.zeros((len(query_names), len(target_names)), dtype=np.float32)
        if tsv_path.stat().st_size > 0:
            import pandas as pd

            df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "tmscore"])
            for _, row in df.iterrows():
                qi = q_idx.get(row["query"])
                ti = t_idx.get(row["target"])
                if qi is not None and ti is not None:
                    cross_matrix[qi, ti] = max(cross_matrix[qi, ti], float(row["tmscore"]))
        else:
            logger.warning("Foldseek cross-alignment file is cached but empty — no hits found")
        return query_names, target_names, cross_matrix

    # Create separate databases
    subprocess.run([
        foldseek_bin, "createdb", str(query_dir), str(query_db),
    ], check=True, capture_output=True)

    target_db_marker = Path(str(target_db) + ".dbtype")
    if not target_db_marker.exists():
        subprocess.run([
            foldseek_bin, "createdb", str(target_dir), str(target_db),
        ], check=True, capture_output=True)

    # Search query against target (NOT all-vs-all)
    search_cmd = [
        foldseek_bin, "search",
        str(query_db), str(target_db), str(result_path),
        str(output_dir / "tmp"),
        "-a",
        "--alignment-type", "1",
    ]
    if threads is not None and threads > 0:
        search_cmd.extend(["--threads", str(int(threads))])
    subprocess.run(search_cmd, check=True, capture_output=True)

    subprocess.run([
        foldseek_bin, "convertalis",
        str(query_db), str(target_db), str(result_path), str(tsv_path),
        "--format-output", "query,target,alntmscore",
    ], check=True, capture_output=True)

    import pandas as pd

    cross_matrix = np.zeros((len(query_names), len(target_names)), dtype=np.float32)

    if tsv_path.stat().st_size > 0:
        df = pd.read_csv(tsv_path, sep="\t", header=None, names=["query", "target", "tmscore"])
        for _, row in df.iterrows():
            qi = q_idx.get(row["query"])
            ti = t_idx.get(row["target"])
            if qi is not None and ti is not None:
                cross_matrix[qi, ti] = max(cross_matrix[qi, ti], float(row["tmscore"]))
    else:
        logger.warning("Foldseek cross-alignment file is empty — no hits found")

    return query_names, target_names, cross_matrix
