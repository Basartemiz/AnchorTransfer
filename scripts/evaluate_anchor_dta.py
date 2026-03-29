from __future__ import annotations
#!/usr/bin/env python3
"""Anchor-based DTA vs sequence-only DTA experiment.

For each benchmark protein:
  1. Select 10 diverse conformations (k-medoids on RMSD)
  2. Find best anchor per conformation via Foldseek cross-similarity
  3. Run DeepDTA with anchor sequence (anchor-based) and own sequence (baseline)
  4. Compare metrics by protein type (IDP vs ordered)

Usage:
  python scripts/evaluate_anchor_dta.py \
    --domain-dir data/processed/alphafold_human_domains \
    --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
    --benchmark data/raw/benchmark_affinity.csv \
    --deepdta-model models/deepdta_fair/best_model.pt \
    --n-conformations 10 --device cuda
"""

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# from idr_gat.data.foldseek import compute_cross_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DeepDTA model (matches the saved checkpoint architecture)
# ---------------------------------------------------------------------------

CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25,
}
CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47,
    "L": 13, "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "[": 50,
    "T": 17, "]": 51, "V": 18, "Y": 19, "c": 20, "e": 21, "l": 22,
    "n": 23, "o": 24, "r": 25, "s": 26, "t": 27, "u": 28,
}
PROTEIN_MAX_LEN = 1000
SMILES_MAX_LEN = 100


def encode_protein(seq: str, max_len: int = PROTEIN_MAX_LEN) -> list[int]:
    return [CHARPROTSET.get(c, 0) for c in seq[:max_len]] + [0] * max(0, max_len - len(seq))


def encode_smiles(smi: str, max_len: int = SMILES_MAX_LEN) -> list[int]:
    return [CHARISOSMISET.get(c, 0) for c in smi[:max_len]] + [0] * max(0, max_len - len(smi))


class DeepDTAFair(nn.Module):
    """Architecture matching the deepdta_fair checkpoint (parallel multi-kernel CNN)."""

    def __init__(self, smiles_vocab=52, protein_vocab=26, embed_dim=128, num_filters=32):
        super().__init__()
        self.protein_embed = nn.Embedding(protein_vocab, embed_dim, padding_idx=0)
        self.smiles_embed = nn.Embedding(smiles_vocab, embed_dim, padding_idx=0)

        # 3 parallel convolutions with kernels 4, 6, 8
        self.protein_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])
        self.smiles_convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, k) for k in [4, 6, 8]
        ])

        # 3 branches × 32 filters × 2 modalities = 192
        self.fc1 = nn.Linear(num_filters * 3 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, smiles_seq, protein_seq):
        smiles = self.smiles_embed(smiles_seq).permute(0, 2, 1)
        smiles_outs = [F.relu(conv(smiles)).max(dim=2)[0] for conv in self.smiles_convs]
        smiles_feat = torch.cat(smiles_outs, dim=1)

        protein = self.protein_embed(protein_seq).permute(0, 2, 1)
        protein_outs = [F.relu(conv(protein)).max(dim=2)[0] for conv in self.protein_convs]
        protein_feat = torch.cat(protein_outs, dim=1)

        x = torch.cat([smiles_feat, protein_feat], dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.out(x).squeeze(-1)


def load_deepdta_fair(model_path: str, device: str = "cpu") -> DeepDTAFair:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = DeepDTAFair()
    sd = checkpoint["model_state_dict"]
    # Map short key names (pe/se/pc/sc) to DeepDTAFair names if needed
    key_map = {
        "pe.weight": "protein_embed.weight",
        "se.weight": "smiles_embed.weight",
    }
    for i in range(3):
        key_map[f"pc.{i}.weight"] = f"protein_convs.{i}.weight"
        key_map[f"pc.{i}.bias"] = f"protein_convs.{i}.bias"
        key_map[f"sc.{i}.weight"] = f"smiles_convs.{i}.weight"
        key_map[f"sc.{i}.bias"] = f"smiles_convs.{i}.bias"
    if any(k in sd for k in key_map):
        sd = {key_map.get(k, k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Step 1: Conformation selection
# ---------------------------------------------------------------------------

def _parse_ca_coords(pdb_path: Path) -> np.ndarray | None:
    """Parse Cα coordinates from a PDB file."""
    coords = []
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    if not coords:
        return None
    return np.array(coords, dtype=np.float32)


def select_diverse_conformations(
    conformation_dirs: list[Path],
    k: int = 10,
) -> list[Path]:
    """Select k diverse conformations via k-medoids on pairwise Cα RMSD.

    If fewer than k conformations available, returns all.
    """
    if len(conformation_dirs) <= k:
        return conformation_dirs

    # Parse Cα coordinates from each PDB
    coords_list = []
    valid_dirs = []
    for d in conformation_dirs:
        if isinstance(d, Path) and d.is_file() and d.suffix == ".pdb":
            pdb_path = d
        elif isinstance(d, Path) and d.is_dir():
            pdb_files = sorted(d.glob("*.pdb"))
            if not pdb_files:
                continue
            pdb_path = pdb_files[0]
        else:
            continue
        coords = _parse_ca_coords(pdb_path)
        if coords is not None and len(coords) > 0:
            coords_list.append(coords)
            valid_dirs.append(d)

    if len(coords_list) <= k:
        return valid_dirs

    # Compute pairwise RMSD matrix
    n = len(coords_list)
    rmsd_matrix = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            min_len = min(len(coords_list[i]), len(coords_list[j]))
            ci = coords_list[i][:min_len]
            cj = coords_list[j][:min_len]
            rmsd = np.sqrt(np.mean(np.sum((ci - cj) ** 2, axis=1)))
            rmsd_matrix[i, j] = rmsd
            rmsd_matrix[j, i] = rmsd

    # K-medoids (greedy: pick point that maximizes min-distance to selected)
    selected = [np.argmax(np.sum(rmsd_matrix, axis=1))]
    for _ in range(k - 1):
        min_dists = rmsd_matrix[:, selected].min(axis=1)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))

    return [valid_dirs[i] for i in selected]


# ---------------------------------------------------------------------------
# Step 2: Anchor finding via Foldseek
# ---------------------------------------------------------------------------

def find_anchors_for_protein(
    conformation_paths: list[Path],
    domain_pdb_dir: Path,
    domain_metadata: dict,
    target_db_path: str | None = None,
    foldseek_bin: str = "foldseek",
    tm_threshold: float = 0.4,
    threads: int = 8,
) -> list[dict]:
    """Find best anchor for each conformation via fast Foldseek search.

    Uses Foldseek search (3Di prefilter, fast) instead of full TM-align.
    If target_db_path is provided, reuses the pre-built Foldseek database.

    Returns list of dicts with keys: conformation, anchor_domain, anchor_uniprot,
    anchor_sequence, anchor_tm.
    """
    import subprocess

    with tempfile.TemporaryDirectory(dir=None) as tmp_dir:
        query_dir = Path(tmp_dir) / "query"
        query_dir.mkdir()
        conf_name_map = {}
        for i, conf_path in enumerate(conformation_paths):
            if isinstance(conf_path, Path) and conf_path.is_file():
                src = conf_path
            elif isinstance(conf_path, Path) and conf_path.is_dir():
                pdb_files = sorted(conf_path.glob("*.pdb"))
                if not pdb_files:
                    continue
                src = pdb_files[0]
            else:
                continue
            dst_name = f"conf_{i:03d}.pdb"
            shutil.copy2(src, query_dir / dst_name)
            conf_name_map[f"conf_{i:03d}"] = conf_path

        if not any(query_dir.glob("*.pdb")):
            return []

        # Create query database
        query_db = Path(tmp_dir) / "querydb"
        subprocess.run([
            foldseek_bin, "createdb", str(query_dir), str(query_db),
        ], check=True, capture_output=True)

        # Use pre-built target DB or create one
        if target_db_path and Path(str(target_db_path) + ".dbtype").exists():
            target_db = target_db_path
        else:
            target_db = Path(tmp_dir) / "targetdb"
            subprocess.run([
                foldseek_bin, "createdb", str(domain_pdb_dir), str(target_db),
            ], check=True, capture_output=True)

        # Fast search (3Di prefilter + alignment type 2)
        result_path = Path(tmp_dir) / "result"
        tsv_path = Path(tmp_dir) / "alignment.tsv"

        subprocess.run([
            foldseek_bin, "search",
            str(query_db), str(target_db), str(result_path),
            str(Path(tmp_dir) / "tmp"),
            "-a",
            "--alignment-type", "1",
            "--max-accept", "10000",
            "-e", "inf",
            "-s", "9.5",
            "--max-seqs", "10000",
            "--threads", str(threads),
        ], check=True, capture_output=True)

        subprocess.run([
            foldseek_bin, "convertalis",
            str(query_db), str(target_db), str(result_path), str(tsv_path),
            "--format-output", "query,target,alntmscore",
        ], check=True, capture_output=True)

        # Parse results
        import pandas as pd
        top_k_hits = {}  # qname -> [(target, tm_score), ...]

        if tsv_path.exists() and tsv_path.stat().st_size > 0:
            df = pd.read_csv(tsv_path, sep="\t", header=None,
                             names=["query", "target", "tmscore"])
            for _, row in df.iterrows():
                qname = row["query"]
                try:
                    tm = float(row["tmscore"])
                except (ValueError, TypeError):
                    continue
                if qname not in top_k_hits:
                    top_k_hits[qname] = []
                top_k_hits[qname].append((row["target"], tm))

    conf_to_protein = domain_metadata["conformation_to_protein"]
    protein_sequences = domain_metadata["protein_sequences"]

    results = []
    for qname, hits in top_k_hits.items():
        conf_path = conf_name_map.get(qname)
        if conf_path is None:
            continue

        # Sort by TM descending, take top 5 above threshold
        hits_sorted = sorted(hits, key=lambda x: x[1], reverse=True)
        seen_uniprots = set()
        k_count = 0
        for anchor_domain, best_tm in hits_sorted:
            if best_tm < tm_threshold:
                break
            if k_count >= 5:
                break
            anchor_uniprot = conf_to_protein.get(anchor_domain, "unknown")
            if anchor_uniprot in seen_uniprots:
                continue
            seen_uniprots.add(anchor_uniprot)
            anchor_seq = protein_sequences.get(anchor_uniprot, "")
            results.append({
                "conformation": str(conf_path),
                "anchor_domain": anchor_domain,
                "anchor_uniprot": anchor_uniprot,
                "anchor_sequence": anchor_seq,
                "anchor_tm": best_tm,
            })
            k_count += 1


    return results


# ---------------------------------------------------------------------------
# Step 3: DeepDTA inference
# ---------------------------------------------------------------------------

def predict_deepdta(
    model: DeepDTAFair,
    protein_seq: str,
    smiles_list: list[str],
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Run DeepDTA inference for one protein against multiple drugs.

    Returns array of predicted pKi values, shape (len(smiles_list),).
    """
    protein_enc = encode_protein(protein_seq)
    protein_tensor = torch.tensor([protein_enc], dtype=torch.long, device=device)

    all_preds = []
    for i in range(0, len(smiles_list), batch_size):
        batch_smiles = smiles_list[i:i + batch_size]
        smiles_encs = [encode_smiles(s) for s in batch_smiles]
        smiles_tensor = torch.tensor(smiles_encs, dtype=torch.long, device=device)
        protein_batch = protein_tensor.expand(len(batch_smiles), -1)

        with torch.no_grad():
            preds = model(smiles_tensor, protein_batch)
        all_preds.append(preds.cpu().numpy().flatten())

    return np.concatenate(all_preds)


def predict_anchor_based(
    model: DeepDTAFair,
    anchors: list[dict],
    smiles_list: list[str],
    device: str = "cpu",
) -> np.ndarray | None:
    """Run DeepDTA with each anchor's sequence, return TM-weighted average.

    Returns array of predicted pKi values, or None if no valid anchors.
    """
    if not anchors:
        return None

    weighted_sum = np.zeros(len(smiles_list), dtype=np.float64)
    tm_sum = 0.0

    for anchor in anchors:
        seq = anchor["anchor_sequence"]
        tm = anchor["anchor_tm"]
        if not seq:
            continue
        preds = predict_deepdta(model, seq, smiles_list, device=device)
        w = np.exp(2.0 * tm)  # exponential weighting
        weighted_sum += w * preds
        tm_sum += w

    if tm_sum == 0:
        return None

    return (weighted_sum / tm_sum).astype(np.float32)


# ---------------------------------------------------------------------------
# Step 4: Main evaluation loop
# ---------------------------------------------------------------------------

def load_idrome_index(idrome_index_path: Path) -> dict[str, list[Path]]:
    """Load pre-built IDRome conformation index."""
    with open(idrome_index_path) as f:
        raw = json.load(f)
    return {uid: [Path(p) for p in paths] for uid, paths in raw.items()}


def extract_trajectory_conformations(
    conf_dirs: list[Path],
    n_frames: int = 100,
    scratch_dir: Path | None = None,
) -> list[Path]:
    """Extract n_frames equally-spaced from traj.xtc trajectories as Cα PDBs.

    Returns list of PDB file paths (written to scratch_dir or temp dir).
    """
    import MDAnalysis as mda

    if scratch_dir is None:
        scratch_dir = Path(tempfile.mkdtemp(dir=None))
    scratch_dir.mkdir(parents=True, exist_ok=True)

    all_pdbs = []
    for conf_dir in conf_dirs:
        top_pdb = conf_dir / "top.pdb" if conf_dir.is_dir() else conf_dir
        traj_xtc = conf_dir / "traj.xtc" if conf_dir.is_dir() else conf_dir.parent / "traj.xtc"

        if not traj_xtc.exists():
            # No trajectory — just use top.pdb
            if top_pdb.exists():
                all_pdbs.append(top_pdb)
            continue

        try:
            u = mda.Universe(str(top_pdb), str(traj_xtc))
            total_frames = len(u.trajectory)
            ca = u.select_atoms("name CA")

            per_region = max(1, n_frames // len(conf_dirs))
            if total_frames <= per_region:
                indices = list(range(total_frames))
            else:
                indices = np.linspace(0, total_frames - 1, per_region, dtype=int).tolist()

            region_name = conf_dir.name if conf_dir.is_dir() else conf_dir.stem
            for i, frame_idx in enumerate(indices):
                u.trajectory[frame_idx]
                pdb_path = scratch_dir / f"{region_name}_f{i:03d}.pdb"
                with open(pdb_path, "w") as f:
                    for j, atom in enumerate(ca):
                        pos = atom.position
                        resname = atom.resname
                        f.write(
                            f"ATOM  {j+1:5d}  CA  {resname:3s} A{j+1:4d}    "
                            f"{pos[0]:8.3f}{pos[1]:8.3f}{pos[2]:8.3f}  1.00  0.00\n"
                        )
                    f.write("END\n")
                all_pdbs.append(pdb_path)
        except Exception as e:
            logger.debug("Failed to extract %s: %s", conf_dir, e)
            if top_pdb.exists():
                all_pdbs.append(top_pdb)

    return all_pdbs


def main():
    parser = argparse.ArgumentParser(description="Anchor-based DTA experiment")
    parser.add_argument("--domain-dir", type=str, required=True,
                        help="AlphaFold domain PDB directory")
    parser.add_argument("--domain-metadata", type=str, required=True,
                        help="domain_metadata.json path")
    parser.add_argument("--benchmark", type=str, default="data/raw/benchmark_affinity.csv")
    parser.add_argument("--deepdta-model", type=str, default="models/deepdta_fair/best_model.pt")
    parser.add_argument("--idrome-index", type=str, default="data/processed/idrome_conformation_index.json")
    parser.add_argument("--sequences", type=str, default="data/processed/uniprot_sequences.json")
    parser.add_argument("--n-conformations", type=int, default=10)
    parser.add_argument("--anchor-tm-threshold", type=float, default=0.4)
    parser.add_argument("--target-db", type=str, default=None,
                        help="Pre-built Foldseek target DB path (skips createdb)")
    parser.add_argument("--foldseek-bin", type=str, default="foldseek")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="results/alphafold_anchor_dta")
    parser.add_argument("--max-proteins", type=int, default=0,
                        help="Limit to first N proteins (0=all, useful for testing)")
    parser.add_argument("--max-drugs", type=int, default=0,
                        help="Random sample N drugs per protein (0=all)")
    parser.add_argument("--disorder-analysis", action="store_true",
                        help="Compute disorder score and report per-quartile metrics + TM×disorder heatmap")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    domain_pdb_dir = Path(args.domain_dir)

    # Load IDRome conformation index (instant lookup vs slow rglob)
    logger.info("Loading IDRome conformation index...")
    idrome_index = load_idrome_index(Path(args.idrome_index))
    logger.info("IDRome index: %d proteins, %d total conformations",
                len(idrome_index), sum(len(v) for v in idrome_index.values()))

    # Load domain metadata
    logger.info("Loading domain metadata...")
    with open(args.domain_metadata) as f:
        domain_metadata = json.load(f)

    # Load protein sequences (for sequence-only baseline)
    logger.info("Loading protein sequences...")
    with open(args.sequences) as f:
        uniprot_sequences = json.load(f)

    # Load benchmark
    logger.info("Loading benchmark...")
    benchmark_df = pd.read_csv(args.benchmark)
    proteins = benchmark_df.groupby("uniprot_id").first().reset_index()
    logger.info("Benchmark: %d proteins (%d IDPs, %d ordered), %d pairs",
                len(proteins),
                (proteins["protein_type"] == "idp").sum(),
                (proteins["protein_type"] == "ordered").sum(),
                len(benchmark_df))

    # Load DeepDTA
    logger.info("Loading DeepDTA model from %s...", args.deepdta_model)
    model = load_deepdta_fair(args.deepdta_model, device=args.device)

    # Foldseek target database (reused for all proteins)
    import subprocess
    if args.target_db and Path(args.target_db + ".dbtype").exists():
        target_db = args.target_db
        logger.info("[PRE-BUILT] Using Foldseek target DB at %s", target_db)
    else:
        target_db_dir = domain_pdb_dir.parent / (domain_pdb_dir.name + "_foldseek_db")
        target_db_dir.mkdir(parents=True, exist_ok=True)
        target_db = str(target_db_dir / "structdb")
        if not Path(target_db + ".dbtype").exists():
            logger.info("Building Foldseek target database for %s...", domain_pdb_dir)
            subprocess.run([
                args.foldseek_bin, "createdb", str(domain_pdb_dir), target_db,
            ], check=True, capture_output=True)
            logger.info("Target database built.")
        else:
            logger.info("[CACHED] Foldseek target database at %s", target_db)

    # Main evaluation loop
    all_rows = []
    protein_summaries = []

    protein_groups = list(benchmark_df.groupby("uniprot_id"))
    if args.max_proteins > 0:
        protein_groups = protein_groups[:args.max_proteins]
        logger.info("Limiting to first %d proteins (test mode)", args.max_proteins)

    for idx, (uniprot_id, group) in enumerate(protein_groups):
        protein_type = group["protein_type"].iloc[0]
        # Optionally subsample drugs for speed
        if args.max_drugs > 0 and len(group) > args.max_drugs:
            group = group.sample(n=args.max_drugs, random_state=42)
        smiles_list = group["ligand_smiles"].tolist()
        true_pki = group["pki"].values
        own_seq = uniprot_sequences.get(uniprot_id, "")

        logger.info("[%d/%d] %s (%s) — %d drugs",
                    idx + 1, len(protein_groups), uniprot_id, protein_type, len(smiles_list))

        # --- Sequence-only baseline ---
        if own_seq:
            pred_seq = predict_deepdta(model, own_seq, smiles_list, device=args.device)
        else:
            logger.warning("  No sequence for %s, skipping sequence-only", uniprot_id)
            pred_seq = np.full(len(smiles_list), np.nan)

        # --- Anchor-based ---
        conf_dirs = idrome_index.get(uniprot_id, [])
        selected = []
        n_anchors = 0
        mean_tm = 0.0

        if not conf_dirs:
            logger.warning("  No IDRome conformations for %s", uniprot_id)
            pred_anchor = np.full(len(smiles_list), np.nan)
        else:
            # Check if dirs have PDB files directly (fallback conformations)
            direct_pdbs = []
            traj_dirs = []
            for cd in conf_dirs:
                cd_path = Path(cd)
                if cd_path.is_dir() and not (cd_path / "traj.xtc").exists():
                    direct_pdbs.extend(sorted(cd_path.glob("*.pdb")))
                else:
                    traj_dirs.append(cd_path)

            if direct_pdbs:
                all_frames = direct_pdbs
                logger.info("  Found %d direct PDB conformations", len(all_frames))
            elif traj_dirs:
                all_frames = extract_trajectory_conformations(
                    traj_dirs, n_frames=100,
                    scratch_dir=Path(tempfile.mkdtemp(dir=None)),
                )
                logger.info("  Extracted %d frames from %d regions", len(all_frames), len(traj_dirs))
            else:
                all_frames = []
                logger.warning("  No valid conformations for %s", uniprot_id)
            # Wrap each PDB as its own "dir" for select_diverse_conformations
            frame_dirs = [f.parent if f.name != "top.pdb" else f.parent for f in all_frames]
            # For diverse selection, treat each PDB individually
            selected = select_diverse_conformations(
                [f for f in all_frames],  # pass PDB files directly
                k=args.n_conformations,
            )
            logger.info("  Selected %d diverse conformations", len(selected))

            anchors = find_anchors_for_protein(
                selected, domain_pdb_dir, domain_metadata,
                target_db_path=target_db,
                foldseek_bin=args.foldseek_bin,
                tm_threshold=args.anchor_tm_threshold,
                threads=args.threads,
            )
            n_anchors = len(anchors)
            mean_tm = np.mean([a["anchor_tm"] for a in anchors]) if anchors else 0.0
            logger.info("  Found %d anchors (mean TM=%.3f)", n_anchors, mean_tm)

            pred_anchor = predict_anchor_based(model, anchors, smiles_list, device=args.device)
            if pred_anchor is None:
                pred_anchor = np.full(len(smiles_list), np.nan)

        # Collect rows
        for i in range(len(smiles_list)):
            all_rows.append({
                "uniprot_id": uniprot_id,
                "ligand_smiles": smiles_list[i],
                "true_pki": true_pki[i],
                "pred_sequence_only": pred_seq[i],
                "pred_anchor_based": pred_anchor[i],
                "protein_type": protein_type,
            })

        # --- Per-protein CI comparison (live) ---
        from scipy.stats import pearsonr, spearmanr

        def _ci(y_true, y_pred):
            n = len(y_true)
            if n < 2:
                return float("nan")
            concordant = 0
            total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if y_true[i] != y_true[j]:
                        total += 1
                        if (y_true[i] - y_true[j]) * (y_pred[i] - y_pred[j]) > 0:
                            concordant += 1
                        elif y_pred[i] == y_pred[j]:
                            concordant += 0.5
            return concordant / total if total > 0 else float("nan")

        # Compute per-protein metrics for this protein
        valid_seq = ~np.isnan(pred_seq)
        valid_anc = ~np.isnan(pred_anchor)
        ci_seq = _ci(true_pki[valid_seq], pred_seq[valid_seq]) if valid_seq.sum() >= 2 else float("nan")
        ci_anc = _ci(true_pki[valid_anc], pred_anchor[valid_anc]) if valid_anc.sum() >= 2 else float("nan")

        mse_seq = float(np.mean((true_pki[valid_seq] - pred_seq[valid_seq]) ** 2)) if valid_seq.sum() > 0 else float("nan")
        mse_anc = float(np.mean((true_pki[valid_anc] - pred_anchor[valid_anc]) ** 2)) if valid_anc.sum() > 0 else float("nan")

        logger.info("  >>> %s (%s, %d drugs): seq_CI=%.3f  anc_CI=%.3f  delta=%.3f | seq_MSE=%.3f  anc_MSE=%.3f",
                    uniprot_id, protein_type, len(smiles_list),
                    ci_seq, ci_anc, ci_anc - ci_seq,
                    mse_seq, mse_anc)

        protein_summaries.append({
            "uniprot_id": uniprot_id,
            "protein_type": protein_type,
            "n_drugs": len(smiles_list),
            "n_conformations": len(conf_dirs),
            "n_selected": len(selected),
            "n_anchors": n_anchors,
            "mean_anchor_tm": mean_tm,
            "ci_seq": ci_seq,
            "ci_anchor": ci_anc,
            "mse_seq": mse_seq,
            "mse_anchor": mse_anc,
        })

        # Running averages by protein type
        if len(protein_summaries) % 5 == 0 or idx == len(protein_groups) - 1:
            ps_df = pd.DataFrame(protein_summaries)
            for ptype in ["idp", "ordered"]:
                sub = ps_df[ps_df["protein_type"] == ptype]
                if len(sub) == 0:
                    continue
                valid = sub.dropna(subset=["ci_seq", "ci_anchor"])
                if len(valid) == 0:
                    continue
                mean_ci_seq = valid["ci_seq"].mean()
                mean_ci_anc = valid["ci_anchor"].mean()
                mean_mse_seq = valid["mse_seq"].mean()
                mean_mse_anc = valid["mse_anchor"].mean()
                logger.info("  === RUNNING AVG [%s] (n=%d): seq_CI=%.3f  anc_CI=%.3f  delta=%.3f | seq_MSE=%.3f  anc_MSE=%.3f ===",
                            ptype.upper(), len(valid),
                            mean_ci_seq, mean_ci_anc, mean_ci_anc - mean_ci_seq,
                            mean_mse_seq, mean_mse_anc)

    # Save predictions
    pred_df = pd.DataFrame(all_rows)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    logger.info("Saved predictions: %s", output_dir / "predictions.csv")

    # Save protein summary
    summary_df = pd.DataFrame(protein_summaries)
    summary_df.to_csv(output_dir / "protein_summary.csv", index=False)

    # Compute metrics
    from scripts.affinity_eval_utils import compute_regression_metrics

    metrics = {}
    for label, mask in [("all", pred_df["protein_type"].notna()),
                        ("idp", pred_df["protein_type"] == "idp"),
                        ("ordered", pred_df["protein_type"] == "ordered")]:
        subset = pred_df[mask].copy()

        # Sequence-only metrics
        seq_eval = subset.dropna(subset=["pred_sequence_only"]).copy()
        seq_eval = seq_eval[["uniprot_id", "ligand_smiles", "true_pki", "pred_sequence_only"]].copy()
        seq_eval.columns = ["uniprot_id", "ligand_smiles", "true_pki", "pred_pki"]
        seq_metrics = compute_regression_metrics(seq_eval)

        # Anchor-based metrics
        anc_eval = subset.dropna(subset=["pred_anchor_based"]).copy()
        anc_eval = anc_eval[["uniprot_id", "ligand_smiles", "true_pki", "pred_anchor_based"]].copy()
        anc_eval.columns = ["uniprot_id", "ligand_smiles", "true_pki", "pred_pki"]
        anc_metrics = compute_regression_metrics(anc_eval)

        metrics[label] = {
            "sequence_only": seq_metrics,
            "anchor_based": anc_metrics,
        }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for label in ["idp", "ordered", "all"]:
        seq_ci = metrics[label]["sequence_only"].get("ci", "N/A")
        anc_ci = metrics[label]["anchor_based"].get("ci", "N/A")
        logger.info("  %s: seq_CI=%.4f  anchor_CI=%.4f  delta=%.4f",
                    label.upper(), float(seq_ci), float(anc_ci),
                    float(anc_ci) - float(seq_ci))
    logger.info("=" * 60)

    # Disorder entropy analysis
    if args.disorder_analysis:
        from idr_gat.evaluation.disorder_score import (
            threedi_entropy as _threedi_entropy, compute_disorder_score, bin_by_disorder,
        )
        from idr_gat.data.foldseek import encode_3di
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        logger.info("Computing disorder scores...")
        idp_summary = summary_df[summary_df["protein_type"] == "idp"].copy()

        disorder_scores = {}
        for _, row in idp_summary.iterrows():
            uid = row["uniprot_id"]
            conf_dirs = idrome_index.get(uid, [])
            if not conf_dirs:
                continue

            pdb_paths = []
            for cd in conf_dirs:
                cd_path = Path(cd)
                if cd_path.is_dir():
                    if not (cd_path / "traj.xtc").exists():
                        pdb_paths.extend(sorted(cd_path.glob("*.pdb")))
                    elif (cd_path / "top.pdb").exists():
                        pdb_paths.append(cd_path / "top.pdb")

            if len(pdb_paths) < 2:
                continue

            import shutil
            with tempfile.TemporaryDirectory() as tmp:
                tmp_pdb_dir = Path(tmp) / "pdbs"
                tmp_pdb_dir.mkdir()
                for i, p in enumerate(pdb_paths[:20]):
                    shutil.copy2(p, tmp_pdb_dir / f"conf_{i:03d}.pdb")
                threedi_seqs = encode_3di(tmp_pdb_dir, foldseek_bin=args.foldseek_bin)
                if len(threedi_seqs) >= 2:
                    ent = _threedi_entropy(list(threedi_seqs.values()))
                    plddt_proxy = 1.0 - min(row.get("mean_anchor_tm", 0.5), 1.0)
                    disorder_scores[uid] = compute_disorder_score(ent, 1.0 - plddt_proxy)

        if disorder_scores:
            idp_summary["disorder_score"] = idp_summary["uniprot_id"].map(disorder_scores)
            scored = idp_summary.dropna(subset=["disorder_score", "ci_anchor", "ci_seq"])

            bins = bin_by_disorder(
                {row["uniprot_id"]: row["disorder_score"] for _, row in scored.iterrows()},
                n_bins=4,
            )
            logger.info("=" * 60)
            logger.info("DISORDER QUARTILE ANALYSIS (IDPs)")
            logger.info("=" * 60)
            for label, uids in bins.items():
                sub = scored[scored["uniprot_id"].isin(uids)]
                if len(sub) == 0:
                    continue
                sci = sub["ci_seq"].mean()
                aci = sub["ci_anchor"].mean()
                smse = sub["mse_seq"].mean()
                amse = sub["mse_anchor"].mean()
                dm = (amse - smse) / smse * 100 if smse > 0 else 0
                ds_lo, ds_hi = sub["disorder_score"].min(), sub["disorder_score"].max()
                logger.info("  %s (n=%d, ds=%.2f-%.2f): dCI=%+.3f  dMSE=%+.1f%%",
                           label, len(sub), ds_lo, ds_hi, aci - sci, dm)

            # 2D Heatmap: TM bins × Disorder bins
            scored_with_tm = scored[scored["mean_anchor_tm"] > 0]
            if len(scored_with_tm) >= 8:
                tm_vals = scored_with_tm["mean_anchor_tm"].values
                ds_vals = scored_with_tm["disorder_score"].values
                dmse_vals = ((scored_with_tm["mse_anchor"] - scored_with_tm["mse_seq"]) /
                             scored_with_tm["mse_seq"].clip(lower=0.01) * 100).values

                tm_edges = np.percentile(tm_vals, [0, 25, 50, 75, 100])
                ds_edges = np.percentile(ds_vals, [0, 25, 50, 75, 100])

                heatmap = np.full((4, 4), np.nan)
                counts = np.zeros((4, 4), dtype=int)

                for tm, ds, dmse in zip(tm_vals, ds_vals, dmse_vals):
                    ti = min(int(np.searchsorted(tm_edges[1:], tm, side="right")), 3)
                    di = min(int(np.searchsorted(ds_edges[1:], ds, side="right")), 3)
                    if np.isnan(heatmap[di, ti]):
                        heatmap[di, ti] = 0.0
                    heatmap[di, ti] += dmse
                    counts[di, ti] += 1

                mask = counts > 0
                heatmap[mask] /= counts[mask]

                fig, ax = plt.subplots(figsize=(8, 6))
                im = ax.imshow(heatmap, cmap="RdYlGn_r", aspect="auto", origin="lower")
                ax.set_xticks(range(4))
                ax.set_xticklabels([f"TM Q{i+1}" for i in range(4)])
                ax.set_yticks(range(4))
                ax.set_yticklabels([f"DS Q{i+1}" for i in range(4)])
                ax.set_xlabel("Anchor TM-score Quartile")
                ax.set_ylabel("Disorder Score Quartile")
                ax.set_title("dMSE% by Anchor TM x Disorder Score (IDPs)")
                for i in range(4):
                    for j in range(4):
                        if counts[i, j] > 0:
                            ax.text(j, i, f"{heatmap[i,j]:+.0f}%\nn={counts[i,j]}",
                                   ha="center", va="center", fontsize=9)
                plt.colorbar(im, label="dMSE%")
                plt.tight_layout()
                fig.savefig(output_dir / "tm_disorder_heatmap.pdf", dpi=150)
                fig.savefig(output_dir / "tm_disorder_heatmap.png", dpi=150)
                plt.close()
                logger.info("Saved heatmap: %s", output_dir / "tm_disorder_heatmap.png")

            scored.to_csv(output_dir / "disorder_scores.csv", index=False)
            logger.info("Saved disorder scores: %s", output_dir / "disorder_scores.csv")

    logger.info("DONE — Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
