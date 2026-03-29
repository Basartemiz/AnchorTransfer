#!/usr/bin/env python3
"""Train AffinityGAT on AlphaFold graph — same anchor aggregation as eval.

For each protein:
  1. Extract conformations from IDRome trajectories (or use graph node if in-graph)
  2. Find anchor per conformation via Foldseek
  3. GATv2 full graph → read out each anchor → predict pKi per anchor
  4. TM-weighted aggregation → final prediction
  5. MSE loss vs true pKi

Usage:
  PYTHONPATH=src:. python scripts/train_affinity_gat.py \
    --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
    --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
    --benchmark data/raw/benchmark_affinity.csv \
    --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
    --domain-dir /usr/local/scratch/alphafold_domains \
    --target-db /usr/local/scratch/foldseek_work/domaindb \
    --idrome-index data/processed/idrome_conformation_index.json \
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from idr_gat.model.affinity_gat import AffinityGAT, encode_smiles
from scripts.evaluate_anchor_dta import (
    find_anchors_for_protein,
    extract_trajectory_conformations,
    select_diverse_conformations,
    load_idrome_index,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_nma_conformations(
    uniprot_id: str,
    sequences: dict,
    scratch_dir: Path,
    n_conformations: int = 10,
) -> list[Path]:
    """Generate NMA conformations for an ordered protein via AlphaFold + ANM.

    Downloads AlphaFold structure, extracts Cα, runs ANM, writes PDBs.
    """
    from idr_gat.data.alphafold import download_alphafold_pdb, extract_ca_coords_and_plddt
    from idr_gat.data.nma import generate_anm_conformations

    scratch_dir.mkdir(parents=True, exist_ok=True)
    af_pdb = scratch_dir / f"{uniprot_id}_af.pdb"

    # Download AlphaFold structure
    if not af_pdb.exists():
        ok = download_alphafold_pdb(uniprot_id, af_pdb)
        if not ok:
            return []

    # Extract Cα coords
    try:
        coords, plddt, seq = extract_ca_coords_and_plddt(af_pdb)
    except Exception:
        return []

    if len(coords) < 10:
        return []

    # Generate ANM conformations
    conformations = generate_anm_conformations(
        coords, n_modes=5, amplitudes=[1.0, 2.0, 3.0],
        n_conformations=n_conformations,
    )
    if len(conformations) == 0:
        return []

    # Write PDBs
    AA3 = "ALA"  # simplified — Cα only needs residue name
    pdbs = []
    for ci, conf_coords in enumerate(conformations):
        pdb_path = scratch_dir / f"{uniprot_id}_nma_{ci:03d}.pdb"
        with open(pdb_path, "w") as f:
            for j in range(len(conf_coords)):
                x, y, z = conf_coords[j]
                f.write(
                    f"ATOM  {j+1:5d}  CA  {AA3:3s} A{j+1:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
                )
            f.write("END\n")
        pdbs.append(pdb_path)
    return pdbs


def precompute_protein_anchors(
    train_df: pd.DataFrame,
    node_ranges: dict,
    domain_metadata: dict,
    idrome_index: dict,
    domain_pdb_dir: str,
    target_db: str | None = None,
    foldseek_bin: str = "foldseek",
    tm_threshold: float = 0.4,
    threads: int = 8,
    n_conformations: int = 10,
    sequences: dict | None = None,
) -> dict[str, list[dict]]:
    """Pre-compute anchors for all training proteins.

    For IDPs: extract IDRome trajectory frames → Foldseek → anchors.
    For ordered: download AlphaFold → NMA → Foldseek → anchors.
    Every protein finds anchors in the graph — none need to be in the graph.

    Returns: {uniprot_id: [{"anchor_node": int, "anchor_tm": float}, ...]}
    """
    protein_anchors = {}
    all_uids = train_df["uniprot_id"].unique()
    protein_types = train_df.groupby("uniprot_id")["protein_type"].first() if "protein_type" in train_df.columns else {}

    # Fast path: proteins directly in graph use their own node
    n_direct = 0
    need_foldseek = []
    for uniprot_id in all_uids:
        if uniprot_id in node_ranges:
            start, _ = node_ranges[uniprot_id]
            protein_anchors[uniprot_id] = [{"anchor_node": start, "anchor_tm": 1.0}]
            n_direct += 1
        else:
            need_foldseek.append(uniprot_id)
    logger.info("  Direct graph nodes: %d, need Foldseek: %d", n_direct, len(need_foldseek))

    # Parallel Foldseek anchor finding for remaining proteins
    if need_foldseek:
        import multiprocessing as mp
        import concurrent.futures
        n_workers = min(mp.cpu_count(), 32)
        logger.info("  Finding anchors for %d proteins with %d parallel workers...", len(need_foldseek), n_workers)

        # Use ThreadPoolExecutor (avoids pickling issues, Foldseek is I/O bound anyway)
        def _find_one(uid):
            try:
                ptype = protein_types.get(uid, "ordered")
                if ptype == "idp" and uid in idrome_index:
                    conf_dirs = idrome_index[uid]
                    frames = extract_trajectory_conformations(
                        conf_dirs, n_frames=100,
                        scratch_dir=Path(tempfile.mkdtemp(dir="/workspace/tmp")),
                    )
                else:
                    frames = generate_nma_conformations(
                        uid, sequences or {},
                        scratch_dir=Path(tempfile.mkdtemp(dir="/workspace/tmp")),
                        n_conformations=n_conformations * 3,
                    )
                if not frames:
                    return uid, None
                sel = select_diverse_conformations(frames, k=n_conformations)
                ancs = find_anchors_for_protein(
                    sel, Path(domain_pdb_dir), domain_metadata,
                    target_db_path=target_db,
                    foldseek_bin=foldseek_bin,
                    tm_threshold=tm_threshold,
                    threads=2,
                )
                if ancs:
                    result = []
                    for a in ancs:
                        a_uid = a["anchor_uniprot"]
                        if a_uid in node_ranges:
                            s, _ = node_ranges[a_uid]
                            result.append({"anchor_node": s, "anchor_tm": a["anchor_tm"]})
                    return uid, result if result else None
                return uid, None
            except Exception:
                return uid, None

        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_find_one, uid): uid for uid in need_foldseek}
            done_count = 0
            for future in concurrent.futures.as_completed(futures):
                uid, anchor_list = future.result()
                if anchor_list:
                    protein_anchors[uid] = anchor_list
                done_count += 1
                if done_count % 100 == 0:
                    logger.info("  Foldseek: %d/%d done (%d with anchors)",
                                done_count, len(need_foldseek), len(protein_anchors) - n_direct)

    logger.info("Anchor precompute done: %d/%d proteins have anchors",
                len(protein_anchors), len(all_uids))
    return protein_anchors


def prepare_training_data(
    benchmark_df: pd.DataFrame,
    protein_anchors: dict[str, list[dict]],
    val_fraction: float = 0.2,
    seed: int = 42,
    max_drugs: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Split by protein, each item has protein's anchor list + drug + pki."""
    valid_proteins = [p for p in benchmark_df["uniprot_id"].unique() if p in protein_anchors]
    rng = np.random.RandomState(seed)
    protein_types = benchmark_df.groupby("uniprot_id")["protein_type"].first()

    val_proteins = set()
    for ptype in ["idp", "ordered"]:
        prots = [p for p in valid_proteins if protein_types.get(p) == ptype]
        n_val = max(1, int(len(prots) * val_fraction))
        rng.shuffle(prots)
        val_proteins.update(prots[:n_val])

    train_data, val_data = [], []
    for uid in valid_proteins:
        group = benchmark_df[benchmark_df["uniprot_id"] == uid]
        if max_drugs > 0 and len(group) > max_drugs:
            group = group.sample(n=max_drugs, random_state=seed)
        anchors = protein_anchors[uid]
        for _, row in group.iterrows():
            item = {
                "uniprot_id": uid,
                "smiles": row["ligand_smiles"],
                "pki": float(row["pki"]),
                "anchors": anchors,  # list of {anchor_node, anchor_tm}
            }
            if uid in val_proteins:
                val_data.append(item)
            else:
                train_data.append(item)

    logger.info("Train: %d pairs (%d proteins), Val: %d pairs (%d proteins)",
                len(train_data), len(set(d["uniprot_id"] for d in train_data)),
                len(val_data), len(set(d["uniprot_id"] for d in val_data)))
    return train_data, val_data


def prebuild_flat_tensors(data_items: list[dict], smiles_lookup: dict[str, int]):
    """Pre-build flat tensors for batched training.

    For each item with N anchors, emits N rows in flat arrays.
    Returns dict with:
      - smiles_indices: (total_rows,) int — index into pre-encoded SMILES tensor
      - anchor_nodes: (total_rows,) int — graph node index
      - tm_weights: (total_rows,) float — TM-score weight
      - targets: (total_rows,) float — pKi target (same for all anchors of one item)
      - item_indices: (total_rows,) int — which item this row belongs to
      - n_items: int — total number of valid items
    """
    smiles_idx_list = []
    anchor_node_list = []
    tm_weight_list = []
    target_list = []
    item_idx_list = []
    item_count = 0

    for item in data_items:
        anchors = item["anchors"]
        smi = item["smiles"]
        if smi not in smiles_lookup:
            continue
        s_idx = smiles_lookup[smi]
        has_valid = False
        for anc in anchors:
            smiles_idx_list.append(s_idx)
            anchor_node_list.append(anc["anchor_node"])
            tm_weight_list.append(anc["anchor_tm"])
            target_list.append(item["pki"])
            item_idx_list.append(item_count)
            has_valid = True
        if has_valid:
            item_count += 1

    # Build item->row index for fast batch gathering (avoids np.isin)
    item_row_ranges = {}  # item_id -> (start, end) into flat arrays
    if item_idx_list:
        cur_item = item_idx_list[0]
        cur_start = 0
        for i in range(1, len(item_idx_list)):
            if item_idx_list[i] != cur_item:
                item_row_ranges[cur_item] = (cur_start, i)
                cur_item = item_idx_list[i]
                cur_start = i
        item_row_ranges[cur_item] = (cur_start, len(item_idx_list))

    return {
        "smiles_indices": np.array(smiles_idx_list, dtype=np.int64),
        "anchor_nodes": np.array(anchor_node_list, dtype=np.int64),
        "tm_weights": np.array(tm_weight_list, dtype=np.float32),
        "targets": np.array(target_list, dtype=np.float32),
        "item_indices": np.array(item_idx_list, dtype=np.int64),
        "item_row_ranges": item_row_ranges,
        "n_items": item_count,
    }


def train_epoch(model, graph, train_data, optimizer, device, batch_size=64,
                scaler=None, smiles_tensor=None, smiles_lookup=None,
                epoch=0, max_epochs=200, proteins_per_batch=4,
                pos_per_protein=8, hard_neg_per_protein=8, mid_per_protein=4,
                pos_threshold=7.0, neg_threshold=5.0):
    model.train()

    # Pre-compute graph node embeddings ONCE per epoch (detached)
    with torch.no_grad():
        graph_dev = graph.to(device) if graph.x_3di.device != device else graph
        node_embs = model.encode_graph(graph_dev)  # (N, hidden_dim)

    # Multi-task path: protein-centric batching with InfoNCE + MSE
    use_multitask = hasattr(model, 'proj_head') and model.proj_head is not None
    if use_multitask:
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        from idr_gat.model.affinity_gat import infonce_loss, curriculum_weights

        sampler = ProteinBatchSampler(
            train_data, proteins_per_batch=proteins_per_batch,
            pos_per_protein=pos_per_protein, hard_neg_per_protein=hard_neg_per_protein,
            mid_per_protein=mid_per_protein, pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )

        total_loss, total_mse, total_nce, n_batches = 0.0, 0.0, 0.0, 0

        for batch in sampler:
            anchor_list, smi_list, tgt_list, lbl_list = [], [], [], []
            for item in batch:
                smi = item["smiles"]
                if smi not in smiles_lookup:
                    continue
                anc = item["anchors"][0]
                anchor_list.append(anc["anchor_node"])
                smi_list.append(smiles_lookup[smi])
                tgt_list.append(item["pki"])
                lbl_list.append(item["label"])

            if not anchor_list:
                continue

            b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
            b_smi_enc = smiles_tensor[torch.tensor(smi_list, dtype=torch.long, device=device)]
            b_targets = torch.tensor(tgt_list, dtype=torch.float32, device=device)
            b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

            prot_emb = node_embs[b_anchors]
            drug_emb = model.drug_encoder(b_smi_enc)
            interaction = model.cross_attn(prot_emb, drug_emb)

            preds = model.head(interaction).squeeze(-1)
            mse = F.mse_loss(preds, b_targets)

            proj = F.normalize(model.proj_head(interaction), dim=1)
            pos_mask = b_labels == 1
            neg_mask = b_labels == 0
            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                nce = infonce_loss(proj[pos_mask], proj[pos_mask], proj[neg_mask])
            else:
                nce = torch.tensor(0.0, device=device)

            alpha, beta = curriculum_weights(epoch, max_epochs)
            loss = alpha * mse + beta * nce

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_mse += mse.item()
            total_nce += nce.item()
            n_batches += 1

        alpha, beta = curriculum_weights(epoch, max_epochs)
        logger.info("  [curriculum] alpha=%.2f beta=%.2f | mse=%.4f nce=%.4f total=%.4f",
                    alpha, beta,
                    total_mse / max(n_batches, 1),
                    total_nce / max(n_batches, 1),
                    total_loss / max(n_batches, 1))
        return total_loss / max(n_batches, 1)

    # Fallback: original MSE-only training
    # Pre-build flat tensors for ALL training data
    flat = prebuild_flat_tensors(train_data, smiles_lookup)
    if flat["n_items"] == 0:
        return 0.0

    # Shuffle at item level, then gather flat rows
    item_perm = np.random.permutation(flat["n_items"])
    total_loss = 0.0
    n_batches = 0

    row_ranges = flat["item_row_ranges"]

    for start in range(0, len(item_perm), batch_size):
        batch_item_ids = item_perm[start:start + batch_size]

        # Gather flat rows via pre-built index (O(batch_size) not O(n_total))
        row_slices = []
        item_map = {}  # old_item_id -> new_0based_id
        for item_id in batch_item_ids:
            if item_id in row_ranges:
                s, e = row_ranges[item_id]
                item_map[item_id] = len(item_map)
                row_slices.append((s, e, len(item_map) - 1))
        if not row_slices:
            continue

        # Build contiguous arrays for this batch
        b_smi_list, b_anc_list, b_tm_list, b_tgt_list, b_iid_list = [], [], [], [], []
        for s, e, new_id in row_slices:
            n = e - s
            b_smi_list.append(flat["smiles_indices"][s:e])
            b_anc_list.append(flat["anchor_nodes"][s:e])
            b_tm_list.append(flat["tm_weights"][s:e])
            b_tgt_list.append(flat["targets"][s:e])
            b_iid_list.append(np.full(n, new_id, dtype=np.int64))

        b_smi_idx = torch.tensor(np.concatenate(b_smi_list), dtype=torch.long, device=device)
        b_anc_nodes = torch.tensor(np.concatenate(b_anc_list), dtype=torch.long, device=device)
        b_tm = torch.tensor(np.concatenate(b_tm_list), dtype=torch.float32, device=device)
        b_targets_flat = torch.tensor(np.concatenate(b_tgt_list), dtype=torch.float32, device=device)
        inverse = torch.tensor(np.concatenate(b_iid_list), dtype=torch.long, device=device)

        n_items_batch = len(row_slices)

        # Batched forward: drug encoder + cross-attention + head
        smi_enc = smiles_tensor[b_smi_idx]  # (R, 100)
        drug_emb = model.drug_encoder(smi_enc)  # (R, hidden)
        prot_emb = node_embs[b_anc_nodes]  # (R, hidden)
        interaction = model.cross_attn(prot_emb, drug_emb)  # (R, hidden)
        raw_pred = model.head(interaction).squeeze(-1)  # (R,)

        # Scatter-reduce: TM-weighted average per item
        weighted_pred = raw_pred * b_tm
        sum_pred = torch.zeros(n_items_batch, device=device)
        sum_tm = torch.zeros(n_items_batch, device=device)
        sum_pred.scatter_add_(0, inverse, weighted_pred)
        sum_tm.scatter_add_(0, inverse, b_tm)
        agg_pred = sum_pred / sum_tm.clamp(min=1e-8)

        # Get one target per item (all anchors of same item have same target)
        agg_target = torch.zeros(n_items_batch, device=device)
        agg_target.scatter_(0, inverse, b_targets_flat)  # last write wins, all same

        optimizer.zero_grad()
        loss = F.mse_loss(agg_pred, agg_target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, graph, val_data, device, batch_size=512,
             smiles_tensor=None, smiles_lookup=None):
    model.eval()
    graph_dev = graph.to(device) if graph.x_3di.device != device else graph
    node_embs = model.encode_graph(graph_dev)

    flat = prebuild_flat_tensors(val_data, smiles_lookup)
    if flat["n_items"] == 0:
        return 0.0

    total_loss = 0.0
    n_batches = 0

    row_ranges = flat["item_row_ranges"]

    for start in range(0, flat["n_items"], batch_size):
        batch_item_ids = np.arange(start, min(start + batch_size, flat["n_items"]))

        # Gather flat rows via pre-built index
        row_slices = []
        for item_id in batch_item_ids:
            if item_id in row_ranges:
                s, e = row_ranges[item_id]
                row_slices.append((s, e, len(row_slices)))
        if not row_slices:
            continue

        b_smi_list, b_anc_list, b_tm_list, b_tgt_list, b_iid_list = [], [], [], [], []
        for s, e, new_id in row_slices:
            n = e - s
            b_smi_list.append(flat["smiles_indices"][s:e])
            b_anc_list.append(flat["anchor_nodes"][s:e])
            b_tm_list.append(flat["tm_weights"][s:e])
            b_tgt_list.append(flat["targets"][s:e])
            b_iid_list.append(np.full(n, new_id, dtype=np.int64))

        b_smi_idx = torch.tensor(np.concatenate(b_smi_list), dtype=torch.long, device=device)
        b_anc_nodes = torch.tensor(np.concatenate(b_anc_list), dtype=torch.long, device=device)
        b_tm = torch.tensor(np.concatenate(b_tm_list), dtype=torch.float32, device=device)
        b_targets_flat = torch.tensor(np.concatenate(b_tgt_list), dtype=torch.float32, device=device)
        inverse = torch.tensor(np.concatenate(b_iid_list), dtype=torch.long, device=device)
        n_items_batch = len(row_slices)

        smi_enc = smiles_tensor[b_smi_idx]
        drug_emb = model.drug_encoder(smi_enc)
        prot_emb = node_embs[b_anc_nodes]
        interaction = model.cross_attn(prot_emb, drug_emb)
        raw_pred = model.head(interaction).squeeze(-1)

        weighted_pred = raw_pred * b_tm
        sum_pred = torch.zeros(n_items_batch, device=device)
        sum_tm = torch.zeros(n_items_batch, device=device)
        sum_pred.scatter_add_(0, inverse, weighted_pred)
        sum_tm.scatter_add_(0, inverse, b_tm)
        agg_pred = sum_pred / sum_tm.clamp(min=1e-8)

        agg_target = torch.zeros(n_items_batch, device=device)
        agg_target.scatter_(0, inverse, b_targets_flat)

        loss = F.mse_loss(agg_pred, agg_target)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-ranges", required=True)
    parser.add_argument("--benchmark", default="data/raw/benchmark_affinity.csv")
    parser.add_argument("--domain-metadata", required=True)
    parser.add_argument("--domain-dir", required=True, help="AlphaFold domain PDB dir")
    parser.add_argument("--target-db", default=None, help="Pre-built Foldseek DB")
    parser.add_argument("--idrome-index", default="data/processed/idrome_conformation_index.json")
    parser.add_argument("--output-dir", default="models/affinity_gat_alphafold")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--gat-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--anchor-tm-threshold", type=float, default=0.4)
    parser.add_argument("--foldseek-bin", default="foldseek")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--n-conformations", type=int, default=10)
    parser.add_argument("--max-drugs", type=int, default=100)
    parser.add_argument("--anchor-cache", default=None, help="Cache file for precomputed anchors")
    parser.add_argument("--proj-dim", type=int, default=128,
                        help="Projection head dim for InfoNCE (0=disable, MSE only)")
    parser.add_argument("--pos-threshold", type=float, default=7.0)
    parser.add_argument("--neg-threshold", type=float, default=5.0)
    parser.add_argument("--proteins-per-batch", type=int, default=4)
    parser.add_argument("--pos-per-protein", type=int, default=8)
    parser.add_argument("--hard-neg-per-protein", type=int, default=8)
    parser.add_argument("--mid-per-protein", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device

    logger.info("Loading graph from %s...", args.graph)
    graph = torch.load(args.graph, map_location=device, weights_only=False)
    node_ranges = torch.load(args.node_ranges, map_location="cpu", weights_only=False)
    logger.info("Graph: %d nodes, %d edges", graph.num_nodes, graph.edge_index.shape[1])
    # Keep graph on CPU — move to GPU only during forward pass to save VRAM
    # graph stays on CPU

    with open(args.domain_metadata) as f:
        domain_metadata = json.load(f)

    idrome_index = load_idrome_index(Path(args.idrome_index))
    benchmark_df = pd.read_csv(args.benchmark)
    holdout_proteins = set(benchmark_df["uniprot_id"].unique())
    logger.info("Holdout: %d benchmark proteins", len(holdout_proteins))

    # Load ALL BindingDB interactions for ordered proteins in graph
    train_interactions = pd.read_csv("data/processed/training_interactions_affinity.csv")
    train_interactions["uniprot_id"] = train_interactions["uniprot_id"].str.replace("af_", "", regex=False)
    # Keep all proteins except holdouts — they don't need to be in the graph
    train_interactions = train_interactions[
        ~train_interactions["uniprot_id"].isin(holdout_proteins)
    ]
    # Compute pKi if not present
    if "pki" not in train_interactions.columns:
        train_interactions["pki"] = 9.0 - np.log10(train_interactions["binding_affinity"].clip(lower=1e-3))
        train_interactions["pki"] = train_interactions["pki"].clip(3.0, 12.0)
    train_interactions["protein_type"] = "ordered"
    logger.info("Training data: %d pairs, %d proteins (ordered, non-holdout)",
                len(train_interactions), train_interactions.uniprot_id.nunique())

    # Pre-compute anchors for all training proteins via Foldseek
    # Every protein finds anchors in the graph — they don't need to be in the graph
    cache_path = Path(args.anchor_cache) if args.anchor_cache else output_dir / "anchor_cache.json"
    if cache_path.exists():
        logger.info("[CACHED] Loading anchors from %s", cache_path)
        with open(cache_path) as f:
            protein_anchors = json.load(f)
        logger.info("Loaded anchors for %d proteins", len(protein_anchors))
    else:
        logger.info("Pre-computing Foldseek anchors for %d training proteins...",
                    train_interactions.uniprot_id.nunique())
        protein_anchors = precompute_protein_anchors(
            train_interactions, node_ranges, domain_metadata, idrome_index,
            domain_pdb_dir=args.domain_dir,
            target_db=args.target_db,
            foldseek_bin=args.foldseek_bin,
            tm_threshold=args.anchor_tm_threshold,
            threads=args.threads,
            n_conformations=args.n_conformations,
        )
        with open(cache_path, "w") as f:
            json.dump(protein_anchors, f)
        logger.info("Saved anchor cache to %s (%d proteins)", cache_path, len(protein_anchors))

    train_data, val_data = prepare_training_data(
        train_interactions, protein_anchors, max_drugs=0,  # use ALL data for training
    )

    # Pre-encode all unique SMILES into a lookup + tensor
    all_smiles = set()
    for item in train_data + val_data:
        all_smiles.add(item["smiles"])
    smiles_list_unique = sorted(all_smiles)
    smiles_lookup = {s: i for i, s in enumerate(smiles_list_unique)}
    smiles_tensor = torch.tensor(
        [encode_smiles(s) for s in smiles_list_unique],
        dtype=torch.long, device=device,
    )
    logger.info("Pre-encoded %d unique SMILES", len(smiles_list_unique))

    esm2_dim = graph.x_esm2.shape[1] if hasattr(graph, "x_esm2") else 480

    model = AffinityGAT(
        esm2_input_dim=esm2_dim,
        hidden_dim=args.hidden_dim,
        gat_layers=args.gat_layers,
        dropout=args.dropout,
        proj_dim=args.proj_dim,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %d trainable parameters", n_params)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda") if "cuda" in device else None

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, graph, train_data, optimizer, device, args.batch_size, scaler,
            smiles_tensor=smiles_tensor, smiles_lookup=smiles_lookup,
            epoch=epoch, max_epochs=args.epochs,
            proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            hard_neg_per_protein=args.hard_neg_per_protein,
            mid_per_protein=args.mid_per_protein,
            pos_threshold=args.pos_threshold,
            neg_threshold=args.neg_threshold,
        )
        val_loss = validate(
            model, graph, val_data, device,
            smiles_tensor=smiles_tensor, smiles_lookup=smiles_lookup,
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args),
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        logger.info("Epoch %d/%d: train=%.4f val=%.4f lr=%.6f %s (patience %d/%d)",
                    epoch, args.epochs, train_loss, val_loss, lr,
                    "*" if improved else "", patience_counter, args.patience)

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info("DONE — Best val loss: %.4f, saved to %s", best_val_loss, output_dir)


if __name__ == "__main__":
    main()
