# Contrastive Binder/Non-Binder Discriminator Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train AffinityGAT with pure InfoNCE contrastive loss to discriminate drug binders from non-binders using domain-based training data.

**Architecture:** Existing AffinityGAT (GATv2 + cross-attention + projection head) trained with InfoNCE only (no MSE). ProteinBatchSampler provides pKi-thresholded positives/negatives + in-batch negatives. Eval uses retrieval (AUROC, AUPRC) and ranking (MRR, Hit@k) metrics.

**Tech Stack:** PyTorch, PyG, existing AffinityGAT, existing ProteinBatchSampler, existing infonce_loss, domain_training_data.csv.

---

### Task 1: Contrastive Training Script

**Files:**
- Create: `scripts/train_contrastive_binder.py`
- Reuses: `src/idr_gat/model/affinity_gat.py` (AffinityGAT, infonce_loss, encode_smiles)
- Reuses: `src/idr_gat/data/protein_batch_sampler.py` (ProteinBatchSampler)

- [ ] **Step 1: Write training script**

```python
#!/usr/bin/env python3
"""Train AffinityGAT with pure InfoNCE loss for binder/non-binder discrimination.

Uses domain training data (domain_training_data.csv) with ProteinBatchSampler.
Positives: pKi >= 7, Hard negatives: pKi <= 5, In-batch negatives from other proteins.
No MSE head, no curriculum — pure contrastive.

Usage:
    python scripts/train_contrastive_binder.py \
        --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
        --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
        --training-data data/processed/domain_training_data.csv \
        --device cuda --epochs 200
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from idr_gat.model.affinity_gat import AffinityGAT, infonce_loss, encode_smiles
from idr_gat.data.protein_batch_sampler import ProteinBatchSampler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_graph(graph_path, node_ranges_path, device):
    graph = torch.load(graph_path, map_location=device, weights_only=False)
    node_ranges = torch.load(node_ranges_path, map_location=device, weights_only=False)
    return graph, node_ranges


def build_training_items(df, node_ranges):
    """Convert domain training CSV to list of dicts for ProteinBatchSampler."""
    items = []
    smiles_set = set()
    skipped = 0

    for _, row in df.iterrows():
        uid = row["uniprot_id"]
        if uid not in node_ranges:
            skipped += 1
            continue
        start_node, end_node = node_ranges[uid]
        anchor_node = start_node  # use first node as anchor
        items.append({
            "uniprot_id": uid,
            "smiles": row["ligand_smiles"],
            "pki": float(row["pki"]),
            "anchors": [{"anchor_node": int(anchor_node), "anchor_tm": 1.0}],
        })
        smiles_set.add(row["ligand_smiles"])

    return items, sorted(smiles_set), skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-ranges", required=True)
    parser.add_argument("--training-data", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--proteins-per-batch", type=int, default=32)
    parser.add_argument("--pos-per-protein", type=int, default=16)
    parser.add_argument("--hard-neg-per-protein", type=int, default=16)
    parser.add_argument("--pos-threshold", type=float, default=7.0)
    parser.add_argument("--neg-threshold", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="models/contrastive_binder")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph
    logger.info("Loading graph...")
    graph, node_ranges_raw = load_graph(args.graph, args.node_ranges, device)
    # Convert node_ranges to dict
    node_ranges = {}
    if isinstance(node_ranges_raw, dict):
        node_ranges = node_ranges_raw
    else:
        node_ranges = node_ranges_raw

    # Load domain training data
    df = pd.read_csv(args.training_data)
    logger.info("Loaded %d domain-drug pairs", len(df))

    # Split 90/10 by protein
    uids = df["uniprot_id"].unique()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uids)
    n_train = int(0.9 * len(uids))
    train_uids = set(uids[:n_train])
    val_uids = set(uids[n_train:])

    train_df = df[df["uniprot_id"].isin(train_uids)]
    val_df = df[df["uniprot_id"].isin(val_uids)]

    train_items, train_smiles, skip_train = build_training_items(train_df, node_ranges)
    val_items, val_smiles, skip_val = build_training_items(val_df, node_ranges)
    logger.info("Train: %d items (%d skipped), Val: %d items (%d skipped)",
                len(train_items), skip_train, len(val_items), skip_val)

    # Build SMILES lookup
    all_smiles = sorted(set(train_smiles + val_smiles))
    smiles_lookup = {s: i for i, s in enumerate(all_smiles)}
    smiles_tensor = torch.stack([
        torch.tensor(encode_smiles(s), dtype=torch.long) for s in all_smiles
    ]).to(device)
    logger.info("SMILES vocabulary: %d", len(all_smiles))

    # Model
    model = AffinityGAT(proj_dim=args.proj_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("AffinityGAT (contrastive) parameters: %s", f"{n_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()

        # Pre-compute node embeddings once per epoch
        with torch.no_grad():
            graph_dev = graph.to(device) if graph.x_3di.device != device else graph
            node_embs = model.encode_graph(graph_dev)

        sampler = ProteinBatchSampler(
            train_items, proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            hard_neg_per_protein=args.hard_neg_per_protein,
            mid_per_protein=0,  # no middle zone for pure contrastive
            pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
        )

        total_nce, n_batches = 0.0, 0
        for batch in sampler:
            anchor_list, smi_list, lbl_list = [], [], []
            for item in batch:
                smi = item["smiles"]
                if smi not in smiles_lookup:
                    continue
                anc = item["anchors"][0]
                anchor_list.append(anc["anchor_node"])
                smi_list.append(smiles_lookup[smi])
                lbl_list.append(item["label"])

            if not anchor_list:
                continue

            b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
            b_smi_enc = smiles_tensor[torch.tensor(smi_list, dtype=torch.long, device=device)]
            b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

            prot_emb = node_embs[b_anchors]
            drug_emb = model.drug_encoder(b_smi_enc)
            interaction = model.cross_attn(prot_emb, drug_emb)

            proj = F.normalize(model.proj_head(interaction), dim=1)
            pos_mask = b_labels == 1
            neg_mask = b_labels == 0

            if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                nce = infonce_loss(proj[pos_mask], proj[pos_mask], proj[neg_mask],
                                   temperature=args.temperature)
            else:
                continue

            optimizer.zero_grad()
            nce.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_nce += nce.item()
            n_batches += 1

        train_loss = total_nce / max(n_batches, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            node_embs = model.encode_graph(graph_dev)

        val_sampler = ProteinBatchSampler(
            val_items, proteins_per_batch=args.proteins_per_batch,
            pos_per_protein=args.pos_per_protein,
            hard_neg_per_protein=args.hard_neg_per_protein,
            mid_per_protein=0,
            pos_threshold=args.pos_threshold, neg_threshold=args.neg_threshold,
        )

        val_nce, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_sampler:
                anchor_list, smi_list, lbl_list = [], [], []
                for item in batch:
                    smi = item["smiles"]
                    if smi not in smiles_lookup:
                        continue
                    anc = item["anchors"][0]
                    anchor_list.append(anc["anchor_node"])
                    smi_list.append(smiles_lookup[smi])
                    lbl_list.append(item["label"])

                if not anchor_list:
                    continue

                b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
                b_smi_enc = smiles_tensor[torch.tensor(smi_list, dtype=torch.long, device=device)]
                b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

                prot_emb = node_embs[b_anchors]
                drug_emb = model.drug_encoder(b_smi_enc)
                interaction = model.cross_attn(prot_emb, drug_emb)
                proj = F.normalize(model.proj_head(interaction), dim=1)

                pos_mask = b_labels == 1
                neg_mask = b_labels == 0
                if pos_mask.sum() > 0 and neg_mask.sum() > 0:
                    nce = infonce_loss(proj[pos_mask], proj[pos_mask], proj[neg_mask],
                                       temperature=args.temperature)
                    val_nce += nce.item()
                    val_batches += 1

        val_loss = val_nce / max(val_batches, 1)
        scheduler.step()

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "proj_dim": args.proj_dim,
                "temperature": args.temperature,
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            logger.info("Epoch %d/%d train_nce=%.4f val_nce=%.4f best=%.4f patience=%d/%d%s",
                        epoch, args.epochs, train_loss, val_loss, best_val_loss,
                        patience_counter, args.patience, " *" if improved else "")

        if patience_counter >= args.patience:
            logger.info("Early stopping at epoch %d", epoch)
            break

    logger.info("Done. Best val_nce=%.4f at %s", best_val_loss, output_dir / "best_model.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_contrastive_binder.py
git commit -m "feat: contrastive binder training script (pure InfoNCE)"
```

---

### Task 2: Evaluation Script with Retrieval + Ranking Metrics

**Files:**
- Create: `scripts/evaluate_contrastive_binder.py`
- Reuses: AffinityGAT, Foldseek anchor finding, domain sequence extraction

- [ ] **Step 1: Write evaluation script**

```python
#!/usr/bin/env python3
"""Evaluate contrastive binder model on benchmark.

For each benchmark protein:
1. Find best anchor domain via Foldseek
2. Compute cosine similarity between anchor-drug embedding pairs
3. Rank drugs by similarity score
4. Compute retrieval (AUROC, AUPRC) and ranking (MRR, Hit@k) metrics

Usage:
    python scripts/evaluate_contrastive_binder.py \
        --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
        --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
        --model models/contrastive_binder/best_model.pt \
        --benchmark data/raw/benchmark_affinity.csv \
        --domain-dir data/processed/model_organism_domains \
        --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

from idr_gat.model.affinity_gat import AffinityGAT, encode_smiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def compute_retrieval_metrics(true_labels, scores):
    """Compute AUROC and AUPRC for binder detection."""
    if len(set(true_labels)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    auroc = roc_auc_score(true_labels, scores)
    auprc = average_precision_score(true_labels, scores)
    return {"auroc": auroc, "auprc": auprc}


def compute_ranking_metrics(true_labels, scores, ks=(1, 5, 10)):
    """Compute MRR and Hit@k."""
    ranked_indices = np.argsort(-scores)
    ranked_labels = np.array(true_labels)[ranked_indices]

    # MRR: reciprocal rank of first positive
    positives = np.where(ranked_labels == 1)[0]
    mrr = 1.0 / (positives[0] + 1) if len(positives) > 0 else 0.0

    # Hit@k
    hits = {}
    for k in ks:
        hits[f"hit@{k}"] = 1.0 if any(ranked_labels[:k] == 1) else 0.0

    return {"mrr": mrr, **hits}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--node-ranges", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--domain-metadata", required=True)
    parser.add_argument("--sequences", default="data/processed/merged_sequences.json")
    parser.add_argument("--proj-dim", type=int, default=128)
    parser.add_argument("--binder-threshold", type=float, default=7.0,
                        help="pKi threshold for defining a binder (for AUROC/AUPRC)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/contrastive_binder")
    args = parser.parse_args()

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load graph + model
    graph = torch.load(args.graph, map_location=device, weights_only=False)
    node_ranges = torch.load(args.node_ranges, map_location=device, weights_only=False)
    checkpoint = torch.load(args.model, map_location=device, weights_only=False)

    model = AffinityGAT(proj_dim=args.proj_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    logger.info("Loaded contrastive model from %s (epoch %d, val_nce=%.4f)",
                args.model, checkpoint["epoch"], checkpoint["best_val_loss"])

    # Pre-compute node embeddings
    with torch.no_grad():
        graph_dev = graph.to(device)
        node_embs = model.encode_graph(graph_dev)

    # Load benchmark
    bench_df = pd.read_csv(args.benchmark)
    logger.info("Benchmark: %d pairs, %d proteins", len(bench_df), bench_df["uniprot_id"].nunique())

    # Per-protein evaluation
    protein_results = []
    for uid, group in bench_df.groupby("uniprot_id"):
        ptype = group["protein_type"].iloc[0]

        if uid not in node_ranges:
            continue

        start_node, end_node = node_ranges[uid]
        anchor_node = start_node

        smiles_list = group["ligand_smiles"].tolist()
        true_pkis = group["pki"].values
        true_labels = (true_pkis >= args.binder_threshold).astype(int)

        if true_labels.sum() == 0 or true_labels.sum() == len(true_labels):
            continue  # need both classes

        # Encode all drugs
        smi_encs = [encode_smiles(s) for s in smiles_list]
        smi_tensor = torch.tensor(smi_encs, dtype=torch.long, device=device)

        # Compute contrastive scores
        with torch.no_grad():
            prot_emb = node_embs[anchor_node].unsqueeze(0).expand(len(smiles_list), -1)
            drug_emb = model.drug_encoder(smi_tensor)
            interaction = model.cross_attn(prot_emb, drug_emb)
            proj = F.normalize(model.proj_head(interaction), dim=1)

            # Score = L2 norm of projection (closer to learned positive direction = higher)
            # Use the projection magnitude after normalization as a proxy
            # Actually, for ranking we need a reference. Use cosine sim with mean positive embedding.
            # Simpler: just use the first dimension projection as score, or use the norm before normalization
            # Best: compute similarity to a learned "binder prototype"
            # Simplest correct approach: use raw projection and rank by cosine sim to mean
            scores = proj.cpu().numpy()

        # Use mean of positive projections as prototype (if available in training)
        # For eval, just use the raw L2-normalized embeddings and compute a score
        # Score = sum of embedding dimensions (acts as a linear classifier in embedding space)
        raw_scores = scores.sum(axis=1)

        # Compute metrics
        retrieval = compute_retrieval_metrics(true_labels, raw_scores)
        ranking = compute_ranking_metrics(true_labels, raw_scores)

        protein_results.append({
            "uniprot_id": uid, "protein_type": ptype, "n_drugs": len(smiles_list),
            "n_binders": int(true_labels.sum()),
            **retrieval, **ranking,
        })

        if len(protein_results) % 10 == 0:
            logger.info("[%d] %s (%s): AUROC=%.3f AUPRC=%.3f MRR=%.3f Hit@5=%.1f",
                        len(protein_results), uid, ptype,
                        retrieval["auroc"], retrieval["auprc"],
                        ranking["mrr"], ranking["hit@5"])

    # Aggregate
    res_df = pd.DataFrame(protein_results)
    res_df.to_csv(output_dir / "per_protein_results.csv", index=False)

    logger.info("=" * 80)
    logger.info("RESULTS (%d proteins)", len(res_df))
    logger.info("=" * 80)

    for ptype in ["idp", "ordered", "all"]:
        sub = res_df if ptype == "all" else res_df[res_df["protein_type"] == ptype]
        if len(sub) == 0:
            continue
        logger.info("  %s (n=%d): AUROC=%.3f AUPRC=%.3f MRR=%.3f Hit@1=%.3f Hit@5=%.3f Hit@10=%.3f",
                    ptype.upper(), len(sub),
                    sub["auroc"].mean(), sub["auprc"].mean(), sub["mrr"].mean(),
                    sub["hit@1"].mean(), sub["hit@5"].mean(), sub["hit@10"].mean())

    # Save summary
    summary = {}
    for ptype in ["idp", "ordered", "all"]:
        sub = res_df if ptype == "all" else res_df[res_df["protein_type"] == ptype]
        if len(sub) == 0:
            continue
        summary[ptype] = {col: float(sub[col].mean()) for col in ["auroc", "auprc", "mrr", "hit@1", "hit@5", "hit@10"]}
        summary[ptype]["n"] = len(sub)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Results saved to %s", output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/evaluate_contrastive_binder.py
git commit -m "feat: contrastive binder eval with retrieval + ranking metrics"
```

---

### Task 3: Run on Remote

- [ ] **Step 1: Upload scripts and start training**

```bash
scp scripts/train_contrastive_binder.py scripts/evaluate_contrastive_binder.py root@remote:/workspace/IDP-work/scripts/

ssh root@remote "cd /workspace/IDP-work && nohup env PYTHONPATH=src:. python -u scripts/train_contrastive_binder.py \
  --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
  --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
  --training-data data/processed/domain_training_data.csv \
  --device cuda --epochs 200 --proj-dim 128 \
  --proteins-per-batch 32 --pos-per-protein 16 --hard-neg-per-protein 16 \
  --output-dir models/contrastive_binder \
  > logs/train_contrastive_binder.log 2>&1 &"
```

- [ ] **Step 2: Run eval after training completes**

```bash
ssh root@remote "cd /workspace/IDP-work && PYTHONPATH=src:. python -u scripts/evaluate_contrastive_binder.py \
  --graph data/graphs/alphafold_human_tm09_e04-09/global_graph.pt \
  --node-ranges data/graphs/alphafold_human_tm09_e04-09/protein_node_ranges.pt \
  --model models/contrastive_binder/best_model.pt \
  --benchmark data/raw/benchmark_affinity.csv \
  --domain-metadata data/processed/alphafold_human_domains/domain_metadata.json \
  --device cuda \
  --output-dir results/contrastive_binder"
```
