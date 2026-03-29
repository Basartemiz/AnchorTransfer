# Multi-Task AffinityGAT (InfoNCE + MSE) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a contrastive InfoNCE head with hard negatives alongside the existing MSE regression head in AffinityGAT, with curriculum scheduling from ranking-focused to regression-focused.

**Architecture:** Same GATv2 graph encoder. Add a projection head (512→128, L2-norm) branching from the shared cross-attention output. Protein-centric batches with pKi-filtered positives (≥7), hard negatives (≤5), and middle zone (5-7, MSE only). Linear curriculum interpolates loss weights over training.

**Tech Stack:** PyTorch, PyG (torch_geometric), existing AffinityGAT, existing train_affinity_gat.py training loop.

---

## File Structure

```
src/idr_gat/model/affinity_gat.py    — Add projection head, InfoNCE loss, curriculum compute_loss
scripts/train_affinity_gat.py         — Add protein-centric batch sampler, curriculum schedule, dual-loss training loop
tests/test_affinity_gat_multitask.py  — New test file for all multi-task components
```

---

### Task 1: Projection Head + InfoNCE Loss in AffinityGAT

**Files:**
- Modify: `src/idr_gat/model/affinity_gat.py`
- Create: `tests/test_affinity_gat_multitask.py`

- [ ] **Step 1: Write failing tests for projection head and InfoNCE**

Create `tests/test_affinity_gat_multitask.py`:

```python
"""Tests for multi-task AffinityGAT: projection head + InfoNCE + curriculum."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest
from torch_geometric.data import Data


def _make_graph(n_nodes=20):
    """Minimal graph with 3Di + ESM-2 features."""
    edge_index = torch.stack([
        torch.arange(n_nodes - 1),
        torch.arange(1, n_nodes),
    ])
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return Data(
        x_3di=torch.randint(0, 20, (n_nodes, 50)),
        x_seq_lens=torch.full((n_nodes,), 50, dtype=torch.long),
        x_esm2=torch.randn(n_nodes, 480),
        edge_index=edge_index,
        edge_attr=torch.rand(edge_index.size(1), 1),
    )


class TestProjectionHead:
    def test_projection_output_shape(self):
        from idr_gat.model.affinity_gat import AffinityGAT
        model = AffinityGAT(proj_dim=128)
        graph = _make_graph()
        anchor_idx = torch.tensor([0, 5])
        smiles_idx = torch.randint(0, 50, (2, 100))
        proj = model.project(graph, anchor_idx, smiles_idx)
        assert proj.shape == (2, 128)

    def test_projection_is_unit_norm(self):
        from idr_gat.model.affinity_gat import AffinityGAT
        model = AffinityGAT(proj_dim=128)
        model.eval()
        graph = _make_graph()
        anchor_idx = torch.tensor([0, 5, 10])
        smiles_idx = torch.randint(0, 50, (3, 100))
        proj = model.project(graph, anchor_idx, smiles_idx)
        norms = proj.norm(dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_no_proj_head_when_dim_zero(self):
        from idr_gat.model.affinity_gat import AffinityGAT
        model = AffinityGAT(proj_dim=0)
        assert not hasattr(model, "proj_head") or model.proj_head is None


class TestInfoNCELoss:
    def test_infonce_perfect_separation(self):
        from idr_gat.model.affinity_gat import infonce_loss
        # Positive pair has high similarity, negatives have low
        anchor = F.normalize(torch.randn(1, 128), dim=1)
        positive = anchor + torch.randn(1, 128) * 0.01  # very close
        negatives = torch.randn(5, 128)  # random = far
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        assert loss.item() < 0.5  # should be low for perfect separation

    def test_infonce_no_negatives_raises(self):
        from idr_gat.model.affinity_gat import infonce_loss
        anchor = torch.randn(1, 128)
        positive = torch.randn(1, 128)
        negatives = torch.zeros(0, 128)
        # Should handle gracefully (return 0 or small value)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        assert loss.item() >= 0

    def test_infonce_gradient_flows(self):
        from idr_gat.model.affinity_gat import infonce_loss
        anchor = F.normalize(torch.randn(2, 128, requires_grad=True), dim=1)
        positive = F.normalize(torch.randn(2, 128, requires_grad=True), dim=1)
        negatives = F.normalize(torch.randn(8, 128, requires_grad=True), dim=1)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        loss.backward()
        assert anchor.grad is not None
        assert positive.grad is not None


class TestCurriculumLoss:
    def test_early_epoch_infonce_dominant(self):
        from idr_gat.model.affinity_gat import curriculum_weights
        alpha, beta = curriculum_weights(epoch=0, max_epochs=200)
        assert beta > alpha  # InfoNCE weight > MSE weight
        assert abs(alpha - 0.2) < 1e-5
        assert abs(beta - 0.8) < 1e-5

    def test_late_epoch_mse_dominant(self):
        from idr_gat.model.affinity_gat import curriculum_weights
        alpha, beta = curriculum_weights(epoch=200, max_epochs=200)
        assert alpha > beta
        assert abs(alpha - 0.8) < 1e-5
        assert abs(beta - 0.2) < 1e-5

    def test_midpoint(self):
        from idr_gat.model.affinity_gat import curriculum_weights
        alpha, beta = curriculum_weights(epoch=100, max_epochs=200)
        assert abs(alpha - 0.5) < 1e-5
        assert abs(beta - 0.5) < 1e-5

    def test_weights_sum_to_one(self):
        from idr_gat.model.affinity_gat import curriculum_weights
        for epoch in [0, 50, 100, 150, 200]:
            alpha, beta = curriculum_weights(epoch, 200)
            assert abs(alpha + beta - 1.0) < 1e-5


class TestMultiTaskComputeLoss:
    def test_compute_multitask_loss(self):
        from idr_gat.model.affinity_gat import AffinityGAT
        model = AffinityGAT(proj_dim=128)
        graph = _make_graph()
        # 4 samples: 2 positive (pKi>=7), 2 hard negative (pKi<=5)
        anchor_idx = torch.tensor([0, 3, 7, 12])
        smiles_idx = torch.randint(0, 50, (4, 100))
        targets = torch.tensor([8.0, 7.5, 4.0, 3.5])
        labels = torch.tensor([1, 1, 0, 0])  # 1=positive, 0=hard_neg

        losses = model.compute_multitask_loss(
            graph, anchor_idx, smiles_idx, targets, labels,
            epoch=0, max_epochs=200,
        )
        assert "total" in losses
        assert "mse" in losses
        assert "infonce" in losses
        assert losses["total"].dim() == 0
        assert losses["mse"].item() >= 0
        assert losses["infonce"].item() >= 0

    def test_gradient_flows_both_heads(self):
        from idr_gat.model.affinity_gat import AffinityGAT
        model = AffinityGAT(proj_dim=128)
        graph = _make_graph()
        anchor_idx = torch.tensor([0, 3, 7, 12])
        smiles_idx = torch.randint(0, 50, (4, 100))
        targets = torch.tensor([8.0, 7.5, 4.0, 3.5])
        labels = torch.tensor([1, 1, 0, 0])

        losses = model.compute_multitask_loss(
            graph, anchor_idx, smiles_idx, targets, labels,
            epoch=50, max_epochs=200,
        )
        losses["total"].backward()
        # Regression head gets gradient
        assert model.head[0].weight.grad is not None
        # Projection head gets gradient
        assert model.proj_head.weight.grad is not None
        # Shared encoder gets gradient
        assert model.drug_encoder.proj.weight.grad is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py -v`

Expected: FAIL — `AffinityGAT` has no `proj_dim`, `project`, `proj_head`, `compute_multitask_loss`; `infonce_loss` and `curriculum_weights` don't exist.

- [ ] **Step 3: Implement projection head, InfoNCE, curriculum in affinity_gat.py**

Add these to the end of `src/idr_gat/model/affinity_gat.py` (before the `AffinityGAT` class closing), and modify `__init__` and add methods:

Add the standalone functions before the class:

```python
def infonce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Compute InfoNCE loss.

    Args:
        anchor: (B, D) L2-normalized embeddings
        positive: (B, D) L2-normalized embeddings (one positive per anchor)
        negatives: (N, D) L2-normalized embeddings (shared negatives)
        temperature: softmax temperature
    Returns:
        scalar loss
    """
    if negatives.size(0) == 0:
        return torch.tensor(0.0, device=anchor.device, requires_grad=True)

    # (B,) positive similarity
    pos_sim = (anchor * positive).sum(dim=1) / temperature
    # (B, N) negative similarities
    neg_sim = torch.mm(anchor, negatives.t()) / temperature
    # (B, 1+N) logits: positive first, then negatives
    logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
    # Target: index 0 is the positive
    labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


def curriculum_weights(epoch: int, max_epochs: int) -> tuple[float, float]:
    """Compute curriculum MSE (alpha) and InfoNCE (beta) weights.

    Linear interpolation: alpha 0.2→0.8, beta 0.8→0.2.
    """
    progress = min(epoch / max(max_epochs, 1), 1.0)
    alpha = 0.2 + 0.6 * progress
    beta = 0.8 - 0.6 * progress
    return alpha, beta
```

Modify `AffinityGAT.__init__` to accept `proj_dim`:

```python
def __init__(
    self,
    threedi_vocab_size: int = 21,
    threedi_embed_dim: int = 64,
    esm2_input_dim: int = 480,
    esm2_proj_dim: int = 256,
    hidden_dim: int = 512,
    gat_layers: int = 4,
    gat_heads: int = 8,
    dropout: float = 0.3,
    smiles_vocab: int = 52,
    smiles_embed_dim: int = 128,
    smiles_filters: int = 32,
    cross_attn_heads: int = 4,
    proj_dim: int = 0,
):
    super().__init__()
    # ... existing code unchanged ...

    # Projection head for contrastive learning (optional)
    self.proj_head = nn.Linear(hidden_dim, proj_dim) if proj_dim > 0 else None
```

Add `project` and `compute_multitask_loss` methods to AffinityGAT:

```python
def project(
    self,
    graph_data,
    anchor_indices: torch.Tensor,
    smiles_indices: torch.Tensor,
) -> torch.Tensor:
    """Project protein-drug pairs to L2-normalized contrastive space.

    Returns: (B, proj_dim) unit-norm embeddings.
    """
    device = smiles_indices.device
    graph_on_device = graph_data.to(device) if graph_data.x_3di.device != device else graph_data
    node_embs = self.encode_graph(graph_on_device)
    prot_emb = node_embs[anchor_indices]
    drug_emb = self.drug_encoder(smiles_indices)
    interaction = self.cross_attn(prot_emb, drug_emb)
    proj = self.proj_head(interaction)
    return F.normalize(proj, dim=1)

def compute_multitask_loss(
    self,
    graph_data,
    anchor_indices: torch.Tensor,
    smiles_indices: torch.Tensor,
    targets: torch.Tensor,
    labels: torch.Tensor,
    epoch: int = 0,
    max_epochs: int = 200,
) -> dict[str, torch.Tensor]:
    """Compute combined InfoNCE + MSE loss with curriculum weighting.

    Args:
        graph_data: PyG Data with full graph
        anchor_indices: (B,) graph node indices
        smiles_indices: (B, max_len) encoded SMILES
        targets: (B,) pKi values for all samples
        labels: (B,) int — 1=positive (pKi>=7), 0=hard_negative (pKi<=5), -1=middle
        epoch: current epoch for curriculum scheduling
        max_epochs: total epochs for curriculum scheduling
    Returns:
        dict with keys: total, mse, infonce, preds
    """
    device = smiles_indices.device
    graph_on_device = graph_data.to(device) if graph_data.x_3di.device != device else graph_data
    node_embs = self.encode_graph(graph_on_device)
    prot_emb = node_embs[anchor_indices]
    drug_emb = self.drug_encoder(smiles_indices)
    interaction = self.cross_attn(prot_emb, drug_emb)

    # Regression head — all samples
    preds = self.head(interaction).squeeze(-1)
    mse = F.mse_loss(preds, targets)

    # Contrastive head — positives vs (hard negatives + easy negatives)
    if self.proj_head is not None:
        proj = F.normalize(self.proj_head(interaction), dim=1)
        pos_mask = labels == 1
        neg_mask = labels == 0

        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_proj = proj[pos_mask]
            neg_proj = proj[neg_mask]
            # Each positive is its own anchor, other positives are not used as positives
            # (they're for the same protein, so they're all valid)
            # Use first positive as anchor, rest as additional negatives for diversity
            # Simple: for each positive, the "positive target" is itself (self-supervised)
            # and negatives are all hard negatives + other positives from other proteins
            # For single-protein batch: each positive vs all hard negatives
            anchor_emb = pos_proj
            # Positive = itself (each row), negatives = all hard negatives
            nce = infonce_loss(anchor_emb, anchor_emb, neg_proj, temperature=0.07)
        else:
            nce = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        nce = torch.tensor(0.0, device=device, requires_grad=True)

    alpha, beta = curriculum_weights(epoch, max_epochs)
    total = alpha * mse + beta * nce

    return {"total": total, "mse": mse, "infonce": nce, "preds": preds}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py -v`

Expected: All 11 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add src/idr_gat/model/affinity_gat.py tests/test_affinity_gat_multitask.py
git commit -m "feat: add projection head, InfoNCE loss, and curriculum weights to AffinityGAT"
```

---

### Task 2: Protein-Centric Batch Sampler

**Files:**
- Create: `src/idr_gat/data/protein_batch_sampler.py`
- Modify: `tests/test_affinity_gat_multitask.py`

- [ ] **Step 1: Write failing tests for the batch sampler**

Append to `tests/test_affinity_gat_multitask.py`:

```python
class TestProteinBatchSampler:
    def _make_items(self):
        """Fake training data: 3 proteins, each with pos/mid/neg drugs."""
        items = []
        for uid in ["P001", "P002", "P003"]:
            # 10 positives (pKi 7-10)
            for i in range(10):
                items.append({"uniprot_id": uid, "smiles": f"{uid}_pos_{i}", "pki": 7.0 + i * 0.3,
                              "anchors": [{"anchor_node": 0, "anchor_tm": 0.9}]})
            # 5 middle (pKi 5-7)
            for i in range(5):
                items.append({"uniprot_id": uid, "smiles": f"{uid}_mid_{i}", "pki": 5.5 + i * 0.3,
                              "anchors": [{"anchor_node": 0, "anchor_tm": 0.9}]})
            # 10 hard negatives (pKi 3-5)
            for i in range(10):
                items.append({"uniprot_id": uid, "smiles": f"{uid}_neg_{i}", "pki": 3.0 + i * 0.2,
                              "anchors": [{"anchor_node": 0, "anchor_tm": 0.9}]})
        return items

    def test_batch_has_correct_counts(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        batch = next(iter(sampler))
        # 2 proteins × (4 pos + 4 neg + 2 mid) = 20 items
        assert len(batch) == 20

    def test_batch_labels_correct(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        batch = next(iter(sampler))
        labels = [item["label"] for item in batch]
        assert labels.count(1) == 8   # 2 proteins × 4 positives
        assert labels.count(0) == 8   # 2 proteins × 4 hard negatives
        assert labels.count(-1) == 4  # 2 proteins × 2 middle

    def test_iterates_all_proteins(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        seen_proteins = set()
        for batch in sampler:
            for item in batch:
                seen_proteins.add(item["uniprot_id"])
        assert seen_proteins == {"P001", "P002", "P003"}

    def test_len(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        # 3 proteins, 2 per batch = ceil(3/2) = 2 batches
        assert len(sampler) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py::TestProteinBatchSampler -v`

Expected: FAIL — `idr_gat.data.protein_batch_sampler` doesn't exist.

- [ ] **Step 3: Implement ProteinBatchSampler**

Create `src/idr_gat/data/protein_batch_sampler.py`:

```python
"""Protein-centric batch sampler for multi-task affinity training."""
from __future__ import annotations

import math
import random
from collections import defaultdict


class ProteinBatchSampler:
    """Yields batches of protein-drug items grouped by protein.

    Each batch contains `proteins_per_batch` proteins. For each protein,
    samples pos/hard_neg/mid drugs based on pKi thresholds.

    Items in each batch get a 'label' field added:
      1 = positive (pKi >= pos_threshold)
      0 = hard negative (pKi <= neg_threshold)
     -1 = middle zone (excluded from InfoNCE, included in MSE)
    """

    def __init__(
        self,
        items: list[dict],
        proteins_per_batch: int = 4,
        pos_per_protein: int = 8,
        hard_neg_per_protein: int = 8,
        mid_per_protein: int = 4,
        pos_threshold: float = 7.0,
        neg_threshold: float = 5.0,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.proteins_per_batch = proteins_per_batch
        self.pos_per_protein = pos_per_protein
        self.hard_neg_per_protein = hard_neg_per_protein
        self.mid_per_protein = mid_per_protein
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.shuffle = shuffle
        self.rng = random.Random(seed)

        # Group items by protein and category
        self.protein_pools: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: {"pos": [], "neg": [], "mid": []}
        )
        for item in items:
            uid = item["uniprot_id"]
            pki = item["pki"]
            if pki >= pos_threshold:
                self.protein_pools[uid]["pos"].append(item)
            elif pki <= neg_threshold:
                self.protein_pools[uid]["neg"].append(item)
            else:
                self.protein_pools[uid]["mid"].append(item)

        # Only keep proteins that have at least 1 positive AND 1 negative
        self.valid_proteins = [
            uid for uid, pools in self.protein_pools.items()
            if len(pools["pos"]) >= 1 and len(pools["neg"]) >= 1
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.valid_proteins) / self.proteins_per_batch)

    def _sample_k(self, pool: list[dict], k: int) -> list[dict]:
        if len(pool) <= k:
            return list(pool)
        return self.rng.sample(pool, k)

    def __iter__(self):
        proteins = list(self.valid_proteins)
        if self.shuffle:
            self.rng.shuffle(proteins)

        for start in range(0, len(proteins), self.proteins_per_batch):
            batch_proteins = proteins[start:start + self.proteins_per_batch]
            batch = []
            for uid in batch_proteins:
                pools = self.protein_pools[uid]
                for item in self._sample_k(pools["pos"], self.pos_per_protein):
                    batch.append({**item, "label": 1})
                for item in self._sample_k(pools["neg"], self.hard_neg_per_protein):
                    batch.append({**item, "label": 0})
                for item in self._sample_k(pools["mid"], self.mid_per_protein):
                    batch.append({**item, "label": -1})
            yield batch
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py::TestProteinBatchSampler -v`

Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add src/idr_gat/data/protein_batch_sampler.py tests/test_affinity_gat_multitask.py
git commit -m "feat: add ProteinBatchSampler for protein-centric multi-task batching"
```

---

### Task 3: Update Training Loop

**Files:**
- Modify: `scripts/train_affinity_gat.py`
- Modify: `tests/test_affinity_gat_multitask.py`

- [ ] **Step 1: Write failing test for multi-task training step**

Append to `tests/test_affinity_gat_multitask.py`:

```python
class TestMultiTaskTrainStep:
    def test_train_step_runs(self):
        """Smoke test: one multi-task training step completes without error."""
        from idr_gat.model.affinity_gat import AffinityGAT, encode_smiles, curriculum_weights
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        import torch

        graph = _make_graph(n_nodes=30)
        model = AffinityGAT(proj_dim=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Fake items: 2 proteins with pos/neg drugs
        items = []
        smiles_set = set()
        for uid_idx, uid in enumerate(["P1", "P2"]):
            for i in range(5):
                smi = f"CCO{uid_idx}{i}"
                items.append({"uniprot_id": uid, "smiles": smi, "pki": 8.0,
                              "anchors": [{"anchor_node": uid_idx * 10, "anchor_tm": 0.9}]})
                smiles_set.add(smi)
            for i in range(5):
                smi = f"CCN{uid_idx}{i}"
                items.append({"uniprot_id": uid, "smiles": smi, "pki": 4.0,
                              "anchors": [{"anchor_node": uid_idx * 10, "anchor_tm": 0.9}]})
                smiles_set.add(smi)

        # Pre-encode SMILES
        smiles_list = sorted(smiles_set)
        smiles_lookup = {s: i for i, s in enumerate(smiles_list)}
        smiles_tensor = torch.stack([torch.tensor(encode_smiles(s), dtype=torch.long) for s in smiles_list])

        sampler = ProteinBatchSampler(items, proteins_per_batch=2,
                                       pos_per_protein=3, hard_neg_per_protein=3, mid_per_protein=0)

        # One batch
        batch = next(iter(sampler))
        b_anchors = torch.tensor([item["anchors"][0]["anchor_node"] for item in batch])
        b_smiles = smiles_tensor[torch.tensor([smiles_lookup[item["smiles"]] for item in batch])]
        b_targets = torch.tensor([item["pki"] for item in batch])
        b_labels = torch.tensor([item["label"] for item in batch])

        optimizer.zero_grad()
        losses = model.compute_multitask_loss(graph, b_anchors, b_smiles, b_targets, b_labels,
                                               epoch=10, max_epochs=200)
        losses["total"].backward()
        optimizer.step()

        assert losses["mse"].item() > 0
        assert losses["infonce"].item() >= 0
        assert losses["total"].item() > 0
```

- [ ] **Step 2: Run test to verify it passes (uses already-implemented components)**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py::TestMultiTaskTrainStep -v`

Expected: PASS (all components from Tasks 1-2 are implemented).

- [ ] **Step 3: Update train_affinity_gat.py training loop**

Modify `scripts/train_affinity_gat.py`. Replace the `train_epoch` function. The key changes:

1. Add `--proj-dim` and `--curriculum` args to argparse
2. Use `ProteinBatchSampler` instead of flat shuffled batches
3. Call `model.compute_multitask_loss` instead of `F.mse_loss`
4. Log both loss components

Add to argparse section (find `parser.add_argument` block):

```python
parser.add_argument("--proj-dim", type=int, default=128,
                    help="Projection head dimension for InfoNCE (0=disable)")
parser.add_argument("--pos-threshold", type=float, default=7.0)
parser.add_argument("--neg-threshold", type=float, default=5.0)
parser.add_argument("--proteins-per-batch", type=int, default=4)
parser.add_argument("--pos-per-protein", type=int, default=8)
parser.add_argument("--hard-neg-per-protein", type=int, default=8)
parser.add_argument("--mid-per-protein", type=int, default=4)
```

Replace `train_epoch` function:

```python
def train_epoch(model, graph, train_data, optimizer, device, batch_size=64,
                scaler=None, smiles_tensor=None, smiles_lookup=None,
                epoch=0, max_epochs=200, proteins_per_batch=4,
                pos_per_protein=8, hard_neg_per_protein=8, mid_per_protein=4,
                pos_threshold=7.0, neg_threshold=5.0):
    model.train()

    # Pre-compute graph node embeddings ONCE per epoch
    with torch.no_grad():
        graph_dev = graph.to(device) if graph.x_3di.device != device else graph
        node_embs = model.encode_graph(graph_dev)

    # Use protein-centric batching if model has projection head
    use_multitask = hasattr(model, 'proj_head') and model.proj_head is not None

    if use_multitask:
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        sampler = ProteinBatchSampler(
            train_data, proteins_per_batch=proteins_per_batch,
            pos_per_protein=pos_per_protein, hard_neg_per_protein=hard_neg_per_protein,
            mid_per_protein=mid_per_protein, pos_threshold=pos_threshold,
            neg_threshold=neg_threshold,
        )

        total_loss = 0.0
        total_mse = 0.0
        total_nce = 0.0
        n_batches = 0

        for batch in sampler:
            # Build tensors from batch items
            anchor_list, smi_list, tgt_list, lbl_list = [], [], [], []
            for item in batch:
                smi = item["smiles"]
                if smi not in smiles_lookup:
                    continue
                # Use first anchor for simplicity (TM-weighting handled inside)
                anc = item["anchors"][0]
                anchor_list.append(anc["anchor_node"])
                smi_list.append(smiles_lookup[smi])
                tgt_list.append(item["pki"])
                lbl_list.append(item["label"])

            if not anchor_list:
                continue

            b_anchors = torch.tensor(anchor_list, dtype=torch.long, device=device)
            b_smi_idx = torch.tensor(smi_list, dtype=torch.long, device=device)
            b_smi_enc = smiles_tensor[b_smi_idx]
            b_targets = torch.tensor(tgt_list, dtype=torch.float32, device=device)
            b_labels = torch.tensor(lbl_list, dtype=torch.long, device=device)

            # Forward through shared encoder (use pre-computed node embeddings)
            prot_emb = node_embs[b_anchors]
            drug_emb = model.drug_encoder(b_smi_enc)
            interaction = model.cross_attn(prot_emb, drug_emb)

            # Regression
            preds = model.head(interaction).squeeze(-1)
            mse = F.mse_loss(preds, b_targets)

            # Contrastive
            from idr_gat.model.affinity_gat import infonce_loss, curriculum_weights
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

        avg_loss = total_loss / max(n_batches, 1)
        avg_mse = total_mse / max(n_batches, 1)
        avg_nce = total_nce / max(n_batches, 1)
        alpha, beta = curriculum_weights(epoch, max_epochs)
        logger.info("  [curriculum] alpha=%.2f beta=%.2f | mse=%.4f nce=%.4f total=%.4f",
                    alpha, beta, avg_mse, avg_nce, avg_loss)
        return avg_loss

    else:
        # Fallback: original MSE-only training (unchanged)
        flat = prebuild_flat_tensors(train_data, smiles_lookup)
        if flat["n_items"] == 0:
            return 0.0
        item_perm = np.random.permutation(flat["n_items"])
        total_loss = 0.0
        n_batches = 0
        row_ranges = flat["item_row_ranges"]

        for start in range(0, len(item_perm), batch_size):
            batch_item_ids = item_perm[start:start + batch_size]
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

            optimizer.zero_grad()
            loss = F.mse_loss(agg_pred, agg_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)
```

Update model construction in `main()` to pass `proj_dim`:

```python
model = AffinityGAT(
    esm2_input_dim=esm2_dim,
    proj_dim=args.proj_dim,
).to(device)
```

Update `train_epoch` call in main loop to pass new args:

```python
train_loss = train_epoch(
    model, graph, train_data, optimizer, device,
    batch_size=args.batch_size,
    smiles_tensor=smiles_tensor, smiles_lookup=smiles_lookup,
    epoch=epoch, max_epochs=args.epochs,
    proteins_per_batch=args.proteins_per_batch,
    pos_per_protein=args.pos_per_protein,
    hard_neg_per_protein=args.hard_neg_per_protein,
    mid_per_protein=args.mid_per_protein,
    pos_threshold=args.pos_threshold,
    neg_threshold=args.neg_threshold,
)
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/test_affinity_gat_multitask.py -v`

Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git add scripts/train_affinity_gat.py tests/test_affinity_gat_multitask.py
git commit -m "feat: multi-task training loop with protein-centric batching and curriculum"
```

---

### Task 4: Run Full Test Suite + Push

**Files:**
- All modified files

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && PYTHONPATH=src:. python -m pytest tests/ -v --tb=short`

Expected: All tests PASS (existing + new).

- [ ] **Step 2: Push to remote**

```bash
cd /Users/basar_temiz/Desktop/IDP\ work/alphafold
git push origin alphafold
```

- [ ] **Step 3: Verify**

Run: `cd /Users/basar_temiz/Desktop/IDP\ work/alphafold && git log --oneline -5`

Expected: 3 new commits on alphafold branch.
