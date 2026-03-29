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
        assert model.proj_head is None


class TestInfoNCELoss:
    def test_infonce_perfect_separation(self):
        from idr_gat.model.affinity_gat import infonce_loss
        anchor = F.normalize(torch.randn(1, 128), dim=1)
        positive = anchor + torch.randn(1, 128) * 0.01
        negatives = torch.randn(5, 128)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=1)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        assert loss.item() < 0.5

    def test_infonce_no_negatives(self):
        from idr_gat.model.affinity_gat import infonce_loss
        anchor = torch.randn(1, 128)
        positive = torch.randn(1, 128)
        negatives = torch.zeros(0, 128)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        assert loss.item() >= 0

    def test_infonce_gradient_flows(self):
        from idr_gat.model.affinity_gat import infonce_loss
        anchor_raw = torch.randn(2, 128, requires_grad=True)
        positive_raw = torch.randn(2, 128, requires_grad=True)
        negatives_raw = torch.randn(8, 128, requires_grad=True)
        anchor = F.normalize(anchor_raw, dim=1)
        positive = F.normalize(positive_raw, dim=1)
        negatives = F.normalize(negatives_raw, dim=1)
        loss = infonce_loss(anchor, positive, negatives, temperature=0.07)
        loss.backward()
        assert anchor_raw.grad is not None
        assert positive_raw.grad is not None


class TestCurriculumLoss:
    def test_early_epoch_infonce_dominant(self):
        from idr_gat.model.affinity_gat import curriculum_weights
        alpha, beta = curriculum_weights(epoch=0, max_epochs=200)
        assert beta > alpha
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
        anchor_idx = torch.tensor([0, 3, 7, 12])
        smiles_idx = torch.randint(0, 50, (4, 100))
        targets = torch.tensor([8.0, 7.5, 4.0, 3.5])
        labels = torch.tensor([1, 1, 0, 0])

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
        assert model.head[0].weight.grad is not None
        assert model.proj_head.weight.grad is not None
        assert model.drug_encoder.proj.weight.grad is not None


class TestProteinBatchSampler:
    def _make_items(self):
        items = []
        for uid in ["P001", "P002", "P003"]:
            for i in range(10):
                items.append({"uniprot_id": uid, "smiles": f"{uid}_pos_{i}", "pki": 7.0 + i * 0.3,
                              "anchors": [{"anchor_node": 0, "anchor_tm": 0.9}]})
            for i in range(5):
                items.append({"uniprot_id": uid, "smiles": f"{uid}_mid_{i}", "pki": 5.5 + i * 0.3,
                              "anchors": [{"anchor_node": 0, "anchor_tm": 0.9}]})
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
        assert labels.count(1) == 8
        assert labels.count(0) == 8
        assert labels.count(-1) == 4

    def test_iterates_all_proteins(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        seen = set()
        for batch in sampler:
            for item in batch:
                seen.add(item["uniprot_id"])
        assert seen == {"P001", "P002", "P003"}

    def test_len(self):
        from idr_gat.data.protein_batch_sampler import ProteinBatchSampler
        items = self._make_items()
        sampler = ProteinBatchSampler(
            items, proteins_per_batch=2,
            pos_per_protein=4, hard_neg_per_protein=4, mid_per_protein=2,
            pos_threshold=7.0, neg_threshold=5.0,
        )
        assert len(sampler) == 2
