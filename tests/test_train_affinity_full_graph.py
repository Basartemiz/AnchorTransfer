import importlib

import torch
from torch_geometric.data import Data


def test_build_parser_accepts_rank_v2_full_graph_defaults():
    module = importlib.import_module("scripts.train_affinity_full_graph")
    args = module.build_parser().parse_args([
        "--model-version", "rank_v2",
        "--batch-size", "128",
        "--anchor-cache-workers", "3",
        "--foldseek-threads", "5",
        "--build-anchor-cache-only",
    ])

    assert args.model_version == "rank_v2"
    assert args.batch_size == 128
    assert args.anchor_cache_workers == 3
    assert args.foldseek_threads == 5
    assert args.build_anchor_cache_only is True


def test_normalize_protein_node_ranges_handles_lists_and_spans():
    module = importlib.import_module("scripts.train_affinity_full_graph")
    node_ranges = {
        "P1": [0, 2, 4],
        "P2": (5, 8),
    }

    normalized = module.normalize_protein_node_ranges(node_ranges)

    assert torch.equal(normalized["P1"], torch.tensor([0, 2, 4], dtype=torch.long))
    assert torch.equal(normalized["P2"], torch.tensor([5, 6, 7], dtype=torch.long))


def test_collate_fn_batches_drug_graphs_and_preserves_protein_ids():
    module = importlib.import_module("scripts.train_affinity_full_graph")
    drug_a = Data(x=torch.ones(2, 78), edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long))
    drug_a.num_nodes = 2
    drug_b = Data(x=torch.ones(3, 78), edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long))
    drug_b.num_nodes = 3

    protein_ids, drug_batch, targets = module.collate_fn([
        ("P1", drug_a, 7.0),
        ("P2", drug_b, 6.5),
    ])

    assert protein_ids == ["P1", "P2"]
    assert list(targets.tolist()) == [7.0, 6.5]
    assert drug_batch.num_graphs == 2


def test_aggregate_anchor_predictions_averages_over_anchor_predictions():
    module = importlib.import_module("scripts.train_affinity_full_graph")

    class DummyHead(torch.nn.Module):
        def forward(self, protein_emb, drug_emb):
            return protein_emb[:, 0] + drug_emb[:, 0]

    class DummyModel:
        def __init__(self):
            self.head = DummyHead()

    projected_nodes = torch.tensor(
        [
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=torch.float32,
    )
    training_anchor_cache = {
        "P1": {
            "anchor_indices": torch.tensor([0, 2, 4], dtype=torch.long),
            "anchor_weights": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
        },
        "P2": {
            "anchor_indices": torch.tensor([1, 3], dtype=torch.long),
            "anchor_weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
        },
    }
    drug_emb = torch.tensor([[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]], dtype=torch.float32)

    preds, pooled = module._aggregate_anchor_predictions(
        DummyModel(),
        projected_nodes,
        training_anchor_cache,
        ["P1", "P2", "P1"],
        drug_emb,
    )

    # P1 uses anchors 1,3,5 => average protein contribution 3.0
    assert torch.isclose(preds[0], torch.tensor(13.0))
    assert torch.isclose(preds[2], torch.tensor(33.0))
    # P2 uses anchors 2,4 => average protein contribution 3.0
    assert torch.isclose(preds[1], torch.tensor(23.0))
    assert torch.equal(pooled[0], torch.tensor([3.0, 3.0]))
    assert torch.equal(pooled[1], torch.tensor([3.0, 3.0]))
