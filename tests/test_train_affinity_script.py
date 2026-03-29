import importlib

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch, Data


class DummyAffinityDataset:
    protein_ids = ["P1", "P2", "P3"]

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        protein = Data(
            x_3di=torch.randint(0, 20, (3, 8)),
            x_seq_lens=torch.full((3,), 8, dtype=torch.long),
            edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
            edge_attr=torch.ones(4, 1),
            center_mask=torch.tensor([True, False, False]),
        )
        protein.num_nodes = 3

        drug = Data(
            x=torch.ones(2, 78),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        drug.num_nodes = 2
        return protein, drug, float(idx), self.protein_ids[idx]


def test_create_loaders_keeps_small_validation_split():
    module = importlib.import_module("scripts.train_affinity")
    module.torch = torch
    module.DataLoader = DataLoader
    module.Batch = Batch

    train_loader, val_loader = module.create_loaders(
        DummyAffinityDataset(),
        batch_size=64,
        seed=42,
    )

    assert len(train_loader) == 1
    assert len(val_loader) == 1


def test_build_parser_accepts_multitask_v2():
    module = importlib.import_module("scripts.train_affinity")
    args = module.build_parser().parse_args([
        "--model-version", "multitask_v2",
        "--drug-cache-workers", "8",
    ])

    assert args.model_version == "multitask_v2"
    assert args.drug_cache_workers == 8


def test_build_parser_accepts_rank_v2():
    module = importlib.import_module("scripts.train_affinity")
    args = module.build_parser().parse_args([
        "--model-version", "rank_v2",
        "--ranking-weight", "0.75",
        "--ranking-margin", "0.3",
    ])

    assert args.model_version == "rank_v2"
    assert args.ranking_weight == 0.75
    assert args.ranking_margin == 0.3
