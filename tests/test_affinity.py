"""Tests for the affinity prediction model components."""

import numpy as np
import torch
import pytest
from torch_geometric.data import Data, Batch


class TestGraphDTAAtomFeatures:
    def test_feature_dimension(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_atom_features
        from rdkit import Chem
        mol = Chem.MolFromSmiles("CCO")
        for atom in mol.GetAtoms():
            feat = graphdta_atom_features(atom)
            assert len(feat) == 78

    def test_one_hot_sum(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_atom_features
        from rdkit import Chem
        mol = Chem.MolFromSmiles("c1ccccc1")
        feat = graphdta_atom_features(mol.GetAtomWithIdx(0))
        assert sum(feat[:44]) == 1.0
        assert feat[77] == 1.0

    def test_normalization(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        data = graphdta_smiles_to_graph("CCO")
        for i in range(data.x.size(0)):
            feat_sum = data.x[i].sum().item()
            assert abs(feat_sum - 1.0) < 1e-5


class TestGraphDTASmilesToGraph:
    def test_basic(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        data = graphdta_smiles_to_graph("CCO")
        assert isinstance(data, Data)
        assert data.x.shape[1] == 78
        assert data.x.shape[0] == 3
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == 4

    def test_invalid_smiles(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        with pytest.raises(ValueError):
            graphdta_smiles_to_graph("INVALID_SMILES")

    def test_extended_smiles_stripped(self):
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        data = graphdta_smiles_to_graph("CCO |r,THB:something|")
        assert data.x.shape[0] == 3


class TestGraphDTADrugEncoder:
    def test_output_shape(self):
        from idr_gat.model.graphdta_drug_encoder import GraphDTADrugEncoder, graphdta_smiles_to_graph
        encoder = GraphDTADrugEncoder()
        graphs = [graphdta_smiles_to_graph("CCO"), graphdta_smiles_to_graph("c1ccccc1")]
        batch = Batch.from_data_list(graphs)
        out = encoder(batch)
        assert out.shape == (2, 128)

    def test_single_atom_molecule(self):
        from idr_gat.model.graphdta_drug_encoder import GraphDTADrugEncoder, graphdta_smiles_to_graph
        encoder = GraphDTADrugEncoder()
        data = graphdta_smiles_to_graph("[Na]")
        batch = Batch.from_data_list([data])
        out = encoder(batch)
        assert out.shape == (1, 128)


# ---------------------------------------------------------------------------
# AffinityHead + AffinityIDRGAT
# ---------------------------------------------------------------------------

class TestAffinityHead:
    def test_output_shape(self):
        from idr_gat.model.affinity import AffinityHead
        head = AffinityHead(input_dim=256)
        out = head(torch.randn(4, 128), torch.randn(4, 128))
        assert out.shape == (4,)

    def test_deterministic(self):
        from idr_gat.model.affinity import AffinityHead
        head = AffinityHead(input_dim=256)
        head.eval()
        p, d = torch.randn(2, 128), torch.randn(2, 128)
        assert torch.allclose(head(p, d), head(p, d))


class TestAffinityIDRGAT:
    def _make_dummy_protein_graph(self):
        n = 5
        return Data(
            x_3di=torch.randint(0, 20, (n, 50)),
            x_seq_lens=torch.full((n,), 50, dtype=torch.long),
            edge_index=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long),
            edge_attr=torch.rand(8, 1),
            center_mask=torch.tensor([True, False, False, False, False]),
        )

    def test_forward(self):
        from idr_gat.model.affinity import AffinityIDRGAT
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        model = AffinityIDRGAT(use_esm2=False)
        pb = Batch.from_data_list([self._make_dummy_protein_graph()])
        db = Batch.from_data_list([graphdta_smiles_to_graph("CCO")])
        assert model(pb, db).shape == (1,)

    def test_loss(self):
        from idr_gat.model.affinity import AffinityIDRGAT
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph
        model = AffinityIDRGAT(use_esm2=False)
        pb = Batch.from_data_list([self._make_dummy_protein_graph()])
        db = Batch.from_data_list([graphdta_smiles_to_graph("CCO")])
        loss = model.compute_loss(pb, db, torch.tensor([7.5]))
        assert loss.dim() == 0 and loss.item() >= 0


class TestAffinityIDRGATMultiTaskV2:
    def _make_dummy_protein_graph(self):
        n = 5
        return Data(
            x_3di=torch.randint(0, 20, (n, 50)),
            x_seq_lens=torch.full((n,), 50, dtype=torch.long),
            edge_index=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long),
            edge_attr=torch.rand(8, 1),
            center_mask=torch.tensor([True, False, False, False, False]),
        )

    def test_forward(self):
        from idr_gat.model.affinity_multitask import AffinityIDRGATMultiTaskV2
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph

        model = AffinityIDRGATMultiTaskV2(use_esm2=False, infonce_weight=0.2)
        pb = Batch.from_data_list([self._make_dummy_protein_graph()])
        db = Batch.from_data_list([graphdta_smiles_to_graph("CCO")])
        assert model(pb, db).shape == (1,)

    def test_compute_losses(self):
        from idr_gat.model.affinity_multitask import AffinityIDRGATMultiTaskV2
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph

        model = AffinityIDRGATMultiTaskV2(use_esm2=False, infonce_weight=0.2)
        pb = Batch.from_data_list([
            self._make_dummy_protein_graph(),
            self._make_dummy_protein_graph(),
        ])
        db = Batch.from_data_list([
            graphdta_smiles_to_graph("CCO"),
            graphdta_smiles_to_graph("c1ccccc1"),
        ])
        targets = torch.tensor([7.5, 6.0], dtype=torch.float32)
        losses = model.compute_losses(pb, db, targets)

        assert set(losses) == {"loss", "affinity_loss", "infonce_loss", "affinity_preds"}
        assert losses["loss"].dim() == 0
        assert losses["affinity_loss"].dim() == 0
        assert losses["infonce_loss"].dim() == 0
        assert losses["affinity_preds"].shape == (2,)


class TestAffinityIDRGATRankV2:
    def _make_dummy_protein_graph(self):
        n = 5
        return Data(
            x_3di=torch.randint(0, 20, (n, 50)),
            x_seq_lens=torch.full((n,), 50, dtype=torch.long),
            edge_index=torch.tensor([[0,1,1,2,2,3,3,4],[1,0,2,1,3,2,4,3]], dtype=torch.long),
            edge_attr=torch.rand(8, 1),
            center_mask=torch.tensor([True, False, False, False, False]),
        )

    def test_compute_losses(self):
        from idr_gat.model.affinity_rank import AffinityIDRGATRankV2
        from idr_gat.model.graphdta_drug_encoder import graphdta_smiles_to_graph

        model = AffinityIDRGATRankV2(use_esm2=False, ranking_weight=0.5, ranking_margin=0.2)
        pb = Batch.from_data_list([
            self._make_dummy_protein_graph(),
            self._make_dummy_protein_graph(),
            self._make_dummy_protein_graph(),
        ])
        db = Batch.from_data_list([
            graphdta_smiles_to_graph("CCO"),
            graphdta_smiles_to_graph("c1ccccc1"),
            graphdta_smiles_to_graph("CCN"),
        ])
        targets = torch.tensor([7.5, 6.0, 8.0], dtype=torch.float32)
        losses = model.compute_losses(pb, db, targets, protein_ids=["P1", "P1", "P2"])

        assert set(losses) == {"loss", "mse_loss", "ranking_loss", "affinity_preds"}
        assert losses["loss"].dim() == 0
        assert losses["mse_loss"].dim() == 0
        assert losses["ranking_loss"].dim() == 0
        assert losses["affinity_preds"].shape == (3,)


# ---------------------------------------------------------------------------
# AffinityDataset
# ---------------------------------------------------------------------------

class TestAffinityDataset:
    def _make_subgraphs(self):
        def _g():
            n = 3
            return Data(x_3di=torch.randint(0, 20, (n, 30)), x_seq_lens=torch.full((n,), 30, dtype=torch.long),
                        edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long),
                        edge_attr=torch.rand(4, 1), center_mask=torch.tensor([True, False, False]))
        return {"P12345": _g(), "P67890": _g()}

    def test_basic(self):
        from idr_gat.data.affinity_dataset import AffinityDataset
        ds = AffinityDataset(self._make_subgraphs(), [
            {"uniprot_id": "P12345", "ligand_smiles": "CCO", "binding_affinity": 10.0},
            {"uniprot_id": "P67890", "ligand_smiles": "c1ccccc1", "binding_affinity": 100.0},
            {"uniprot_id": "MISSING", "ligand_smiles": "CCO", "binding_affinity": 5.0},
        ])
        assert len(ds) == 2

    def test_pki_transform(self):
        from idr_gat.data.affinity_dataset import AffinityDataset
        ds = AffinityDataset(self._make_subgraphs(), [
            {"uniprot_id": "P12345", "ligand_smiles": "CCO", "binding_affinity": 1.0},
        ])
        _, _, pki, uid = ds[0]
        assert abs(pki - 9.0) < 1e-5
        assert uid == "P12345"

    def test_filters_invalid(self):
        from idr_gat.data.affinity_dataset import AffinityDataset
        ds = AffinityDataset(self._make_subgraphs(), [
            {"uniprot_id": "P12345", "ligand_smiles": "CCO", "binding_affinity": float("nan")},
            {"uniprot_id": "P12345", "ligand_smiles": "CCN", "binding_affinity": -5.0},
            {"uniprot_id": "P12345", "ligand_smiles": "CCF", "binding_affinity": 0.0},
            {"uniprot_id": "P12345", "ligand_smiles": "CCC", "binding_affinity": 10.0},
        ])
        assert len(ds) == 1

    def test_getitem_triple(self):
        from idr_gat.data.affinity_dataset import AffinityDataset
        ds = AffinityDataset(self._make_subgraphs(), [
            {"uniprot_id": "P12345", "ligand_smiles": "CCO", "binding_affinity": 50.0},
        ])
        pg, dg, pki, uid = ds[0]
        assert isinstance(pg, Data) and isinstance(dg, Data)
        assert dg.x.shape[1] == 78
        assert isinstance(pki, float)
        assert uid == "P12345"


# ---------------------------------------------------------------------------
# Concordance Index
# ---------------------------------------------------------------------------

class TestConcordanceIndex:
    def test_perfect(self):
        from idr_gat.evaluation.metrics import concordance_index
        assert concordance_index(np.array([9., 7., 5., 3.]), np.array([8.5, 6.5, 4.5, 2.5])) == 1.0

    def test_reversed(self):
        from idr_gat.evaluation.metrics import concordance_index
        assert concordance_index(np.array([9., 7., 5., 3.]), np.array([2.5, 4.5, 6.5, 8.5])) == 0.0

    def test_single(self):
        from idr_gat.evaluation.metrics import concordance_index
        assert np.isnan(concordance_index(np.array([5.0]), np.array([4.0])))


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------

class TestAffinityIntegration:
    def _make_subgraphs(self):
        def _g():
            n = 3
            return Data(x_3di=torch.randint(0, 20, (n, 30)), x_seq_lens=torch.full((n,), 30, dtype=torch.long),
                        edge_index=torch.tensor([[0,1,1,2],[1,0,2,1]], dtype=torch.long),
                        edge_attr=torch.rand(4, 1), center_mask=torch.tensor([True, False, False]))
        return {"P001": _g(), "P002": _g()}

    def test_full_pipeline(self):
        from idr_gat.model.affinity import AffinityIDRGAT
        from idr_gat.data.affinity_dataset import AffinityDataset
        ds = AffinityDataset(self._make_subgraphs(), [
            {"uniprot_id": "P001", "ligand_smiles": "CCO", "binding_affinity": 10.0},
            {"uniprot_id": "P001", "ligand_smiles": "c1ccccc1", "binding_affinity": 100.0},
            {"uniprot_id": "P002", "ligand_smiles": "CC(=O)O", "binding_affinity": 50.0},
        ])
        pg, dg, pki, protein_ids = zip(*[ds[i] for i in range(len(ds))])
        pb = Batch.from_data_list(pg)
        db = Batch.from_data_list(dg)
        targets = torch.tensor(pki, dtype=torch.float32)
        model = AffinityIDRGAT(use_esm2=False)
        loss = model.compute_loss(pb, db, targets, protein_ids=list(protein_ids))
        loss.backward()
        assert model.protein_encoder.projection.weight.grad is not None
        assert model.drug_encoder.fc.weight.grad is not None
        assert model.head.fc1.weight.grad is not None
