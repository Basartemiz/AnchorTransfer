import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data
from rdkit import Chem


ATOM_FEATURE_DIM = 9


def atom_features(atom) -> list[float]:
    """Extract atom features as a float vector."""
    return [
        atom.GetAtomicNum() / 118.0,
        atom.GetDegree() / 5.0,
        atom.GetFormalCharge() / 2.0,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
        atom.GetTotalNumHs() / 4.0,
        atom.GetNumRadicalElectrons() / 2.0,
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2),
        float(atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3),
    ]


def smiles_to_graph(smiles: str) -> Data:
    """Convert SMILES string to PyG graph."""
    # Strip extended SMILES notation (e.g. " |r,THB:...|") that RDKit can't parse
    smiles = smiles.split(" |")[0].strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    x = torch.tensor(
        [atom_features(atom) for atom in mol.GetAtoms()],
        dtype=torch.float32,
    )

    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index)


class DrugEncoder(nn.Module):
    """GIN-based drug molecular graph encoder."""

    def __init__(
        self,
        atom_feature_dim: int = ATOM_FEATURE_DIM,
        hidden_dim: int = 128,
        embedding_dim: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        mlp = nn.Sequential(
            nn.Linear(atom_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.convs.append(GINConv(mlp))
        self.bns.append(nn.BatchNorm1d(hidden_dim))

        for _ in range(num_layers - 1):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, "batch") else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=1)
        return x
