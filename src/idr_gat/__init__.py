"""IDR-GAT: Structural knowledge transfer for IDP drug interaction prediction."""

__version__ = "1.0.0"

from idr_gat.config import Config

# Keep the package importable in lightweight evaluation environments where
# optional training-time dependencies such as torch_geometric are unavailable.
try:
    from idr_gat.model.affinity_gat import AffinityGAT
except ModuleNotFoundError as exc:
    if exc.name != "torch_geometric":
        raise
    AffinityGAT = None
