"""IDR-GAT: Structural knowledge transfer for IDP drug interaction prediction."""

__version__ = "1.0.0"

from idr_gat.config import Config

# Keep the package importable in lightweight evaluation environments where
# optional training-time dependencies such as torch_geometric are unavailable.
try:
    from idr_gat.model.contrastive import IDRGAT
    from idr_gat.model.protein_encoder import ProteinGATEncoder
    from idr_gat.model.drug_encoder import DrugEncoder
    from idr_gat.model.affinity import AffinityIDRGAT
    from idr_gat.model.affinity_multitask import AffinityIDRGATMultiTaskV2
except ModuleNotFoundError as exc:
    if exc.name != "torch_geometric":
        raise
    IDRGAT = None
    ProteinGATEncoder = None
    DrugEncoder = None
    AffinityIDRGAT = None
    AffinityIDRGATMultiTaskV2 = None
