from dataclasses import dataclass, field
import os
from pathlib import Path


@dataclass
class Config:  # noqa: too-many-instance-attributes
    # Paths
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    graph_dir: Path = Path("data/graphs")
    model_dir: Path = Path("models")
    results_dir: Path = Path("results")

    # Graph construction
    foldseek_bin: str = "foldseek"
    cluster_tm_threshold: float = 0.6  # global clustering: merge conformations with TM-score >= this
    edge_tm_range: tuple[float, float] = (0.2, 0.6)  # edges for cluster pairs with TM in this range

    def __post_init__(self):
        """Create TM-threshold-specific directories for models/results/graphs
        so different experiments don't overwrite each other."""
        tm_tag = f"tm{self.cluster_tm_threshold:.1f}".replace(".", "")
        edge_tag = f"e{self.edge_tm_range[0]:.1f}-{self.edge_tm_range[1]:.1f}".replace(".", "")
        suffix = f"{tm_tag}_{edge_tag}"
        self.model_dir = Path("models") / suffix
        self.results_dir = Path("results") / suffix
        self.graph_dir = Path("data/graphs") / suffix
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.graph_dir.mkdir(parents=True, exist_ok=True)
    max_conformations_per_protein: int = 10  # KMeans clusters from MD trajectory

    # ProteinFlow
    proteinflow_tag: str = "20230102_stable"
    n_conformations: int = 10  # ANM conformations per protein (randomly sampled)

    # B-factor filtering
    b_factor_threshold: float = 50.0  # raised from 30 to include more proteins

    # AlphaFold integration
    alphafold_plddt_threshold: float = 70.0  # minimum mean pLDDT for AlphaFold structures
    excluded_uniprots_path: Path = Path("data/raw/excluded_test_uniprots_expanded.json")

    # ANM conformation generation
    anm_n_modes: int = 5
    anm_amplitudes: list[float] = field(default_factory=lambda: [1.0, 2.0, 3.0])
    anm_cutoff: float = 15.0  # ANM interaction cutoff in Angstroms

    # Model
    threedi_vocab_size: int = 21  # 20 3Di states + 1 padding token
    threedi_embed_dim: int = 64  # learned embedding dim per 3Di token
    hidden_dim: int = 512  # wider for more capacity
    embedding_dim: int = 128  # shared space dimension d
    gat_layers: int = 3
    gat_heads: int = 8
    dropout: float = 0.1

    # ESM-2 protein language model (optional, concatenated with 3Di)
    use_esm2: bool = True
    esm2_model_name: str = "esm2_t33_650M_UR50D"
    esm2_proj_dim: int = 256  # project ESM-2 down before concat with 3Di

    # Drug encoder
    drug_feature_dim: int = 256  # wider to match protein encoder
    drug_gnn_layers: int = 3

    # Training
    batch_size: int = 64  # A100 80GB — graph is smaller after TM-score dedup
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    temperature: float = 0.07  # InfoNCE temperature
    num_epochs: int = 400
    hard_neg_lambda: float = 0.5  # weight for semi-hard negative loss term

    # Inference
    subgraph_hops: int = 2

    # Device
    device: str = "cuda" if __import__("torch").cuda.is_available() else "cpu"

    # Reproducibility
    seed: int = 42


def set_deterministic(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
