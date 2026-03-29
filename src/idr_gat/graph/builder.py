import numpy as np
import torch
from torch_geometric.data import Data
from idr_gat.data.foldseek import PADDING_IDX


def threshold_edges(
    similarities: dict[tuple[int, int], float],
    threshold: float,
    threshold_high: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert sparse similarity dict to edge_index and edge_attr by thresholding.

    Excludes self-loops. Keeps edges with similarity >= threshold, or within
    [threshold, threshold_high) when threshold_high is provided.

    edge_attr is 1-dim and contains only the similarity score per edge.
    Each undirected edge (i, j) produces two directed edges (i->j, j->i).
    """
    rows, cols, weights = [], [], []
    for (i, j), score in similarities.items():
        if threshold_high is not None:
            if score < threshold or score >= threshold_high:
                continue
        else:
            if score <= threshold:
                continue
        # Bidirectional edges
        rows.extend([i, j])
        cols.extend([j, i])
        weights.extend([score, score])

    if not rows:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 1), dtype=torch.float32)
        return edge_index, edge_attr

    edge_index = torch.tensor([rows, cols], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr


def pad_3di_sequences(
    sequences: list[np.ndarray],
    padding_idx: int = PADDING_IDX,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length 3Di index arrays to a uniform tensor.

    Returns:
        x_3di: (N, max_len) int64 tensor, padded with padding_idx
        x_seq_lens: (N,) int64 tensor of actual lengths
    """
    max_len = max(len(s) for s in sequences)
    n = len(sequences)
    padded = np.full((n, max_len), padding_idx, dtype=np.int64)
    seq_lens = np.zeros(n, dtype=np.int64)
    for i, seq in enumerate(sequences):
        padded[i, :len(seq)] = seq
        seq_lens[i] = len(seq)
    return torch.tensor(padded, dtype=torch.long), torch.tensor(seq_lens, dtype=torch.long)


def build_conformation_graph(
    threedi_sequences: list[np.ndarray],
    similarities: dict[tuple[int, int], float],
    threshold: float,
    protein_ids: list[str],
    conformation_ids: list[str] | None = None,
    esm2_embeddings: dict[str, np.ndarray] | None = None,
    threshold_high: float | None = None,
) -> Data:
    """Build a PyG Data object for the global conformation graph.

    Args:
        threedi_sequences: list of int arrays (one per node), each containing
            3Di token indices (0-19).
        similarities: sparse dict mapping (i, j) -> TM-score
        threshold: minimum similarity for an edge
        protein_ids: protein ID for each node
        conformation_ids: optional conformation labels
        esm2_embeddings: optional dict mapping protein_id → ESM-2 embedding.
            Each conformation inherits its parent protein's embedding.
        threshold_high: optional exclusive upper bound for edge similarity
    """
    edge_index, edge_attr = threshold_edges(
        similarities,
        threshold,
        threshold_high=threshold_high,
    )
    x_3di, x_seq_lens = pad_3di_sequences(threedi_sequences)

    graph = Data(
        x_3di=x_3di,
        x_seq_lens=x_seq_lens,
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    graph.protein_ids = protein_ids
    graph.conformation_ids = conformation_ids or [f"conf_{i}" for i in range(len(protein_ids))]
    graph.num_nodes = len(protein_ids)

    # Attach ESM-2 embeddings (per domain if available, else per protein)
    if esm2_embeddings is not None:
        embed_dim = next(iter(esm2_embeddings.values())).shape[0]
        esm2_matrix = np.zeros((len(protein_ids), embed_dim), dtype=np.float32)
        cids = conformation_ids or [f"conf_{i}" for i in range(len(protein_ids))]
        for i, (pid, cid) in enumerate(zip(protein_ids, cids)):
            if cid in esm2_embeddings:
                esm2_matrix[i] = esm2_embeddings[cid]
            elif pid in esm2_embeddings:
                esm2_matrix[i] = esm2_embeddings[pid]
        graph.x_esm2 = torch.tensor(esm2_matrix, dtype=torch.float32)

    return graph
