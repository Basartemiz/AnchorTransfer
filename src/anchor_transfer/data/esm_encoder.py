# src/idr_gat/data/esm_encoder.py
"""ESM-2 protein language model encoder for sequence-level embeddings."""
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

# Model name → (repr_layer, embedding_dim)
ESM2_CONFIGS = {
    "esm2_t6_8M_UR50D": (6, 320),
    "esm2_t12_35M_UR50D": (12, 480),
    "esm2_t30_150M_UR50D": (30, 640),
    "esm2_t33_650M_UR50D": (33, 1280),
    "esm2_t36_3B_UR50D": (36, 2560),
}

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common non-standard
    "MSE": "M", "SEC": "C", "PYL": "K",
}


def residue_names_to_sequence(residue_names: list[str]) -> str:
    """Convert 3-letter residue names to 1-letter amino acid sequence."""
    return "".join(AA_3TO1.get(r.upper(), "X") for r in residue_names)


def get_esm2_config(model_name: str) -> tuple[int, int]:
    """Return (repr_layer, embed_dim) for a given ESM-2 model name."""
    if model_name not in ESM2_CONFIGS:
        raise ValueError(
            f"Unknown ESM-2 model: {model_name}. "
            f"Available: {list(ESM2_CONFIGS.keys())}"
        )
    return ESM2_CONFIGS[model_name]


def load_esm2_model(model_name: str = "esm2_t33_650M_UR50D"):
    """Load ESM-2 model and alphabet via the fair-esm package.

    Returns:
        model: ESM-2 model in eval mode
        batch_converter: function to convert (label, sequence) pairs to tokens
        repr_layer: which layer to extract representations from
    """
    try:
        import esm
    except ImportError:
        raise ImportError(
            "ESM-2 requires the 'fair-esm' package. "
            "Install with: pip install fair-esm"
        )

    repr_layer, _ = get_esm2_config(model_name)
    model_fn = getattr(esm.pretrained, model_name)
    model, alphabet = model_fn()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    logger.info(f"Loaded ESM-2 model: {model_name} (repr_layer={repr_layer})")
    return model, batch_converter, repr_layer


def encode_sequences(
    sequences: dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cpu",
    batch_size: int = 4,
    max_seq_len: int = 1022,
) -> dict[str, np.ndarray]:
    """Compute mean-pooled ESM-2 embeddings for a set of protein sequences.

    Args:
        sequences: dict mapping protein_id → amino acid sequence string
        model_name: ESM-2 model variant to use
        device: torch device
        batch_size: number of sequences per forward pass
        max_seq_len: truncate sequences longer than this (ESM-2 max is 1022
            for the 650M model; longer sequences cause OOM on most GPUs)

    Returns:
        dict mapping protein_id → (embed_dim,) numpy array
    """
    model, batch_converter, repr_layer = load_esm2_model(model_name)
    model = model.to(device)

    # Sort by length so batches have similar-length sequences (less padding waste)
    items = sorted(sequences.items(), key=lambda x: len(x[1]))
    embeddings = {}

    for start in range(0, len(items), batch_size):
        batch_items = items[start:start + batch_size]

        # Truncate long sequences and adjust batch size dynamically
        max_len_in_batch = max(len(seq) for _, seq in batch_items)
        if max_len_in_batch > max_seq_len:
            # Process long sequences one at a time
            for pid, seq in batch_items:
                trunc_seq = seq[:max_seq_len]
                trunc_len = len(trunc_seq)
                data = [(pid, trunc_seq)]
                _, _, tokens = batch_converter(data)
                tokens = tokens.to(device)
                with torch.no_grad():
                    results = model(tokens, repr_layers=[repr_layer])
                    reps = results["representations"][repr_layer]
                residue_reps = reps[0, 1:trunc_len + 1, :]
                embeddings[pid] = residue_reps.mean(dim=0).cpu().numpy()
                torch.cuda.empty_cache()
        else:
            data = [(pid, seq) for pid, seq in batch_items]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[repr_layer])
                reps = results["representations"][repr_layer]  # (B, L+2, D)

            for i, (pid, seq) in enumerate(batch_items):
                seq_len = len(seq)
                residue_reps = reps[i, 1:seq_len + 1, :]  # (L, D)
                embeddings[pid] = residue_reps.mean(dim=0).cpu().numpy()

        if (start + batch_size) % 20 == 0 or start + batch_size >= len(items):
            logger.info(f"ESM-2: encoded {min(start + batch_size, len(items))}/{len(items)} proteins")

    return embeddings
