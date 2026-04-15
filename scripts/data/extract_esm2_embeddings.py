#!/usr/bin/env python3
"""Extract ESM-2 mean-pooled embeddings for a set of protein sequences.

Wraps anchor_transfer.data.esm_encoder.encode_sequences() with a CLI.

Usage:
    python scripts/data/extract_esm2_embeddings.py \
        --input data/raw/dtc_proteins.csv \
        --output data/processed/esm2_35m_dtc.pt \
        --model esm2_t12_35M_UR50D \
        --device cuda \
        --batch-size 8
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--input", required=True,
                        help="CSV with columns: uniprot_id, sequence")
    parser.add_argument("--output", required=True,
                        help="Output .pt file (dict: uniprot_id -> tensor)")
    parser.add_argument("--model", default="esm2_t33_650M_UR50D",
                        help="ESM-2 model name (e.g., esm2_t12_35M_UR50D, esm2_t33_650M_UR50D)")
    parser.add_argument("--device", default="cpu",
                        help="torch device (cpu or cuda)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Sequences per forward pass")
    parser.add_argument("--max-seq-len", type=int, default=1022,
                        help="Truncate sequences longer than this")
    args = parser.parse_args()

    from anchor_transfer.data.esm_encoder import encode_sequences

    df = pd.read_csv(args.input)
    if "sequence" not in df.columns:
        raise ValueError(f"Input CSV must have 'sequence' column. Found: {list(df.columns)}")
    if "uniprot_id" not in df.columns:
        raise ValueError(f"Input CSV must have 'uniprot_id' column. Found: {list(df.columns)}")

    sequences = dict(zip(df["uniprot_id"].astype(str), df["sequence"].astype(str)))
    log.info("Loaded %d protein sequences from %s", len(sequences), args.input)

    embeddings_np = encode_sequences(
        sequences,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )

    # Convert numpy arrays to tensors for saving
    embeddings = {pid: torch.from_numpy(emb) for pid, emb in embeddings_np.items()}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(embeddings, out_path)
    log.info("Saved %d embeddings to %s", len(embeddings), out_path)


if __name__ == "__main__":
    main()
