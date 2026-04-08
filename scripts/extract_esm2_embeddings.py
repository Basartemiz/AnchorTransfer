#!/usr/bin/env python3
"""Extract mean-pooled ESM-2 embeddings for a set of protein sequences.

Reads a CSV/TSV with columns (uniprot_id, sequence) and saves a dict
{uniprot_id: tensor(embed_dim,)} to a .pt file.

Usage:
  python scripts/extract_esm2_embeddings.py \
    --input data/raw/dtc_proteins.csv \
    --output data/processed/esm2_35m_dtc_proteins.pt \
    --model esm2_t12_35M_UR50D \
    --device cuda --batch-size 8
"""
import argparse
import logging
from pathlib import Path

import pandas as pd
import torch

from anchor_transfer.data.esm_encoder import encode_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Extract ESM-2 embeddings")
    parser.add_argument("--input", required=True, help="CSV with uniprot_id,sequence columns")
    parser.add_argument("--output", required=True, help="Output .pt file")
    parser.add_argument("--model", default="esm2_t12_35M_UR50D",
                        help="ESM-2 model name (default: 35M)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    sequences = dict(zip(df["uniprot_id"], df["sequence"]))
    logger.info("Loaded %d protein sequences", len(sequences))

    embeddings_np = encode_sequences(
        sequences, model_name=args.model,
        device=args.device, batch_size=args.batch_size,
    )

    embeddings = {pid: torch.from_numpy(emb) for pid, emb in embeddings_np.items()}
    torch.save(embeddings, args.output)
    logger.info("Saved %d embeddings to %s (dim=%d)",
                len(embeddings), args.output,
                next(iter(embeddings.values())).shape[0])


if __name__ == "__main__":
    main()
