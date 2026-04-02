#!/usr/bin/env python3
"""Compute ProstT5 embeddings on GPU with a GPU-aware loading path."""
import json
import logging
import os

import pandas as pd
import torch
from transformers import T5EncoderModel, T5Tokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "true")
os.environ.setdefault("HF_PARALLEL_LOADING_WORKERS", "8")

seqs = json.load(open("data/processed/dtc_sequences.json"))
for _, r in pd.read_csv("data/raw/davis/davis_benchmark.csv").drop_duplicates("protein_name").iterrows():
    seqs[r["protein_name"]] = r["protein_sequence"]
logger.info("Total sequences: %d", len(seqs))

logger.info("Loading ProstT5 (FP16)...")
tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False)
device = torch.device("cuda")
model = T5EncoderModel.from_pretrained(
    "Rostlab/ProstT5",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="cuda:0",
    use_safetensors=True,
)
model = model.eval()
logger.info("ProstT5 loaded; hf_device_map=%s", getattr(model, "hf_device_map", None))
logger.info("GPU memory after load: %.1f MiB", torch.cuda.memory_allocated() / 1024**2)

embeddings = {}
items = list(seqs.items())
batch_size = 32

for i in range(0, len(items), batch_size):
    batch = items[i:i+batch_size]
    batch_seqs = [" ".join(list(seq[:512])) for _, seq in batch]
    batch_ids = [uid for uid, _ in batch]

    ids = tokenizer.batch_encode_plus(batch_seqs, add_special_tokens=True,
                                       padding="longest", return_tensors="pt")
    input_ids = ids["input_ids"].to(device)
    attention_mask = ids["attention_mask"].to(device)

    with torch.no_grad(), torch.amp.autocast("cuda"):
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        for j, uid in enumerate(batch_ids):
            mask = attention_mask[j].bool()
            emb = out.last_hidden_state[j][mask].mean(0).float().cpu()
            embeddings[uid] = emb

    if (i // batch_size + 1) % 50 == 0:
        logger.info("Progress: %d/%d", min(i+batch_size, len(items)), len(items))

torch.save(embeddings, "data/processed/prostt5_all.pt")
logger.info("Saved %d ProstT5 embeddings (dim=%d)", len(embeddings), list(embeddings.values())[0].shape[0])
