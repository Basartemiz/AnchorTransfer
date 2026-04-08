#!/usr/bin/env python3
"""Prepare Raygun embeddings and Morgan fingerprints for BindingDB.

Generates the caches needed by ConciseAnchor training:
  - results/esm2_bdb_embeddings.pt   (ESM-2 650M per-residue, ~7 GB)
  - results/raygun_bdb_embeddings.pt (Raygun 50-token, ~700 MB)
  - results/concise_bdb_morgan_fp.pkl (Morgan FP 2048-bit, ~2 GB)

These caches are reused by train_concise_anchor_bdb.py and eval_bdb_to_davis.py.
"""
import os, sys, json, logging, pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger()

DATA_DIR = Path(os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

log.info("Loading BindingDB...")
bdb = pd.read_csv(DATA_DIR / "processed" / "bindingdb_interactions.csv")
seqs = json.load(open(DATA_DIR / "processed" / "merged_sequences.json"))
log.info(f"BDB: {len(bdb)}, Seqs: {len(seqs)}")

os.makedirs("results", exist_ok=True)

# ============================================================
# 1. ESM-2 650M embeddings
# ============================================================
ESM_CACHE = Path("results/esm2_bdb_embeddings.pt")
RAYGUN_CACHE = Path("results/raygun_bdb_embeddings.pt")

if RAYGUN_CACHE.exists():
    log.info(f"Raygun cache exists at {RAYGUN_CACHE} — skipping ESM-2 + Raygun")
else:
    if ESM_CACHE.exists():
        log.info(f"Loading cached ESM-2 embeddings from {ESM_CACHE}")
        esm_embeddings = torch.load(ESM_CACHE, map_location="cpu", weights_only=False)
    else:
        log.info("Computing ESM-2 650M embeddings...")
        import esm

        esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        bc = esm_alphabet.get_batch_converter()
        esm_model = esm_model.to(DEVICE)
        esm_model.eval()

        all_prots = sorted(set(bdb.uniprot_id) & set(seqs.keys()))
        log.info(f"Computing ESM-2 650M for {len(all_prots)} proteins...")

        esm_embeddings = {}
        with torch.no_grad():
            for i, uid in enumerate(all_prots):
                seq = seqs[uid][:1022]
                _, _, tokens = bc([(uid, seq)])
                emb = esm_model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
                esm_embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
                if (i + 1) % 100 == 0:
                    log.info(f"  ESM-2: {i+1}/{len(all_prots)}")

        del esm_model
        torch.cuda.empty_cache()
        torch.save(esm_embeddings, ESM_CACHE)
        log.info(f"Cached {len(esm_embeddings)} ESM-2 embeddings to {ESM_CACHE}")

    # Raygun encoding
    log.info("Running Raygun encoder...")
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(DEVICE)
    raymodel.eval()

    raygun_embs = {}
    skipped = 0
    with torch.no_grad():
        for i, (uid, emb) in enumerate(esm_embeddings.items()):
            try:
                ray_enc = raymodel.encoder(emb.to(DEVICE)).squeeze().cpu()
                if ray_enc.dim() == 2 and ray_enc.size(0) == 50:
                    raygun_embs[uid] = ray_enc
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log.warning(f"  Skipped {uid} (seq len {emb.size(1)}): {e}")
            if (i + 1) % 100 == 0:
                log.info(f"  Raygun: {i+1}/{len(esm_embeddings)} (skipped {skipped})")

    del raymodel, esm_embeddings
    torch.cuda.empty_cache()
    torch.save(raygun_embs, RAYGUN_CACHE)
    log.info(f"Saved {len(raygun_embs)} Raygun embeddings to {RAYGUN_CACHE}")

# ============================================================
# 2. Morgan fingerprints
# ============================================================
from molfeat.trans.fp import FPVecTransformer

FP_CACHE = Path("results/concise_bdb_morgan_fp.pkl")
if FP_CACHE.exists():
    log.info(f"Morgan FP cache exists at {FP_CACHE} — skipping")
else:
    log.info("Computing Morgan fingerprints...")
    all_smiles = sorted(set(bdb.ligand_smiles.unique()))
    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for i, smi in enumerate(all_smiles):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except Exception:
            pass
        if (i + 1) % 20000 == 0:
            log.info(f"  FP: {i+1}/{len(all_smiles)}")
    log.info(f"Computed {len(fp_dict)} fingerprints")
    with open(FP_CACHE, "wb") as f:
        pickle.dump(fp_dict, f)
    log.info(f"Saved Morgan FP cache to {FP_CACHE}")

log.info("=== BDB cache preparation complete ===")
