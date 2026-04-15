#!/usr/bin/env python3
"""Prepare BindingDB caches for ConciseAnchor training.

Computes:
  1. ESM-2 650M embeddings for all BDB proteins (cached to results/esm2_bdb_embeddings.pt)
  2. Raygun protein representations (cached to results/raygun_bdb_embeddings.pt)
  3. Morgan fingerprints for all BDB drugs (cached to results/concise_bdb_morgan_fp.pkl)

All outputs are cached: rerunning skips completed steps.

Usage:
    python scripts/data/prepare_bdb_caches.py
"""
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("DATA_DIR", str(PROJECT / "data")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ESM_CACHE = Path("results/esm2_bdb_embeddings.pt")
RAYGUN_CACHE = Path("results/raygun_bdb_embeddings.pt")
FP_CACHE = Path("results/concise_bdb_morgan_fp.pkl")


def compute_esm2_embeddings(protein_ids, sequences):
    """Compute ESM-2 650M per-residue embeddings for BDB proteins."""
    if ESM_CACHE.exists():
        log.info("Loading cached ESM-2 embeddings from %s", ESM_CACHE)
        return torch.load(ESM_CACHE, map_location="cpu", weights_only=False)

    log.info("Computing ESM-2 650M embeddings for %d proteins...", len(protein_ids))
    import esm

    esm_model, esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    bc = esm_alphabet.get_batch_converter()
    esm_model = esm_model.to(DEVICE)
    esm_model.eval()

    embeddings = {}
    with torch.no_grad():
        for i, uid in enumerate(tqdm(protein_ids, desc="ESM-2")):
            seq = sequences[uid][:1022]
            _, _, tokens = bc([(uid, seq)])
            emb = esm_model(tokens.to(DEVICE), repr_layers=[33], return_contacts=False)
            embeddings[uid] = emb["representations"][33][:, 1:-1, :].cpu()
            if (i + 1) % 100 == 0:
                log.info("  ESM-2: %d/%d", i + 1, len(protein_ids))

    del esm_model
    torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    torch.save(embeddings, ESM_CACHE)
    log.info("Cached %d ESM-2 embeddings to %s", len(embeddings), ESM_CACHE)
    return embeddings


def compute_raygun_embeddings(esm_embeddings):
    """Compute Raygun encoder representations from ESM-2 per-residue embeddings."""
    if RAYGUN_CACHE.exists():
        log.info("Loading cached Raygun embeddings from %s", RAYGUN_CACHE)
        return torch.load(RAYGUN_CACHE, map_location="cpu", weights_only=False)

    log.info("Computing Raygun embeddings for %d proteins...", len(esm_embeddings))
    raymodel, _, _ = torch.hub.load("rohitsinghlab/raygun", "pretrained_uniref50_95000_750M")
    raymodel = raymodel.to(DEVICE)
    raymodel.eval()

    raygun_embs = {}
    skipped = 0
    with torch.no_grad():
        for i, (uid, emb) in enumerate(tqdm(esm_embeddings.items(), desc="Raygun")):
            try:
                ray_enc = raymodel.encoder(emb.to(DEVICE)).squeeze().cpu()
                if ray_enc.dim() == 2 and ray_enc.size(0) == 50:
                    raygun_embs[uid] = ray_enc
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log.warning("  Skipped %s (seq len %d): %s", uid, emb.size(1), e)
            if (i + 1) % 100 == 0:
                log.info("  Raygun: %d/%d (skipped %d)", i + 1, len(esm_embeddings), skipped)

    del raymodel, esm_embeddings
    torch.cuda.empty_cache()

    torch.save(raygun_embs, RAYGUN_CACHE)
    log.info("Saved %d Raygun embeddings to %s (skipped %d)", len(raygun_embs), RAYGUN_CACHE, skipped)
    return raygun_embs


def compute_morgan_fps(smiles_list):
    """Compute Morgan/ECFP4 fingerprints for all BDB drugs."""
    if FP_CACHE.exists():
        log.info("Loading cached fingerprints from %s", FP_CACHE)
        with open(FP_CACHE, "rb") as f:
            return pickle.load(f)

    log.info("Computing Morgan fingerprints for %d drugs...", len(smiles_list))
    from molfeat.trans.fp import FPVecTransformer

    transformer = FPVecTransformer(kind="ecfp:4", length=2048, verbose=False)
    fp_dict = {}
    for i, smi in enumerate(tqdm(smiles_list, desc="Morgan FP")):
        try:
            fp = transformer(smi)
            if fp is not None and len(fp) > 0:
                fp_dict[smi] = np.array(fp[0], dtype=np.float32)
        except Exception:
            pass
        if (i + 1) % 20000 == 0:
            log.info("  FP: %d/%d", i + 1, len(smiles_list))

    log.info("Computed %d fingerprints", len(fp_dict))
    os.makedirs("results", exist_ok=True)
    with open(FP_CACHE, "wb") as f:
        pickle.dump(fp_dict, f)
    return fp_dict


def main():
    import pandas as pd

    log.info("Loading BindingDB data...")
    bdb = pd.read_csv(DATA_DIR / "processed" / "bindingdb_interactions.csv")
    seqs = json.load(open(DATA_DIR / "processed" / "merged_sequences.json"))
    log.info("BDB: %d interactions, Sequences: %d proteins", len(bdb), len(seqs))

    # Step 1: ESM-2 embeddings
    protein_ids = sorted(set(bdb.uniprot_id) & set(seqs.keys()))
    esm_embs = compute_esm2_embeddings(protein_ids, seqs)

    # Step 2: Raygun embeddings
    raygun_embs = compute_raygun_embeddings(esm_embs)
    log.info("Raygun embeddings: %d proteins, dim=%s",
             len(raygun_embs), next(iter(raygun_embs.values())).shape)

    # Step 3: Morgan fingerprints
    all_smiles = sorted(set(bdb.ligand_smiles.unique()))
    fp_dict = compute_morgan_fps(all_smiles)
    log.info("Morgan FP dict: %d entries", len(fp_dict))

    log.info("All BDB caches ready.")


if __name__ == "__main__":
    main()
