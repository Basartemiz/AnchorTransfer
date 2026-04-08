import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.generate_benchmark_filter_ci_panels import (
    QUARTILES,
    AnchorTransferDTAv2,
    ConPlex,
    DeepDTAModel,
    EsmDTAModel,
    apply_protocol_filters,
    anchored_subset,
    build_anchor_maps,
    build_dtc_reference,
    build_sequences,
    load_benchmarks,
    load_embeddings,
    load_model,
    predict_conplex,
    predict_deepdta,
    predict_esm_dta,
    predict_v2,
)


def main():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seqs = build_sequences()
    emb = load_embeddings()
    dtc_ref = build_dtc_reference(seqs)
    bench = load_benchmarks()["GLASS"]
    sdf = apply_protocol_filters(bench, "filtered", seqs, emb, dtc_ref)
    strongest_uid, second_uid, weakest_uid, strongest_pki, all_uids = build_anchor_maps(sdf)
    anchor_df = anchored_subset(sdf, strongest_uid, second_uid).copy()
    anchor_df["anchor_pki"] = anchor_df["ligand_smiles"].map(strongest_pki)
    anchor_df = anchor_df.dropna(subset=["anchor_pki"]).copy()
    anchor_df["anchor_quartile"] = pd.qcut(
        anchor_df["anchor_pki"], 4, labels=QUARTILES, duplicates="drop"
    )

    v2 = load_model(AnchorTransferDTAv2(esm2_dim=480).to(device), "models/v2_dtc/best_model.pt", device)
    deepdta = load_model(DeepDTAModel().to(device), "models/deepdta_dtc/best_model.pt", device)
    conplex = load_model(ConPlex(esm2_dim=480).to(device), "models/conplex_dtc/best_model.pt", device)
    esm_dta = load_model(EsmDTAModel(esm2_dim=480).to(device), "models/esm_dta_dtc/best_model.pt", device)

    def oracle(uid, smi):
        anc = strongest_uid.get(smi)
        if anc == uid:
            anc = second_uid.get(smi)
        return anc

    def weakest(uid, smi):
        anc = weakest_uid.get(smi)
        if anc == uid:
            anc = second_uid.get(smi)
        return anc

    rng = random.Random(seed)

    def rand_anchor(uid, _smi):
        choices = [protein for protein in all_uids if protein != uid]
        return rng.choice(choices) if choices else None

    predictions = {
        "V2_oracle": predict_v2(anchor_df, emb, v2, oracle, device),
        "V2_weakest": predict_v2(anchor_df, emb, v2, weakest, device),
        "V2_random": predict_v2(anchor_df, emb, v2, rand_anchor, device),
        "DeepDTA": predict_deepdta(anchor_df, seqs, deepdta, device),
        "ConPlex": predict_conplex(anchor_df, emb, conplex, device),
        "ESM-DTA": predict_esm_dta(anchor_df, emb, esm_dta, device),
    }

    rows = []
    for model_name, pred_df in predictions.items():
        pred_df = pred_df.copy()
        pred_df["anchor_quartile"] = anchor_df.loc[pred_df.index, "anchor_quartile"]
        for quartile in QUARTILES:
            subset = pred_df[pred_df["anchor_quartile"].astype(str) == quartile]
            for uid, group in subset.groupby("uniprot_id"):
                rmse = math.sqrt(float(((group["pki"] - group["pred"]) ** 2).mean()))
                rows.append(
                    {
                        "model": model_name,
                        "quartile": quartile,
                        "uniprot_id": uid,
                        "n": int(len(group)),
                        "rmse": rmse,
                    }
                )
    rmse_df = pd.DataFrame(rows)
    out = Path("results/benchmark_filter_ci_panels/glass_filtered_quartile_per_protein_rmse.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    rmse_df.to_csv(out, index=False)
    summary = rmse_df.groupby(["model", "quartile"])["rmse"].agg(["mean", "median"]).round(3)
    print(summary.to_string())
    print(f"\nWROTE {out}")


if __name__ == "__main__":
    main()
