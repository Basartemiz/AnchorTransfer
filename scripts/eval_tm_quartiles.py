"""Evaluate localization models by TM-score quartiles."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

CLASSES = [
    "Nucleus", "Cytoplasm", "Extracellular", "Mitochondrion",
    "Cell.membrane", "Endoplasmic.reticulum", "Plastid",
    "Golgi.apparatus", "Lysosome/Vacuole", "Peroxisome",
]


def load_data(graph_dir="data/graphs/tm08_e03-08"):
    le = LabelEncoder()
    le.fit(CLASSES)
    anchors = json.load(open("results/tm08_e03-08/localization_finetune/anchors.json"))
    meta = json.load(open("data/processed/metadata.json"))
    c2p = meta["conformation_to_protein"]
    pseqs = meta["protein_sequences"]
    df = pd.read_csv("data/processed/deeploc/deeploc_data.csv")
    g = torch.load(f"{graph_dir}/global_graph.pt", map_location="cpu", weights_only=False)
    cids = list(g.conformation_ids)
    c2i = {n: i for i, n in enumerate(cids)}

    labels, tms, nidxs, accs = [], [], [], []
    for _, row in df.iterrows():
        acc = row["accession"]
        if acc not in anchors:
            continue
        tgt, tm = anchors[acc]
        if tgt not in c2i:
            continue
        pid = c2p.get(tgt)
        if pid is None or pid not in pseqs:
            continue
        labels.append(le.transform([row["location"]])[0])
        tms.append(tm)
        nidxs.append(c2i[tgt])
        accs.append(acc)

    labels = np.array(labels)
    tms = np.array(tms)
    nidxs = np.array(nidxs)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return labels, tms, nidxs, train_idx, test_idx, g


def eval_by_quartiles(test_labels, test_tms, preds, model_name):
    # Use actual quartiles of the data
    q25, q50, q75 = np.percentile(test_tms, [25, 50, 75])
    bins = [
        (0, q25, f"Q1 [0-{q25:.2f})"),
        (q25, q50, f"Q2 [{q25:.2f}-{q50:.2f})"),
        (q50, q75, f"Q3 [{q50:.2f}-{q75:.2f})"),
        (q75, 1.01, f"Q4 [{q75:.2f}-1.0]"),
    ]

    print(f"\n=== {model_name} ===")
    print(f"{'TM Range':<25} {'N_test':<8} {'Acc':<8} {'F1':<8}")
    print("-" * 50)
    acc = accuracy_score(test_labels, preds)
    f1 = f1_score(test_labels, preds, average="macro", zero_division=0)
    print(f"{'ALL':<25} {len(test_labels):<8} {acc:<8.4f} {f1:<8.4f}")
    for lo, hi, label in bins:
        mask = (test_tms >= lo) & (test_tms < hi)
        n = mask.sum()
        if n >= 2:
            qa = accuracy_score(test_labels[mask], preds[mask])
            qf = f1_score(test_labels[mask], preds[mask], average="macro", zero_division=0)
            print(f"{label:<25} {n:<8} {qa:<8.4f} {qf:<8.4f}")
        else:
            print(f"{label:<25} {n:<8} {'N/A':<8}")


def main():
    import torch.nn.functional as F
    from idr_gat.config import Config
    from idr_gat.model.contrastive import IDRGAT
    from train_localization import LocalizationClassifier
    from train_localization_v2 import LocalizationGATModel

    labels, tms, nidxs, train_idx, test_idx, g = load_data()
    test_labels = labels[test_idx]
    test_tms = tms[test_idx]

    print(f"TM-score distribution: min={test_tms.min():.3f} Q1={np.percentile(test_tms,25):.3f} "
          f"median={np.median(test_tms):.3f} Q3={np.percentile(test_tms,75):.3f} max={test_tms.max():.3f}")

    # --- V1: Pretrained encoder + classification head ---
    config = Config()
    ckpt_v1 = torch.load("results/tm08_e03-08/localization_finetune/best_model.pt",
                         map_location="cuda", weights_only=False)
    # Get pretrained encoder
    contrastive_ckpt = torch.load("models/tm08_e03-08/best_model.pt", map_location="cpu", weights_only=False)
    state = contrastive_ckpt["model_state_dict"]
    uses_esm2 = any("esm2" in k for k in state.keys())
    idrgat = IDRGAT(
        threedi_embed_dim=config.threedi_embed_dim, threedi_vocab_size=config.threedi_vocab_size,
        hidden_dim=config.hidden_dim, embedding_dim=config.embedding_dim,
        gat_layers=config.gat_layers, gat_heads=config.gat_heads,
        drug_hidden_dim=config.drug_feature_dim, drug_gnn_layers=config.drug_gnn_layers,
        temperature=config.temperature, hard_neg_lambda=config.hard_neg_lambda,
        dropout=config.dropout, use_esm2=uses_esm2, esm2_input_dim=1280,
        esm2_proj_dim=config.esm2_proj_dim,
    )
    idrgat.load_state_dict(state, strict=False)

    # Compute node embeddings with pretrained encoder
    encoder = idrgat.protein_encoder.cuda().eval()
    data = g.cuda()
    data.batch = torch.zeros(data.x_3di.size(0), dtype=torch.long, device="cuda")
    with torch.no_grad():
        x = encoder.build_node_features(data)
        for conv, bn in zip(encoder.convs, encoder.bns):
            x = conv(x, data.edge_index, edge_attr=data.edge_attr)
            x = bn(x)
            x = F.elu(x)
        if hasattr(encoder, "projection"):
            x = encoder.projection(x)
        x = F.normalize(x, p=2, dim=1)
        node_embs_v1 = x

    v1_model = LocalizationClassifier(encoder, embedding_dim=config.embedding_dim, num_classes=10).cuda()
    v1_model.load_state_dict(ckpt_v1["model_state_dict"], strict=False)
    v1_model.eval()
    with torch.no_grad():
        test_nidx = torch.tensor(nidxs[test_idx], device="cuda")
        v1_preds = v1_model.forward_from_embeddings(node_embs_v1[test_nidx]).argmax(dim=1).cpu().numpy()

    eval_by_quartiles(test_labels, test_tms, v1_preds, "V1 (pretrained + head)")

    # Free GPU
    del v1_model, encoder, idrgat, node_embs_v1
    torch.cuda.empty_cache()

    # --- V2: Full 1024-dim from scratch ---
    v2_model = LocalizationGATModel(
        threedi_embed_dim=1024, esm2_proj_dim=1024, hidden_dim=1024, embedding_dim=1024,
    ).cuda()
    ckpt_v2 = torch.load("results/tm08_e03-08/localization_v2/best_model.pt",
                         map_location="cuda", weights_only=False)
    v2_model.load_state_dict(ckpt_v2["model_state_dict"])
    v2_model.eval()
    with torch.no_grad():
        node_embs_v2 = v2_model.forward_graph(data)
        test_nidx = torch.tensor(nidxs[test_idx], device="cuda")
        v2_preds = v2_model.classify(node_embs_v2[test_nidx]).argmax(dim=1).cpu().numpy()

    eval_by_quartiles(test_labels, test_tms, v2_preds, "V2 (1024-dim, from scratch)")


if __name__ == "__main__":
    main()
