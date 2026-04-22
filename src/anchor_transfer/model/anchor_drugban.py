"""Anchor-DrugBAN: DrugBAN with anchor transfer.

Reuses the upstream DrugBAN components verbatim (`MolecularGCN`, `ProteinCNN`,
`BANLayer` with weight_norm), runs bilinear attention for both (drug, anchor)
and (drug, query), then compares the two pooled interaction representations.

Architecture:
  1. Drug: MolecularGCN → per-atom features v_d  (upstream)
  2. Protein CNN (shared): anchor tokens → v_a, query tokens → v_q (upstream)
  3. Shared BANLayer (weight_norm'd): bilinear attention
     - (v_d, v_a) → f_anchor  (pooled interaction vector)
     - (v_d, v_q) → f_query
  4. Compare at the pooled level: [f_anchor, f_query, |diff|, f_anchor * f_query]
  5. MLPDecoder → pKi

Inputs mirror upstream DrugBAN: `bg_d` is a batched dgl.DGLGraph, the two
protein inputs are (B, L) int64 residue-index tensors.
"""
import torch
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from anchor_transfer.model.ban import BANLayer
from anchor_transfer.model.drugban import MLPDecoder, MolecularGCN, ProteinCNN


class AnchorDrugBAN(nn.Module):
    def __init__(self, **config):
        super(AnchorDrugBAN, self).__init__()
        drug_in_feats = config["DRUG"]["NODE_IN_FEATS"]
        drug_embedding = config["DRUG"]["NODE_IN_EMBEDDING"]
        drug_hidden_feats = config["DRUG"]["HIDDEN_LAYERS"]
        protein_emb_dim = config["PROTEIN"]["EMBEDDING_DIM"]
        num_filters = config["PROTEIN"]["NUM_FILTERS"]
        kernel_size = config["PROTEIN"]["KERNEL_SIZE"]
        mlp_in_dim = config["DECODER"]["IN_DIM"]
        mlp_hidden_dim = config["DECODER"]["HIDDEN_DIM"]
        mlp_out_dim = config["DECODER"]["OUT_DIM"]
        drug_padding = config["DRUG"]["PADDING"]
        protein_padding = config["PROTEIN"]["PADDING"]
        out_binary = config["DECODER"]["BINARY"]
        ban_heads = config["BCN"]["HEADS"]

        self.drug_extractor = MolecularGCN(
            in_feats=drug_in_feats, dim_embedding=drug_embedding,
            padding=drug_padding, hidden_feats=drug_hidden_feats,
        )
        # Shared protein encoder — anchor and query pass through the same weights.
        self.protein_extractor = ProteinCNN(
            protein_emb_dim, num_filters, kernel_size, protein_padding,
        )
        # Shared BANLayer — bilinear attention applied to (drug, anchor) and (drug, query).
        self.bcn = weight_norm(
            BANLayer(v_dim=drug_hidden_feats[-1], q_dim=num_filters[-1],
                     h_dim=mlp_in_dim, h_out=ban_heads),
            name='h_mat', dim=None,
        )
        # Comparison input: [f_a ∥ f_q ∥ |f_a − f_q| ∥ f_a ⊙ f_q] → 4 * mlp_in_dim.
        self.mlp_classifier = MLPDecoder(
            mlp_in_dim * 4, mlp_hidden_dim, mlp_out_dim, binary=out_binary,
        )

    def forward(self, bg_d, v_anchor, v_query, mode="train"):
        v_d = self.drug_extractor(bg_d)
        v_a = self.protein_extractor(v_anchor)
        v_q = self.protein_extractor(v_query)

        f_anchor, att_a = self.bcn(v_d, v_a)
        f_query, att_q = self.bcn(v_d, v_q)

        diff = torch.abs(f_anchor - f_query)
        prod = f_anchor * f_query
        fused = torch.cat([f_anchor, f_query, diff, prod], dim=-1)
        score = self.mlp_classifier(fused)

        if mode == "train":
            return v_d, v_a, v_q, f_anchor, f_query, score
        elif mode == "eval":
            return v_d, v_a, v_q, score, att_a, att_q
