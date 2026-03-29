import importlib

import pandas as pd


def test_threshold_binary_pairs_drops_ambiguous_rows():
    module = importlib.import_module("scripts.affinity_eval_utils")
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1", "P1"],
            "ligand_smiles": ["CCO", "CCN", "CCC"],
            "true_pki": [6.2, 4.8, 5.4],
            "pred_pki": [7.0, 3.0, 5.5],
        }
    )

    thresholded = module.threshold_binary_pairs(df)

    assert len(thresholded) == 2
    assert thresholded["ligand_smiles"].tolist() == ["CCO", "CCN"]
    assert thresholded["binary_label"].tolist() == [1, 0]


def test_compute_thresholded_classification_metrics_reports_counts():
    module = importlib.import_module("scripts.affinity_eval_utils")
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1", "P1", "P1", "P1"],
            "ligand_smiles": ["A", "B", "C", "D", "E"],
            "true_pki": [7.5, 6.1, 4.9, 4.2, 5.5],
            "pred_pki": [9.0, 8.0, 2.0, 1.0, 5.0],
        }
    )

    metrics = module.compute_thresholded_classification_metrics(df)

    assert metrics["n_pairs_total"] == 5
    assert metrics["n_pairs_evaluated"] == 4
    assert metrics["n_binders"] == 2
    assert metrics["n_non_binders"] == 2
    assert metrics["n_ambiguous"] == 1
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_compute_per_protein_thresholded_classification_means_across_proteins():
    module = importlib.import_module("scripts.affinity_eval_utils")
    df = pd.DataFrame(
        {
            "uniprot_id": ["P1", "P1", "P2", "P2", "P2"],
            "ligand_smiles": ["A", "B", "C", "D", "E"],
            "true_pki": [6.4, 4.7, 6.8, 4.5, 5.3],
            "pred_pki": [8.0, 2.0, 7.0, 3.0, 5.0],
        }
    )

    metrics = module.compute_per_protein_thresholded_classification(df)

    assert metrics["n_proteins_total"] == 2
    assert metrics["n_proteins_with_thresholded_pairs"] == 2
    assert metrics["n_proteins_evaluated"] == 2
    assert metrics["mean_auroc"] == 1.0
    assert metrics["mean_auprc"] == 1.0
