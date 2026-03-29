"""Tests for disorder entropy score computation."""
from __future__ import annotations

import numpy as np
import pytest


class TestThreeDiEntropy:
    def test_identical_conformations_zero_entropy(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        seqs = ["AAAAAA", "AAAAAA", "AAAAAA"]
        entropy = threedi_entropy(seqs)
        assert entropy == 0.0

    def test_max_entropy_all_different(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        alphabet = "ACDEFGHIKLMNPQRSTVWY"
        seqs = [c * 5 for c in alphabet]
        entropy = threedi_entropy(seqs)
        assert entropy > 0.9

    def test_partial_entropy(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        seqs = ["AA", "CA", "AA", "CA"]
        entropy = threedi_entropy(seqs)
        assert 0.0 < entropy < 1.0

    def test_single_seq_returns_zero(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        assert threedi_entropy(["AAAA"]) == 0.0

    def test_empty_returns_zero(self):
        from idr_gat.evaluation.disorder_score import threedi_entropy
        assert threedi_entropy([]) == 0.0


class TestDisorderScore:
    def test_score_range(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        score = compute_disorder_score(threedi_entropy=0.5, mean_plddt=0.7)
        assert 0.0 <= score <= 1.0

    def test_high_disorder(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        score = compute_disorder_score(threedi_entropy=0.9, mean_plddt=0.3)
        assert score > 0.7

    def test_low_disorder(self):
        from idr_gat.evaluation.disorder_score import compute_disorder_score
        score = compute_disorder_score(threedi_entropy=0.1, mean_plddt=0.9)
        assert score < 0.3


class TestBinByDisorder:
    def test_quartile_binning(self):
        from idr_gat.evaluation.disorder_score import bin_by_disorder
        scores = {"P1": 0.1, "P2": 0.3, "P3": 0.5, "P4": 0.7,
                  "P5": 0.2, "P6": 0.4, "P7": 0.6, "P8": 0.8}
        bins = bin_by_disorder(scores, n_bins=4)
        assert len(bins) == 4
        for label, uids in bins.items():
            assert len(uids) == 2

    def test_bin_labels_ordered(self):
        from idr_gat.evaluation.disorder_score import bin_by_disorder
        scores = {"P1": 0.1, "P2": 0.9}
        bins = bin_by_disorder(scores, n_bins=2)
        labels = list(bins.keys())
        assert "Q1" in labels[0]
        assert "Q2" in labels[1]
