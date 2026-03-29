"""Tests for binding domain identification."""
import pytest


class TestDomainSequenceMapping:
    def test_find_domain_in_full_sequence(self):
        from idr_gat.data.binding_domain import find_domain_range
        full_seq = "AAAAGGGGGCCCCCTTTTTT"
        domain_seq = "GGGGG"
        start, end = find_domain_range(domain_seq, full_seq)
        assert start == 4
        assert end == 8  # 0-indexed inclusive: positions 4,5,6,7,8

    def test_domain_not_found_returns_none(self):
        from idr_gat.data.binding_domain import find_domain_range
        result = find_domain_range("WWWWW", "AAAAA")
        assert result is None


class TestUniProtBindingSites:
    def test_parse_binding_features(self):
        from idr_gat.data.binding_domain import parse_uniprot_binding_sites
        features = [
            {"type": "Binding site", "location": {"start": {"value": 100}, "end": {"value": 105}}},
            {"type": "Active site", "location": {"start": {"value": 200}, "end": {"value": 200}}},
            {"type": "Helix", "location": {"start": {"value": 50}, "end": {"value": 80}}},
        ]
        sites = parse_uniprot_binding_sites(features)
        assert len(sites) == 2
        assert sites[0] == (100, 105)
        assert sites[1] == (200, 200)


class TestSelectBindingDomain:
    def test_selects_domain_overlapping_binding_site(self):
        from idr_gat.data.binding_domain import select_binding_domain
        domains = {
            "P_d0": {"seq": "AAAA", "range": (1, 4), "length": 4},
            "P_d1": {"seq": "BBBB", "range": (90, 110), "length": 21},
        }
        binding_sites = [(95, 105)]
        best = select_binding_domain(domains, binding_sites)
        assert best == "P_d1"

    def test_falls_back_to_largest_domain(self):
        from idr_gat.data.binding_domain import select_binding_domain
        domains = {
            "P_d0": {"seq": "AAAA", "range": (1, 4), "length": 4},
            "P_d1": {"seq": "B" * 200, "range": (10, 210), "length": 200},
        }
        binding_sites = []
        best = select_binding_domain(domains, binding_sites)
        assert best == "P_d1"
