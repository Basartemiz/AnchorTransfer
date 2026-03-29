"""Tests for domain-adapted DeepDTA model."""
import sys
from pathlib import Path

import torch
import pytest

# Add scripts/ to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class TestDomainDeepDTA:
    def test_forward_shape(self):
        from domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        smiles = torch.randint(0, 52, (4, 100))
        protein = torch.randint(0, 26, (4, 500))
        out = model(smiles, protein)
        assert out.shape == (4,)

    def test_shorter_protein_max_len(self):
        from domain_deepdta_model import DOMAIN_MAX_LEN
        assert DOMAIN_MAX_LEN == 500

    def test_smaller_kernels(self):
        from domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        assert model.protein_conv1.kernel_size == (3,)
        assert model.protein_conv2.kernel_size == (4,)
        assert model.protein_conv3.kernel_size == (6,)

    def test_gradient_flows(self):
        from domain_deepdta_model import DomainDeepDTA
        model = DomainDeepDTA()
        smiles = torch.randint(0, 52, (2, 100))
        protein = torch.randint(0, 26, (2, 500))
        out = model(smiles, protein)
        loss = out.sum()
        loss.backward()
        assert model.protein_embed.weight.grad is not None
