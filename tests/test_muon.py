"""Muon / Newton-Schulz: the flagship algorithmic checks (CPU)."""

import pytest
import torch

from megakernels.optim import muon
from megakernels.optim import split_params, build_optimizers
from megakernels.model import Qwen3ForCausalLM
from megakernels.config import tiny


@pytest.mark.parametrize("shape", [(64, 64), (128, 32), (32, 128), (256, 48)])
def test_gram_equals_standard(shape):
    """Gram-NS must equal standard NS in (near-)exact arithmetic."""
    torch.manual_seed(0)
    G = torch.randn(*shape, dtype=torch.float64)
    std = muon.newton_schulz_standard(G)
    gram = muon.gram_newton_schulz(G)
    assert torch.allclose(std, gram, atol=1e-8, rtol=1e-6), \
        (std - gram).abs().max().item()


@pytest.mark.parametrize("shape", [(128, 32), (256, 64), (32, 128)])
def test_output_is_approximately_orthogonal(shape):
    """5-step NS should pull all singular values toward 1.

    Uses rectangular matrices (well-conditioned: Marchenko-Pastur keeps singular
    values away from 0); a square Gaussian can be near-singular and 5 NS steps
    cannot lift a ~0 singular value to 1.
    """
    torch.manual_seed(1)
    G = torch.randn(*shape, dtype=torch.float64)
    X = muon.gram_newton_schulz(G)
    sv = torch.linalg.svdvals(X)
    assert sv.max() < 1.4 and sv.min() > 0.6, (sv.min().item(), sv.max().item())


def test_muon_rejects_1d_params():
    """Muon must refuse 1D tensors (they belong to AdamW)."""
    with pytest.raises(ValueError):
        muon.Muon([torch.zeros(10, requires_grad=True)])


def test_param_split_routes_1d_and_embeddings_to_adamw():
    model = Qwen3ForCausalLM(tiny())
    muon_p, adamw_p = split_params(model)
    # every muon param is a 2D non-embedding matrix
    assert all(p.ndim == 2 for p in muon_p)
    # the (tied) embedding weight is in adamw, never muon
    embed = model.embed_tokens.weight
    assert any(p is embed for p in adamw_p)
    assert all(p is not embed for p in muon_p)
    # 1D norm weights go to adamw
    assert any(p.ndim == 1 for p in adamw_p)


def test_muon_step_updates_2d_params():
    torch.manual_seed(0)
    w = torch.randn(16, 8, requires_grad=True)
    opt = muon.Muon([w], lr=0.1)
    before = w.detach().clone()
    (w.sum()).backward()
    opt.step()
    assert not torch.allclose(before, w.detach())
