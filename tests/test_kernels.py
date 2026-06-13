"""Eager kernel reference ops vs independent math (CPU)."""

import torch
import torch.nn.functional as F

from megakernels.kernels import eager


def test_rms_norm():
    x = torch.randn(4, 8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    out = eager.rms_norm(x, w, 1e-6)
    ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * w
    assert torch.allclose(out, ref, atol=1e-10)


def test_swiglu():
    g = torch.randn(4, 8, dtype=torch.float64)
    u = torch.randn(4, 8, dtype=torch.float64)
    assert torch.allclose(eager.swiglu(g, u), F.silu(g) * u, atol=1e-12)


def test_rope_preserves_norm():
    """Rotary embedding is a rotation -> per-position vector norm is preserved."""
    B, H, T, D = 2, 2, 5, 16
    q = torch.randn(B, H, T, D, dtype=torch.float64)
    k = torch.randn(B, H, T, D, dtype=torch.float64)
    cos, sin = eager.rotary_cos_sin(torch.arange(T), D, 1e6, torch.float64, q.device)
    q2, k2 = eager.apply_rope(q, k, cos, sin)
    assert torch.allclose(q2.norm(dim=-1), q.norm(dim=-1), atol=1e-10)
    assert torch.allclose(k2.norm(dim=-1), k.norm(dim=-1), atol=1e-10)


def test_linear_cross_entropy_matches_explicit():
    N, hid, vocab = 7, 16, 32
    h = torch.randn(N, hid, dtype=torch.float64)
    w = torch.randn(vocab, hid, dtype=torch.float64)
    tgt = torch.randint(0, vocab, (N,))
    fused = eager.linear_cross_entropy(h, w, tgt)
    ref = F.cross_entropy(F.linear(h, w), tgt)
    assert torch.allclose(fused, ref, atol=1e-10)


def test_qk_norm_per_head():
    q = torch.randn(2, 3, 4, 8, dtype=torch.float64)   # B,H,T,D
    k = torch.randn(2, 2, 4, 8, dtype=torch.float64)
    qw = torch.randn(8, dtype=torch.float64)
    kw = torch.randn(8, dtype=torch.float64)
    qn, kn = eager.qk_norm(q, k, qw, kw, 1e-6)
    assert qn.shape == q.shape and kn.shape == k.shape
    # normalization is over the last dim (head_dim)
    ref_q = eager.rms_norm(q, qw, 1e-6)
    assert torch.allclose(qn, ref_q, atol=1e-12)
