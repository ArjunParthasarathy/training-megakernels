"""Custom-backward autograd.Functions: gradients must match autograd (CPU)."""

import torch
from torch.autograd import gradcheck

from megakernels import custom_backward as cb
from megakernels import kernels
from megakernels.kernels import eager


def test_rmsnorm_fn_gradcheck():
    x = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda x, w: cb.RMSNormFn.apply(x, w, 1e-6), (x, w))


def test_swiglu_fn_gradcheck():
    g = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    u = torch.randn(3, 8, dtype=torch.float64, requires_grad=True)
    assert gradcheck(lambda g, u: cb.SwiGLUFn.apply(g, u), (g, u))


def test_custom_backward_forward_matches_eager():
    x = torch.randn(3, 8, dtype=torch.float64)
    w = torch.randn(8, dtype=torch.float64)
    ref = eager.rms_norm(x, w, 1e-6)
    out = cb.RMSNormFn.apply(x, w, 1e-6)
    assert torch.allclose(ref, out, atol=1e-12)


def test_install_swaps_dispatch_ops():
    cb.uninstall()
    orig = kernels.rms_norm
    cb.install()
    try:
        assert kernels.rms_norm is not orig            # now custom-backward backed
        x = torch.randn(3, 8, dtype=torch.float64)
        w = torch.randn(8, dtype=torch.float64)
        assert torch.allclose(kernels.rms_norm(x, w, 1e-6),
                              eager.rms_norm(x, w, 1e-6), atol=1e-12)
    finally:
        cb.uninstall()
    assert kernels.rms_norm is orig
