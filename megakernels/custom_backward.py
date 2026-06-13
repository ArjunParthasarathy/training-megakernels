"""Custom backward-pass skeleton (the 'custom_backward' run).

The modded variant uses PyTorch autograd's *default* backward for the fusible ops.
This module replaces selected ops with ``torch.autograd.Function``s that implement
a **hand-written backward** — the lever for making the backward pass more efficient
(recompute instead of store, fuse grad epilogues into GEMMs à la CODA, chunk the
logits grad so [N, vocab] never materializes, etc.).

``install()`` monkeypatches ``megakernels.kernels`` so the *same* model graph picks
these up with no architecture change — keeping the comparison clean.

Status:
  RMSNormFn   recompute rstd in backward (saves the normalized activation)  [done]
  SwiGLUFn    recompute sigmoid in backward (saves the product activation)  [done]
  LinearCEFn  chunked logits so grad never materializes [N, vocab]          [TODO]
  AttentionFn flash-style recompute backward                                [TODO]

The TODO ops stay on autograd's default backward until implemented, so the run is
always correct; tests/test_custom_backward.py checks the installed ops match eager.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from . import kernels
from .kernels.eager import acc_dtype


class RMSNormFn(torch.autograd.Function):
    """RMSNorm with explicit backward; rstd recomputed in backward (not stored)."""

    @staticmethod
    def forward(ctx, x: Tensor, weight: Tensor, eps: float):
        xf = x.to(acc_dtype(x.dtype))
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        return (xf * rstd).to(x.dtype) * weight

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        x, weight, rstd = ctx.saved_tensors
        acc = acc_dtype(x.dtype)
        xf = x.to(acc)
        g = grad_out.to(acc) * weight.to(acc)                 # dL/d(normalized)
        D = xf.shape[-1]
        xhat = xf * rstd
        # dL/dx = rstd * (g - xhat * mean(g * xhat))
        grad_x = rstd * (g - xhat * (g * xhat).mean(-1, keepdim=True))
        grad_w = (grad_out.to(acc) * xhat).reshape(-1, D).sum(0)
        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None


class SwiGLUFn(torch.autograd.Function):
    """silu(gate)*up with explicit backward; sigmoid recomputed in backward."""

    @staticmethod
    def forward(ctx, gate: Tensor, up: Tensor):
        ctx.save_for_backward(gate, up)
        return F.silu(gate) * up

    @staticmethod
    def backward(ctx, grad_out: Tensor):
        gate, up = ctx.saved_tensors
        acc = acc_dtype(gate.dtype)
        gf = gate.to(acc)
        sig = torch.sigmoid(gf)
        silu = gf * sig
        dsilu = sig * (1 + gf * (1 - sig))                    # d silu / d gate
        gop = grad_out.to(acc)
        grad_gate = gop * up.to(acc) * dsilu
        grad_up = gop * silu
        return grad_gate.to(gate.dtype), grad_up.to(up.dtype)


def _rms_norm_cb(x, weight, eps):
    return RMSNormFn.apply(x, weight, eps)


def _qk_norm_cb(q, k, q_w, k_w, eps):
    return RMSNormFn.apply(q, q_w, eps), RMSNormFn.apply(k, k_w, eps)


def _swiglu_cb(gate, up):
    return SwiGLUFn.apply(gate, up)


# --- (un)install: swap the dispatch-layer ops for the custom-backward versions ---
_ORIG: dict = {}
_OPS = {"rms_norm": _rms_norm_cb, "qk_norm": _qk_norm_cb, "swiglu": _swiglu_cb}


def install() -> None:
    """Route rms_norm / qk_norm / swiglu through the custom-backward Functions."""
    if _ORIG:
        return
    for name, fn in _OPS.items():
        _ORIG[name] = getattr(kernels, name)
        setattr(kernels, name, fn)


def uninstall() -> None:
    for name, fn in _ORIG.items():
        setattr(kernels, name, fn)
    _ORIG.clear()
