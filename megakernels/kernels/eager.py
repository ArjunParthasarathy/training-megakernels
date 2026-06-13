"""Eager PyTorch reference implementations of every fusible op.

These are the *correctness oracle*: the CuTeDSL kernels in ``megakernels.cute``
must match these to within bf16 tolerances (see tests/). They are plain,
unfused, and run anywhere (CPU included), so the whole training stack is testable
without a GPU.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """Accumulation dtype: upcast low precision to fp32, otherwise keep (so fp64
    inputs stay fp64 for exact numerical tests)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Standard RMSNorm over the last dim. (P2 reference.)"""
    dtype = x.dtype
    xf = x.to(acc_dtype(dtype))
    var = xf.pow(2).mean(-1, keepdim=True)
    out = xf * torch.rsqrt(var + eps)
    return (out.to(dtype)) * weight


def qk_norm(q: Tensor, k: Tensor, q_w: Tensor, k_w: Tensor, eps: float):
    """Per-head RMSNorm over head_dim, applied before RoPE. (Qwen3-specific, P3.)

    q: [B, H, T, D], k: [B, Hkv, T, D]; q_w/k_w: [D].
    """
    return rms_norm(q, q_w, eps), rms_norm(k, k_w, eps)


def rotary_cos_sin(positions: Tensor, head_dim: int, theta: float, dtype, device):
    """Precompute RoPE cos/sin for given positions. positions: [T] -> [T, D]."""
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device).float() / half))
    freqs = torch.outer(positions.float(), inv_freq)          # [T, half]
    emb = torch.cat([freqs, freqs], dim=-1)                   # [T, D]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _rotate_half(x: Tensor) -> Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    """Apply rotary embedding. q/k: [B, H, T, D]; cos/sin: [T, D]. (P3 reference.)"""
    cos = cos[None, None]
    sin = sin[None, None]
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out.to(q.dtype), k_out.to(k.dtype)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """SwiGLU activation: silu(gate) * up. (P5 reference.)"""
    return F.silu(gate) * up


def attention(q: Tensor, k: Tensor, v: Tensor, causal: bool = True) -> Tensor:
    """GQA scaled-dot-product attention. q:[B,H,T,D] k/v:[B,Hkv,T,D].

    On the cute backend this routes to FlashAttention-4; here we use PyTorch SDPA,
    which already dispatches to a fused (FlashAttention-2/mem-efficient) kernel.
    GQA handled via enable_gqa.
    """
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)


def linear_cross_entropy(hidden: Tensor, weight: Tensor, targets: Tensor,
                         ignore_index: int = -100) -> Tensor:
    """Fused unembedding + cross-entropy. (P1 reference.)

    hidden: [N, hidden], weight: [vocab, hidden] (tied embedding), targets: [N].
    The eager reference materializes the [N, vocab] logits; the cute kernel avoids
    that HBM traffic (cut-cross-entropy style) but must match this loss.
    """
    acc = acc_dtype(hidden.dtype)
    logits = F.linear(hidden.to(acc), weight.to(acc))          # [N, vocab]
    return F.cross_entropy(logits, targets, ignore_index=ignore_index)
