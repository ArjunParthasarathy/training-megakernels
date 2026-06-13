"""Reuse mature CuTeDSL kernels instead of rewriting them (GPU only).

Per docs/PLAN.md, attention / RMSNorm / cross-entropy already have production
CuTeDSL implementations. This module wires them into the dispatch registry when
their packages are installed; anything missing simply stays on the eager fallback.

  attention            -> flash-attn-4  (flash_attn.cute, GQA, fwd+bwd)
  rms_norm             -> Dao-AILab/quack
  linear_cross_entropy -> stays EAGER: quack ships only cross_entropy over logits,
                          not a fused linear-CE that absorbs the unembed matmul.
"""

from __future__ import annotations

from . import register


def _wire_flash_attn():
    from flash_attn.cute import flash_attn_func   # type: ignore

    def attention(q, k, v, causal: bool = True):
        # flash-attn expects [B, T, H, D]; model passes [B, H, T, D].
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = flash_attn_func(q, k, v, causal=causal)
        # flash-attn-4 (4.0.0bN) returns a tuple (out, lse, ...) even when
        # return_lse is False; take the output tensor.
        if isinstance(o, tuple):
            o = o[0]
        return o.transpose(1, 2)

    register("attention", attention)


def _wire_quack():
    from quack import rmsnorm as _rms  # type: ignore

    # quack.rmsnorm is (x, weight, bias, residual, ..., eps=1e-6, ...): eps is a
    # KEYWORD, not the 3rd positional (that slot is `bias`). Passing eps positionally
    # makes quack run bias.dim() on a float -> AttributeError. Pass eps by name.
    register("rms_norm", lambda x, weight, eps: _rms(x, weight, eps=eps))

    # linear_cross_entropy stays on the eager reference: quack only has cross_entropy
    # over precomputed logits, which cannot fuse the unembed matmul, so there is no
    # quack op to wire here (see module docstring).


for _wire in (_wire_flash_attn, _wire_quack):
    try:  # pragma: no cover - GPU only
        _wire()
    except Exception:  # noqa: BLE001 - package not installed -> eager fallback
        pass
