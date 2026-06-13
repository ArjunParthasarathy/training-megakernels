"""Reuse mature CuTeDSL kernels instead of rewriting them (GPU only).

Per docs/PLAN.md, attention / RMSNorm / cross-entropy already have production
CuTeDSL implementations. This module wires them into the dispatch registry when
their packages are installed; anything missing simply stays on the eager fallback.

  attention            -> flash-attn-4  (flash_attn.cute, GQA, fwd+bwd)
  rms_norm             -> Dao-AILab/quack
  linear_cross_entropy -> quack cross-entropy (fused, no [N,vocab] materialization)
"""

from __future__ import annotations

from . import register


def _wire_flash_attn():
    from flash_attn.cute import flash_attn_func   # type: ignore

    def attention(q, k, v, causal: bool = True):
        # flash-attn expects [B, T, H, D]; model passes [B, H, T, D].
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        o = flash_attn_func(q, k, v, causal=causal)
        return o.transpose(1, 2)

    register("attention", attention)


def _wire_quack():
    import quack  # type: ignore  # noqa: F401
    from quack import rmsnorm as _rms, cross_entropy as _ce  # type: ignore

    register("rms_norm", lambda x, weight, eps: _rms(x, weight, eps))

    def linear_cross_entropy(hidden, weight, targets, ignore_index: int = -100):
        return _ce(hidden, weight, targets, ignore_index=ignore_index)

    register("linear_cross_entropy", linear_cross_entropy)


for _wire in (_wire_flash_attn, _wire_quack):
    try:  # pragma: no cover - GPU only
        _wire()
    except Exception:  # noqa: BLE001 - package not installed -> eager fallback
        pass
