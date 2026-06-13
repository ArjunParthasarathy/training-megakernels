"""Reuse mature CuTeDSL kernels instead of rewriting them (GPU only).

Per docs/PLAN.md, attention / RMSNorm / cross-entropy already have production
CuTeDSL implementations. This module wires them into the dispatch registry when
their packages are installed; anything missing simply stays on the eager fallback.

  attention            -> flash-attn-4 (flash_attn.cute). Kept as flash's own
                          differentiable call (NOT a torch custom op) — see below.
  rms_norm             -> Dao-AILab/quack, wrapped as a torch.library.custom_op so
                          torch.compile/CUDAGraph (the `modded` reduce-overhead path)
                          treats it as one opaque-but-declared node instead of
                          graph-breaking on the CuTeDSL kernel.
  linear_cross_entropy -> stays EAGER: quack ships only cross_entropy over logits,
                          not a fused linear-CE that absorbs the unembed matmul.

Why custom_op + register_fake + register_autograd (rms_norm): under torch.compile,
Dynamo can't trace into an opaque CuTeDSL kernel, so a bare call graph-breaks and
fragments CUDAGraph capture. Declaring it a custom op with a fake (meta) kernel lets
Dynamo keep it in-graph (shape-propagated via the fake kernel, kernel body never
traced), and register_autograd puts our analytic backward into the AOTAutograd
backward graph so `reduce-overhead` can CUDAGraph the backward too. This is the
PyTorch-recommended way to integrate third-party/DSL kernels; there is no library
that auto-wraps CuTeDSL kernels, so it is done per-op here.
"""

from __future__ import annotations

from . import register


def _wire_flash_attn():
    from flash_attn.cute import flash_attn_func   # type: ignore

    # NOTE: attention is not (yet) wrapped as a torch custom op. flash_attn_func is
    # already differentiable through flash-attn's own autograd (FlashAttnFunc.apply ->
    # _flash_attn_bwd; the sm90 backward IS shipped in the flash-attn-4 wheel and runs
    # on our H100 — modded trains fine through it), so today it simply graph-breaks and
    # becomes a CUDAGraph partition boundary: it runs eager while the norm/MLP/GEMM
    # regions around it are captured. To also pull attention into the captured graph we
    # can wrap it as a custom_op returning (out, lse) + register_autograd that calls
    # flash's _flash_attn_bwd. That wiring (exact LSE shape for register_fake,
    # _flash_attn_bwd signature) can't be validated on CPU (flash_attn.cute is GPU-only),
    # so it is a deliberate follow-up to validate during the profiling run rather than a
    # blind commit — a wrong backward would silently corrupt grads.
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
    import torch
    from quack import rmsnorm as _rms  # type: ignore

    from ..kernels.eager import acc_dtype

    @torch.library.custom_op("megakernels::cute_rms_norm", mutates_args=())
    def cute_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        # quack.rmsnorm wants eps as a KEYWORD (its 3rd positional is `bias`); passing
        # eps positionally makes quack run bias.dim() on a float -> AttributeError.
        return _rms(x, weight, eps=eps)

    @cute_rms_norm.register_fake
    def _(x, weight, eps):
        # Meta/fake kernel: RMSNorm preserves x's shape and dtype. No compute (runs on
        # FakeTensors during tracing) — just lets Dynamo propagate shapes past the op.
        return torch.empty_like(x)

    def _setup_context(ctx, inputs, output):
        x, weight, eps = inputs
        ctx.save_for_backward(x, weight)
        ctx.eps = eps

    def _backward(ctx, grad_out):
        # Analytic RMSNorm backward, mirroring custom_backward.RMSNormFn (which
        # tests/test_custom_backward.py checks == eager): recompute rstd instead of
        # saving it. Pure torch, so it is itself CUDAGraph-capturable inside the
        # compiled backward graph — which is the whole point of capturing the backward.
        x, weight = ctx.saved_tensors
        acc = acc_dtype(x.dtype)
        xf = x.to(acc)
        rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + ctx.eps)
        xhat = xf * rstd
        g = grad_out.to(acc) * weight.to(acc)                 # dL/d(normalized)
        grad_x = rstd * (g - xhat * (g * xhat).mean(-1, keepdim=True))
        grad_w = (grad_out.to(acc) * xhat).reshape(-1, x.shape[-1]).sum(0)
        return grad_x.to(x.dtype), grad_w.to(weight.dtype), None  # None: eps (float)

    cute_rms_norm.register_autograd(_backward, setup_context=_setup_context)
    register("rms_norm", cute_rms_norm)


for _wire in (_wire_flash_attn, _wire_quack):
    try:  # pragma: no cover - GPU only
        _wire()
    except Exception:  # noqa: BLE001 - package not installed -> eager fallback
        pass
