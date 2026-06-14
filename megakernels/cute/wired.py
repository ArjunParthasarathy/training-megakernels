"""Reuse mature CuTeDSL kernels instead of rewriting them (GPU only).

Per docs/PLAN.md, attention / RMSNorm / cross-entropy already have production
CuTeDSL implementations. This module wires them into the dispatch registry when
their packages are installed; anything missing simply stays on the eager fallback.

  attention            -> flash-attn-4 (flash_attn.cute), wrapped as a torch
                          custom_op returning (out, lse) + register_autograd over
                          flash's _flash_attn_bwd. GPU-UNVALIDATED (CPU can't run
                          flash_attn.cute) — see the !!! note in _wire_flash_attn.
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
    import torch
    from flash_attn.cute import flash_attn_func          # type: ignore
    # FA4's per-arch backward (FlashAttentionBackwardSm90 on Hopper) is reachable via
    # the internal _flash_attn_bwd in flash_attn.cute.interface. If this import path
    # changes, _wire_flash_attn raises and attention safely falls back to eager (the
    # caller's try/except), rather than running with a wrong/missing backward.
    from flash_attn.cute.interface import _flash_attn_bwd  # type: ignore

    # !!! GPU-UNVALIDATED (written, not yet run on an H100). flash_attn.cute is GPU-only,
    # so the pieces below cannot be exercised on CPU. Before relying on this, validate on
    # an H100 that (1) flash_attn_func(..., return_lse=True) returns (out, lse) in this
    # order, (2) the LSE shape assumed in register_fake matches, (3) _flash_attn_bwd's
    # signature/return match, and (4) grads equal those from the plain differentiable
    # flash_attn_func. See module docstring.

    def _to_flash(t):   # model [B, H, T, D] -> flash [B, T, H, D]
        return t.transpose(1, 2).contiguous()

    def _to_model(t):   # flash [B, T, H, D] -> model [B, H, T, D]
        return t.transpose(1, 2).contiguous()

    @torch.library.custom_op("megakernels::cute_attention", mutates_args=())
    def cute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # q:[B,H,T,D] k/v:[B,Hkv,T,D] (GQA handled by flash). return_lse=True so the
        # backward has the softmax LSE. softmax_scale left at flash's default (1/sqrt(D)
        # — matches the eager SDPA reference); the backward also uses the default so the
        # two stay consistent.
        res = flash_attn_func(_to_flash(q), _to_flash(k), _to_flash(v),
                              causal=causal, return_lse=True)
        o, lse = res[0], res[1]
        return _to_model(o), lse

    @cute_attention.register_fake
    def _(q, k, v, causal):
        # out takes q's [B,H,T,D] shape/dtype (GQA output has the query head count).
        # ASSUMED LSE shape [B, H, T] (fp32) — FA's per-(batch,head,query) log-sum-exp;
        # verify on GPU.
        B, H, T, _D = q.shape
        return torch.empty_like(q), q.new_empty((B, H, T), dtype=torch.float32)

    def _setup_context(ctx, inputs, output):
        q, k, v, causal = inputs
        out, lse = output
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = causal

    def _backward(ctx, grad_out, grad_lse):
        # grad_lse is ignored: lse is not consumed downstream (the dispatch wrapper
        # drops it), so only grad_out flows. Call flash's own sm90 backward in flash
        # layout, then map grads back to model layout.
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = _flash_attn_bwd(
            _to_flash(q), _to_flash(k), _to_flash(v),
            _to_flash(out), _to_flash(grad_out), lse,
            softmax_scale=None, causal=ctx.causal,
        )
        return _to_model(dq), _to_model(dk), _to_model(dv), None  # None: causal (bool)

    cute_attention.register_autograd(_backward, setup_context=_setup_context)

    def attention(q, k, v, causal: bool = True):
        # dispatch entry: return only the output tensor (drop lse) so the model sees the
        # same [B,H,T,D] tensor as before; autograd still flows through the custom op.
        out, _lse = cute_attention(q, k, v, causal)
        return out

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
