"""Validate the modded cute custom_ops' GRADIENTS against the eager reference (GPU only).

The modded variant routes rms_norm -> quack and attention -> flash-attn-4, each wired
in megakernels/cute/wired.py as a torch.library.custom_op with a *hand-registered*
backward (register_autograd). Those backwards were committed GPU-UNVALIDATED (commits
ffc370f, 0caa106): flash_attn_func return order, the assumed LSE shape, _flash_attn_bwd's
signature, the layout transposes, and the softmax scale all needed checking on a real
H100. This script is that check.

For each op it runs the SAME inputs through the eager oracle (megakernels.kernels.eager,
the documented correctness reference) and through the cute custom_op, then compares both
the forward output and every input/weight gradient. A *correct* kernel differs from the
fp32-accumulated eager path only by bf16 rounding (rel-L2 ~1e-2, cosine ~1.0); a
*structurally wrong* backward (a bad transpose, a scale mismatch, a swapped tensor) shows
up as rel-L2 ~O(1) and cosine far from 1, which the thresholds below catch.

    python -m profiling.verify_modded_grads          # from /workspace/repo on the H100

Exits non-zero if any forward or gradient check fails, so it gates the profiling run.
"""

from __future__ import annotations

import sys

import torch

from megakernels import cute
from megakernels.kernels import eager
from megakernels import kernels


# --- comparison metrics ------------------------------------------------------
def _stats(name, a, b):
    """rel-L2 = ||a-b|| / ||b||, cosine = <a,b>/(|a||b|). Both in fp64 for a clean
    verdict independent of the tensors' own (bf16) precision."""
    a = a.detach().double().flatten()
    b = b.detach().double().flatten()
    diff = (a - b).norm().item()
    base = b.norm().item()
    rel_l2 = diff / (base + 1e-30)
    cos = torch.dot(a, b).item() / ((a.norm().item() * b.norm().item()) + 1e-30)
    max_abs = (a - b).abs().max().item()
    return {"name": name, "rel_l2": rel_l2, "cos": cos, "max_abs": max_abs}


def _report(stats, rel_tol, cos_tol):
    ok = stats["rel_l2"] <= rel_tol and stats["cos"] >= cos_tol
    flag = "PASS" if ok else "FAIL"
    print(f"    [{flag}] {stats['name']:<22} rel_l2={stats['rel_l2']:.2e} "
          f"cos={stats['cos']:.6f} max_abs={stats['max_abs']:.2e}"
          f"  (tol rel<={rel_tol:.0e} cos>={cos_tol})")
    return ok


def _dual(make_inputs, eager_fn, cute_fn):
    """Run eager_fn and cute_fn on independent leaf copies of the same inputs with the
    same upstream grad; return (eager_out, cute_out, [(name, eager_grad, cute_grad)...])."""
    ins_e = make_inputs()
    ins_c = make_inputs()  # fresh leaves so .grad doesn't collide / accumulate
    out_e = eager_fn(*[t for t, _ in ins_e])
    out_c = cute_fn(*[t for t, _ in ins_c])
    # identical upstream grad (seeded) so the backward comparison is apples-to-apples
    g = torch.randn_like(out_e.float()).to(out_e.dtype)
    out_e.backward(g)
    out_c.backward(g.to(out_c.dtype))
    grads = [(n, te.grad, tc.grad) for (te, n), (tc, _) in zip(ins_e, ins_c)]
    return out_e, out_c, grads


# --- the two ops -------------------------------------------------------------
def check_rms_norm(device, dtype, rel_tol, cos_tol):
    print(f"  rms_norm  (quack cute_rms_norm vs eager)  dtype={dtype}")
    torch.manual_seed(0)
    cute_rms = cute.OPS["rms_norm"]
    eps = 1e-6
    B, T, Dh = 2, 256, 2048   # hidden-size RMSNorm, the most frequent call

    def make():
        torch.manual_seed(0)  # same numbers for eager & cute leaf copies
        x = torch.randn(B, T, Dh, device=device, dtype=dtype, requires_grad=True)
        w = torch.randn(Dh, device=device, dtype=dtype, requires_grad=True)
        return [(x, "grad_x"), (w, "grad_w")]

    out_e, out_c, grads = _dual(make, lambda x, w: eager.rms_norm(x, w, eps),
                                lambda x, w: cute_rms(x, w, eps))
    ok = _report(_stats("forward", out_c, out_e), rel_tol, cos_tol)
    for n, ge, gc in grads:
        ok &= _report(_stats(n, gc, ge), rel_tol, cos_tol)
    return ok


def check_attention(device, dtype, rel_tol, cos_tol):
    print(f"  attention (flash-attn-4 cute_attention vs eager SDPA)  dtype={dtype}")
    torch.manual_seed(0)
    cute_attn = cute.OPS["attention"]
    B, H, Hkv, T, D = 2, 16, 8, 256, 128   # Qwen3-1.7B GQA 2:1, head_dim 128
    causal = True

    def make():
        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, Hkv, T, D, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, Hkv, T, D, device=device, dtype=dtype, requires_grad=True)
        return [(q, "grad_q"), (k, "grad_k"), (v, "grad_v")]

    out_e, out_c, grads = _dual(make, lambda q, k, v: eager.attention(q, k, v, causal),
                                lambda q, k, v: cute_attn(q, k, v, causal))
    ok = _report(_stats("forward", out_c, out_e), rel_tol, cos_tol)
    for n, ge, gc in grads:
        ok &= _report(_stats(n, gc, ge), rel_tol, cos_tol)
    return ok


def check_cce(device, dtype, rel_tol, cos_tol):
    """CCE fused linear-cross-entropy (apple cut-cross-entropy, wired in cute/wired.py)
    vs the eager oracle (F.cross_entropy(F.linear(h, w), tgt)). Checks loss + grad_hidden
    + grad_weight on a *realistic* unembed shape (N≈4096, hidden=2048, vocab=151936).

    We dispatch through the public kernels.linear_cross_entropy under the cute backend
    (exactly what model.py calls with fused_ce=True) so this tests the wired path, not
    `_cce` directly; the oracle is eager.linear_cross_entropy.
    """
    print(f"  linear_cross_entropy (CCE cute vs eager)  dtype={dtype}")
    N, hidden, vocab = 4096, 2048, 151936
    ignore_index = -100

    def make():
        torch.manual_seed(0)  # same numbers for eager & cute leaf copies
        h = torch.randn(N, hidden, device=device, dtype=dtype, requires_grad=True)
        # tied head [vocab, hidden]; scale down so logits/softmax are well-conditioned
        w = (torch.randn(vocab, hidden, device=device, dtype=dtype) * (hidden ** -0.5)
             ).detach().requires_grad_(True)
        return [(h, "grad_hidden"), (w, "grad_weight")]

    def make_tgt():
        torch.manual_seed(1)
        return torch.randint(0, vocab, (N,), device=device)

    tgt = make_tgt()
    # cute dispatch == what training uses (kernels.set_backend('cute') already active)
    cute_ce = lambda h, w: kernels.linear_cross_entropy(h, w, tgt, ignore_index=ignore_index)
    eager_ce = lambda h, w: eager.linear_cross_entropy(h, w, tgt, ignore_index=ignore_index)

    ins_e = make()
    ins_c = make()
    loss_e = eager_ce(*[t for t, _ in ins_e])
    loss_c = cute_ce(*[t for t, _ in ins_c])
    # scalar loss: backward with no upstream grad (== grad 1.0)
    loss_e.backward()
    loss_c.backward()
    grads = [(n, te.grad, tc.grad) for (te, n), (tc, _) in zip(ins_e, ins_c)]

    # loss is a scalar; compare directly (rel-L2 of a 1-vector == |rel err|)
    ok = _report(_stats("loss", loss_c.reshape(1), loss_e.reshape(1)), rel_tol, cos_tol)
    print(f"      loss  eager={loss_e.item():.6f}  cce={loss_c.item():.6f}")
    for n, ge, gc in grads:
        ok &= _report(_stats(n, gc, ge), rel_tol, cos_tol)
    return ok


def check_cce_peakmem(device, dtype):
    """Peak CUDA memory of the wired CCE path vs the explicit-logits path, measured with
    torch.cuda.max_memory_allocated() around a fwd+bwd on identical inputs.

    IMPORTANT trade-off (verified on H100 2026-06-15): CCE's memory-fused impl="cce" keeps
    the [N,vocab] logits off HBM (peak << the logits tensor) BUT computes a wrong
    grad_weight at vocab=151936 (see check_cce). We therefore wire impl="torch_compile",
    which is grad-CORRECT but, called eagerly (here), materializes the logits — so this
    EAGER peak is ~the explicit path, not below it. The HBM win only re-appears when
    inductor fuses the linear+CE epilogue inside the modded torch.compile graph; that is
    NOT what this standalone eager probe measures. So this check is a REPORT (peak of both
    paths + how each relates to the logits tensor); it passes as long as the CCE path does
    not blow well PAST the explicit path. The "logits never materialize" claim holds for
    impl="cce" (shown here for reference) but is sacrificed for grad correctness."""
    print(f"  linear_cross_entropy PEAK-MEM (CCE wired vs explicit logits)  dtype={dtype}")
    N, hidden, vocab = 4096, 2048, 151936
    logits_mb = N * vocab * 4 / 1e6  # fp32 [N, vocab] the explicit path materializes

    def run(fn):
        torch.manual_seed(0)
        h = torch.randn(N, hidden, device=device, dtype=dtype, requires_grad=True)
        w = (torch.randn(vocab, hidden, device=device, dtype=dtype) * (hidden ** -0.5)
             ).detach().requires_grad_(True)
        torch.manual_seed(1)
        tgt = torch.randint(0, vocab, (N,), device=device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.max_memory_allocated()
        loss = fn(h, w, tgt)
        loss.backward()
        torch.cuda.synchronize()
        return (torch.cuda.max_memory_allocated() - base) / 1e6  # MB for this op's fwd+bwd

    explicit_mb = run(lambda h, w, t: eager.linear_cross_entropy(h, w, t))
    wired_mb = run(lambda h, w, t: kernels.linear_cross_entropy(h, w, t))
    # impl="cce" reference (memory-fused but grad-wrong) so the trade-off is visible.
    cce_mb = None
    try:
        from cut_cross_entropy import linear_cross_entropy as _raw_cce
        cce_mb = run(lambda h, w, t: _raw_cce(h, w, t, shift=0, reduction="mean", impl="cce"))
    except Exception:
        pass
    print(f"      explicit (materializes [N,vocab]) peak +{explicit_mb:8.1f} MB "
          f"([N,vocab] fp32 logits alone = {logits_mb:.1f} MB)")
    print(f"      CCE wired (impl=torch_compile)    peak +{wired_mb:8.1f} MB  "
          f"(grad-correct; eager call materializes logits)")
    if cce_mb is not None:
        print(f"      CCE impl=cce (ref: grad-WRONG)    peak +{cce_mb:8.1f} MB  "
              f"(memory-fused: << logits, but wrong grad_weight)")
    # Soft bar: the grad-correct wired path must not blow PAST the explicit path. The true
    # HBM win is impl=cce's (shown above), which we trade away for a correct grad_weight.
    ok = wired_mb <= explicit_mb * 1.25
    print(f"      [{'PASS' if ok else 'FAIL'}] wired peak <= 1.25x explicit "
          f"(impl=cce would be << logits but is grad-wrong; see check_cce)")
    return ok


def main() -> int:
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — the cute kernels are GPU-only.")
        return 2
    if not cute.available():
        why = cute.why_unavailable() or "unknown"
        print(f"FAIL: cute backend unavailable — {why}\n"
              f"      Nothing to validate (eager==eager). Install the cute stack first.")
        return 2

    device = "cuda"
    # Route the public dispatch to cute so kernels.linear_cross_entropy hits the wired
    # CCE op (what model.py calls with fused_ce=True), not the eager fallback.
    backend = kernels.set_backend("cute")
    print(f">> validating modded cute custom_op gradients on {torch.cuda.get_device_name()}")
    print(f">> dispatch backend: {backend}")
    print(f">> cute OPS wired: {sorted(cute.OPS)}")

    all_ok = True
    # bf16 is the training dtype: thresholds separate bf16 rounding (rel_l2 ~1e-2,
    # cos ~1.0) from a structurally-wrong backward (rel_l2 ~O(1), cos far from 1).
    if "rms_norm" in cute.OPS:
        all_ok &= check_rms_norm(device, torch.bfloat16, rel_tol=3e-2, cos_tol=0.999)
        all_ok &= check_rms_norm(device, torch.float32, rel_tol=1e-4, cos_tol=0.99999)
    else:
        print("  rms_norm: NOT wired to cute (quack missing) — skipped"); all_ok = False
    if "attention" in cute.OPS:
        all_ok &= check_attention(device, torch.bfloat16, rel_tol=3e-2, cos_tol=0.999)
    else:
        print("  attention: NOT wired to cute (flash-attn-4 missing) — skipped"); all_ok = False
    # CCE fused linear-cross-entropy (the new modded fused_ce path). vocab=151936 means
    # a chunked-softmax kernel vs eager's fp32 logits: bf16 rounding only, so the same
    # thresholds as the other bf16 ops separate rounding from a structurally-wrong grad.
    if "linear_cross_entropy" in cute.OPS:
        all_ok &= check_cce(device, torch.bfloat16, rel_tol=3e-2, cos_tol=0.999)
        all_ok &= check_cce_peakmem(device, torch.bfloat16)
    else:
        print("  linear_cross_entropy: NOT wired to cute (cut-cross-entropy missing) — skipped")
        all_ok = False

    print()
    if all_ok:
        print(">> GRADIENT MATCH: PASS — modded cute custom_ops match eager fwd+bwd")
        return 0
    print(">> GRADIENT MATCH: FAIL — see the FAIL rows above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
