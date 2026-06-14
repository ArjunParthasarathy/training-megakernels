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
    print(f">> validating modded cute custom_op gradients on {torch.cuda.get_device_name()}")
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

    print()
    if all_ok:
        print(">> GRADIENT MATCH: PASS — modded cute custom_ops match eager fwd+bwd")
        return 0
    print(">> GRADIENT MATCH: FAIL — see the FAIL rows above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
