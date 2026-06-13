"""P3 — fused QK-norm + RoPE prologue (GPU only, skeleton).

Qwen3-specific: per-head RMSNorm(head_dim) on Q and K, then RoPE, in one pass over
the QKV-projection output (no intermediate HBM writes). Not in stock kernel libs.

STATUS: skeleton — until the kernel lands this registers nothing, so qk_norm /
apply_rope use the eager reference. Implement a @cute.kernel that, per (head, tile)
of the QKV output, computes the head-dim RMS reduction and applies the rotary
rotation in registers, writing normed+rotated Q/K once.
"""

from __future__ import annotations

# TODO(gpu): implement fused qk_norm+rope cute kernel and
#   register("qk_norm", ...); register("apply_rope", ...)
