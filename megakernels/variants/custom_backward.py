"""Custom-backward Qwen3-1.7B: the modded variant + a hand-written backward pass.

Identical to modded (fused graphed AdamW + CuTeDSL + fused CE) but installs the
custom-backward ops from megakernels/custom_backward.py. This is the run where we try
to make the backward pass more efficient (recompute / fused grad epilogues / chunked
logit grad). Skeleton: a few ops are converted; the rest stay on autograd's default.

Like modded, Muon is retired in favour of fused AdamW. dev stays eager (no
compile_mode), so it gets the fused (capturable) AdamW kernel but NOT the graphed
opt.step() — graphing is gated on reduce-overhead, which the custom-backward
experiment runs without.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="custom_backward",
        use_muon=False,            # Muon retired; fused AdamW (matches modded)
        kernel_backend="auto",
        fused_ce=True,
        custom_backward=True,
        fused_optimizer=True,
    )
