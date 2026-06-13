"""Custom-backward Qwen3-1.7B: the modded variant + a hand-written backward pass.

Identical to modded (Muon + CuTeDSL + fused CE) but installs the custom-backward
ops from megakernels/custom_backward.py. This is the run where we try to make the
backward pass more efficient (recompute / fused grad epilogues / chunked logit
grad). Skeleton: a few ops are converted; the rest stay on autograd's default.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="custom_backward",
        use_muon=True,
        kernel_backend="auto",
        fused_ce=True,
        ns_impl="gram",
        custom_backward=True,
    )
