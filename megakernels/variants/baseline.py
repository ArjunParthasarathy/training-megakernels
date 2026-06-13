"""Baseline Qwen3-1.7B: the reference training loop.

Stock architecture, plain AdamW over all params, eager kernels (SDPA attention,
explicit-logits cross-entropy). No Muon, no CuTeDSL, no fusion. This is the bar
the modded variant must beat on throughput at equal-or-better loss.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="baseline",
        use_muon=False,
        kernel_backend="eager",
        fused_ce=False,
        custom_backward=False,
    )
