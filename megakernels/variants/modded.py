"""Modded Qwen3-1.7B: speedrun-style training + CuTeDSL kernels.

Same architecture as baseline, but: Muon (Gram Newton-Schulz) over 2D hidden
matrices + AdamW over the rest; CuTeDSL kernel backend (auto -> cute on a GPU,
eager fallback on CPU); fused linear-cross-entropy. **This variant is itself a
baseline** for the custom-backward run.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="modded",
        use_muon=True,
        kernel_backend="auto",     # cute on GPU, eager on CPU
        fused_ce=True,
        ns_impl="gram",
        custom_backward=False,
    )
