"""Modded Qwen3-1.7B: speedrun-style training + CuTeDSL kernels.

Same architecture as baseline, but: Muon (Gram Newton-Schulz) over 2D hidden
matrices + AdamW over the rest; CuTeDSL kernel backend (auto -> cute on a GPU,
eager fallback on CPU); fused linear-cross-entropy. **This variant is itself a
baseline** for the custom-backward run.

It also wraps the model in ``torch.compile(mode="reduce-overhead")`` so the cute
kernels are replayed via CUDAGraphs — cute alone still launches each kernel host-
side, leaving the ~1.2us inter-kernel bubbles a plain nsys timeline shows; graphing
collapses that launch overhead on top of cute's per-kernel speedups. (Unlike the
``cudagraph`` variant, which graphs the *eager* kernels to isolate the launch-
overhead win, this stacks graphing on the cute backend.)
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
        compile_mode="reduce-overhead",   # CUDAGraph replay over the cute kernels
    )
