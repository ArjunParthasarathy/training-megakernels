"""CUDAGraph baseline: the eager model, but graphed.

Identical to ``baseline`` in every axis (plain AdamW, eager SDPA + explicit-logits
CE, no Muon/CuTeDSL/fusion) except it wraps the model in
``torch.compile(mode="reduce-overhead")``. That mode keeps the *same* kernels as
eager but replays them via CUDAGraphs, collapsing the per-launch host overhead
(the ~1.2us inter-kernel bubbles visible between the attention-backward kernels on
an eager nsys timeline).

This isolates exactly the launch-overhead win: it is the apples-to-apples
"how much does graphing eager buy us" bar. We deliberately do *not* use
``max-autotune`` here -- that adds Inductor's Triton-template autotuning, which
substitutes different GEMM kernels and so would confound the launch-overhead
effect with kernel selection.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="cudagraph",
        use_muon=False,
        kernel_backend="eager",
        fused_ce=False,
        custom_backward=False,
        compile_mode="reduce-overhead",
    )
