"""CUDAGraph variant: CuTeDSL drop-in kernels, eager-compatible signatures, graphed.

Same architecture and optimizer as ``baseline`` (plain AdamW, explicit-logits CE,
no Muon, no CE fusion), but two things change together:
  * ``kernel_backend="auto"`` -> the CuTeDSL kernels on a GPU (FlashAttention-4
    attention + quack RMSNorm, wired in ``cute/wired.py``), eager fallback on CPU.
    These are *drop-in*: the dispatch signatures are identical to eager, so the
    module graph is unchanged — only the kernels behind ``kernels.attention`` /
    ``kernels.rms_norm`` differ.
  * ``torch.compile(mode="reduce-overhead")`` replays the **backbone** via CUDAGraphs
    (backbone-only compile; the CE stays eager — see Qwen3ForCausalLM.compile_backbone.
    cudagraph uses explicit CE so this is just where the one compile boundary lives).

So this variant isolates "**CuTe drop-in kernels + graphing**" relative to the eager
``baseline`` — it is the bar that says how much the signature-compatible CuTe kernels
plus graph replay buy, *before* the modded variant layers on Muon + fused linear-CE.

NOTE: with the CuTe kernels now in this variant, **no variant isolates pure launch
overhead** any more (the old "graph the *eager* kernels" role). That was an
intentional re-tiering: ``baseline -> cudagraph (+CuTe +graphs) -> modded (+Muon
+fused-CE) -> dev (+custom bwd)``. We still avoid ``max-autotune`` here so Inductor
doesn't swap in different GEMM kernels and confound the comparison.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="cudagraph",
        use_muon=False,
        kernel_backend="auto",     # cute drop-in on GPU, eager on CPU
        fused_ce=False,            # CE fusion is modded's job; stay eager-compatible
        custom_backward=False,
        compile_mode="reduce-overhead",
    )
