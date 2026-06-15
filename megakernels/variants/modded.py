"""Modded Qwen3-1.7B: speedrun-style training with **maximum kernel fusion**.

Same architecture as baseline, but stacks every fusion we have:
  * Muon (Gram Newton-Schulz) over 2D hidden matrices + AdamW over the rest;
  * CuTeDSL kernel backend (auto -> cute on a GPU, eager fallback on CPU): FA4
    attention + quack RMSNorm;
  * **a genuine fused linear-cross-entropy** — ``fused_ce=True`` now routes through
    apple cut-cross-entropy (wired in ``cute/wired.py``), so the [N, vocab] logits
    never materialize in HBM. (Previously this fell back to the eager path that
    *did* materialize them; quack ships only CE-over-logits, not the fused matmul.)
**This variant is itself the baseline** for the custom-backward (``dev``) run.

It also compiles the **backbone** with ``torch.compile(mode="reduce-overhead")`` so
the cute kernels are replayed via CUDAGraphs — cute alone still launches each kernel
host-side, leaving the ~1.2us inter-kernel bubbles a plain nsys timeline shows;
graphing collapses that launch overhead on top of cute's per-kernel speedups. Only
the backbone is compiled, not the whole step: the fused linear-CE (cut-cross-entropy)
graph-breaks under torch.compile, so — following apple/ml-cross-entropy + torchtune —
the loss runs eager OUTSIDE the graph (see Qwen3ForCausalLM.compile_backbone).

The ``cudagraph`` variant is the rung below: the *same* CuTe drop-in kernels +
graphing but **without** Muon or CE fusion (eager-compatible signatures). So
``modded`` isolates exactly "Muon + fused linear-CE" on top of ``cudagraph``.
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
