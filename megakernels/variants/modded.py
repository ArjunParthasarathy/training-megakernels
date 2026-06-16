"""Modded Qwen3-1.7B: speedrun-style training with **maximum kernel fusion**.

Same architecture as baseline, but stacks every fusion we have:
  * **fused, graphed AdamW** over all params (``fused_optimizer=True``): the eager
    Muon + Gram Newton-Schulz optimizer used to be ~58% of the step (~11.5k kernels),
    and there is no graph-able / widely-supported "FusedMuon" — native torch.optim.Muon
    is single-tensor and not CUDA-graph capturable. So we retired Muon here in favour of
    ``torch.optim.AdamW(fused=True, capturable=True)``, whose step is one multi-tensor
    kernel and gets captured into a CUDA graph alongside the backbone (see
    TrainVariant._optimizer_step). The Muon module is kept for the symmetric-GEMM work.
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
graphing collapses that launch overhead on top of cute's per-kernel speedups. The
backbone fwd/bwd and the AdamW step are each CUDA-graphed; the fused linear-CE
(cut-cross-entropy) graph-breaks under torch.compile, so — following
apple/ml-cross-entropy + torchtune — the loss runs eager OUTSIDE the graph (see
Qwen3ForCausalLM.compile_backbone). Net: the only eager regions are CCE and the
per-step iteration boundary.

The ``cudagraph`` variant is the rung below: the *same* CuTe drop-in kernels +
graphing but **without** CE fusion or the fused/graphed optimizer (plain eager AdamW,
eager-compatible signatures). So ``modded`` isolates exactly "fused linear-CE +
graphed fused optimizer" on top of ``cudagraph``.
"""

from __future__ import annotations

from .base import TrainVariant


def make() -> TrainVariant:
    return TrainVariant(
        name="modded",
        use_muon=False,            # Muon retired; fused graphed AdamW instead
        kernel_backend="auto",     # cute on GPU, eager on CPU
        fused_ce=True,
        custom_backward=False,
        compile_mode="reduce-overhead",   # CUDAGraph replay over the cute kernels
        fused_optimizer=True,             # AdamW(fused, capturable); opt.step() graphed
    )
