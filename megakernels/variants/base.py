"""Variant interface. A *variant* is one training configuration we want to compare:
it picks the optimizer, the kernel backend, and (optionally) a custom backward
pass. The model architecture is shared (megakernels/model.py), so comparing two
variants isolates exactly these axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .. import kernels
from ..config import ModelConfig, RunConfig
from ..model import Qwen3ForCausalLM
from ..optim import build_optimizers


@dataclass
class TrainVariant:
    name: str
    use_muon: bool          # Muon+AdamW split vs plain AdamW
    kernel_backend: str     # 'auto' | 'eager' | 'cute'
    fused_ce: bool          # fused linear-cross-entropy vs explicit logits
    ns_impl: str = "gram"   # newton-schulz variant for Muon
    custom_backward: bool = False
    compile_mode: str | None = None  # torch.compile mode; 'reduce-overhead' = CUDAGraphs. None = pure eager
    fused_optimizer: bool = False    # fused (capturable) AdamW; graphed when compile_mode='reduce-overhead'

    # ---- lifecycle ----
    def setup(self, *, require_cute: bool = False) -> str:
        """Activate this variant's kernel backend. Returns the resolved backend.

        If the variant asked for the cute backend ('auto'/'cute') but it resolved to
        'eager', print *why* (cute.why_unavailable) instead of silently degrading —
        the old behaviour produced an eager profile masquerading as a cute one. Pass
        require_cute=True to make that fall-back a hard error instead of a warning.
        """
        backend = kernels.set_backend(self.kernel_backend)
        if self.kernel_backend in ("auto", "cute") and backend != "cute":
            from .. import cute
            why = cute.why_unavailable() or "unknown reason"
            msg = (f"[{self.name}] requested kernel_backend={self.kernel_backend!r} "
                   f"but resolved to EAGER — {why}")
            if require_cute:
                raise RuntimeError(msg + " (--require-cute)")
            print("WARNING: " + msg, flush=True)
        if self.custom_backward:
            from .. import custom_backward as cb
            cb.install()
        return backend

    def build_model(self, model_cfg: ModelConfig) -> nn.Module:
        return Qwen3ForCausalLM(model_cfg)

    def build_optimizers(self, model: nn.Module, run: RunConfig):
        return build_optimizers(
            model, use_muon=self.use_muon, lr=run.lr, muon_lr=run.muon_lr,
            weight_decay=run.weight_decay, ns_impl=self.ns_impl,
            fused=self.fused_optimizer)

    # ---- one optimization step (shared default) ----
    def _graph_optimizer(self) -> bool:
        """Whether to capture opt.step() into a CUDA graph this run.

        Only when the variant opts into the fused optimizer AND the backbone is
        reduce-overhead (CUDAGraphs) AND we're on CUDA. Off on CPU / eager so tests
        and the baseline stay on the plain path. The compiled-optimizer recipe
        (torch.compile(reduce-overhead) over opt.step) CUDA-graphs the update; with the
        fused AdamW that is a single multi-tensor kernel, so the whole step graphs.
        """
        return (self.fused_optimizer and self.compile_mode == "reduce-overhead"
                and torch.cuda.is_available())

    def _optimizer_step(self, optimizers):
        """Run the optimizer step, graphed when _graph_optimizer(). Cached compiled fn.

        Under CUDA graphs the gradients must keep static addresses across replays, so
        zero_grad uses set_to_none=False (zero in place) instead of dropping buffers.
        """
        if self._graph_optimizer():
            step = getattr(self, "_compiled_opt_step", None)
            if step is None:
                def _raw_step():
                    for opt in optimizers:
                        opt.step()
                step = self._compiled_opt_step = torch.compile(
                    _raw_step, mode="reduce-overhead")
            step()
            for opt in optimizers:
                opt.zero_grad(set_to_none=False)
        else:
            for opt in optimizers:
                opt.step()
                opt.zero_grad(set_to_none=True)

    def training_step(self, model: nn.Module, optimizers, batch: torch.Tensor) -> dict:
        out = model(batch, labels=batch, fused_ce=self.fused_ce)
        loss = out["loss"]
        loss.backward()
        self._optimizer_step(optimizers)
        return {"loss": float(loss.detach())}
