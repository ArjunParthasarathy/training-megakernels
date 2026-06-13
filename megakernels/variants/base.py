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
            weight_decay=run.weight_decay, ns_impl=self.ns_impl)

    # ---- one optimization step (shared default) ----
    def training_step(self, model: nn.Module, optimizers, batch: torch.Tensor) -> dict:
        out = model(batch, labels=batch, fused_ce=self.fused_ce)
        loss = out["loss"]
        loss.backward()
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        return {"loss": float(loss.detach())}
