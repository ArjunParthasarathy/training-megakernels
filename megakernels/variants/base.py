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

    # ---- lifecycle ----
    def setup(self) -> str:
        """Activate this variant's kernel backend. Returns the resolved backend."""
        backend = kernels.set_backend(self.kernel_backend)
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
