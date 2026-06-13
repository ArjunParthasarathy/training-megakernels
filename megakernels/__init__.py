"""training-megakernels: a Qwen3-1.7B training loop with swappable CuTeDSL kernels.

Three comparable training runs (see megakernels.variants):
  baseline         stock Qwen3-1.7B + AdamW + eager kernels (the reference bar)
  modded           Muon (Gram Newton-Schulz) + CuTeDSL kernels + fused CE
  custom_backward  modded + a hand-written, more-efficient backward pass (= 'dev')
"""

from __future__ import annotations

__all__ = ["config", "model", "kernels", "optim", "variants", "data"]
