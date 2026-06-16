"""Variant wiring + end-to-end step on the tiny config (CPU)."""

import math

import pytest
import torch

from megakernels import kernels
from megakernels.config import RunConfig, tiny
from megakernels.train import train
from megakernels.variants import get_variant, list_variants


@pytest.mark.parametrize("variant", ["baseline", "cudagraph", "modded", "custom_backward"])
def test_variant_trains_a_few_steps(variant):
    run = RunConfig(variant=variant, seq_len=32, batch_size=2, max_steps=3,
                    warmup_steps=1, tiny=True, model=tiny())
    metrics = train(run, verbose=False)
    assert len(metrics) == 3
    assert all(math.isfinite(m["loss"]) for m in metrics)


def test_backend_falls_back_to_eager_on_cpu():
    assert kernels.resolve_backend("auto") == "eager"   # no CUDA here
    assert kernels.set_backend("auto") == "eager"


def test_dev_is_alias_for_custom_backward():
    assert get_variant("dev").name == "custom_backward"
    assert "dev" in list_variants()


def test_baseline_and_modded_optimizers():
    """Muon was retired: baseline = plain AdamW, modded = fused (graphed) AdamW."""
    base = get_variant("baseline")
    mod = get_variant("modded")
    assert base.use_muon is False and base.fused_optimizer is False
    assert mod.use_muon is False and mod.fused_ce is True
    assert mod.fused_optimizer is True


def test_cudagraph_is_cute_dropin_with_graphs():
    """cudagraph = CuTe drop-in kernels + CUDAGraphs, but eager-compatible:
    AdamW (no Muon) and no CE fusion. It's the rung below modded."""
    cg = get_variant("cudagraph")
    assert cg.use_muon is False         # stays on AdamW like baseline
    assert cg.fused_ce is False         # CE fusion is modded's job
    assert cg.kernel_backend == "auto"  # CuTe on GPU, eager fallback on CPU
    assert cg.compile_mode == "reduce-overhead"
    assert cg.custom_backward is False


def test_modded_is_max_fusion():
    """modded = max fusion: cute backend + fused linear-CE + fused graphed AdamW + graphs."""
    mod = get_variant("modded")
    assert mod.use_muon is False
    assert mod.fused_optimizer is True
    assert mod.fused_ce is True
    assert mod.kernel_backend == "auto"
    assert mod.compile_mode == "reduce-overhead"
