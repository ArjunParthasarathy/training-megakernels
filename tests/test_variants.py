"""Variant wiring + end-to-end step on the tiny config (CPU)."""

import math

import pytest
import torch

from megakernels import kernels
from megakernels.config import RunConfig, tiny
from megakernels.train import train
from megakernels.variants import get_variant, list_variants


@pytest.mark.parametrize("variant", ["baseline", "modded", "custom_backward"])
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


def test_baseline_uses_adamw_modded_uses_muon():
    base = get_variant("baseline")
    mod = get_variant("modded")
    assert base.use_muon is False
    assert mod.use_muon is True and mod.fused_ce is True
