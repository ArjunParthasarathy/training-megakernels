"""Model architecture checks (CPU, tiny config; full config via meta device)."""

import torch

from megakernels.config import tiny, qwen3_1p7b
from megakernels.model import Qwen3ForCausalLM


def test_forward_shapes_and_finite_loss():
    cfg = tiny()
    model = Qwen3ForCausalLM(cfg).to(torch.float64)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    out = model(ids, labels=ids, fused_ce=False)
    assert torch.isfinite(out["loss"])
    logits = model(ids)["logits"]
    assert logits.shape == (2, 16, cfg.vocab_size)


def test_fused_ce_matches_explicit_ce():
    cfg = tiny()
    model = Qwen3ForCausalLM(cfg).to(torch.float64)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    a = model(ids, labels=ids, fused_ce=False)["loss"]
    b = model(ids, labels=ids, fused_ce=True)["loss"]
    assert torch.allclose(a, b, atol=1e-8), (a.item(), b.item())


def test_tied_embeddings():
    model = Qwen3ForCausalLM(tiny())
    assert model.lm_head is None
    assert model.head_weight is model.embed_tokens.weight


def test_backward_runs():
    cfg = tiny()
    model = Qwen3ForCausalLM(cfg).to(torch.float64)
    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    model(ids, labels=ids)["loss"].backward()
    assert model.embed_tokens.weight.grad is not None


def test_full_config_param_count():
    """Build Qwen3-1.7B on meta (no memory) and sanity-check the param count."""
    with torch.device("meta"):
        model = Qwen3ForCausalLM(qwen3_1p7b())
    n = model.num_params()
    assert 1.5e9 < n < 2.2e9, f"{n/1e9:.3f}B"
