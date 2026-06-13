"""Minimal, self-contained Qwen3-1.7B training step for profiling.

This is a *profiling target*, not a real training run: it builds a randomly
initialized Qwen3-1.7B from config (no weight download) and feeds synthetic
token batches, so you can validate the entire nsys/ncu + Vast.ai pipeline before
any custom CuTeDSL kernels exist. Swap ``build_model_and_data`` for your real
nanochat / torchtitan loop later; the ``profiled_loop`` wrapper stays the same.

Run directly for a smoke test, or under ncu/nsys (see profiling/run_ncu.sh,
profiling/run_nsys.sh). Honors PROFILE_WARMUP / PROFILE_STEPS env vars.

Qwen3-1.7B config (from huggingface.co/Qwen/Qwen3-1.7B/config.json):
  hidden=2048, layers=28, q_heads=16, kv_heads=8 (GQA), head_dim=128,
  intermediate=6144 (SwiGLU), vocab=151936, RMSNorm eps=1e-6, RoPE theta=1e6,
  QK-norm (per-head RMSNorm over head_dim), tied embeddings, bf16.
"""

from __future__ import annotations

import argparse

import torch

from profile_utils import nvtx_range, profiled_loop


def build_model_and_data(seq_len: int, batch: int, device: str, full_size: bool):
    """Return (model, optimizer, data_iter). Random init, synthetic data."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    if full_size:
        cfg = Qwen3Config(
            hidden_size=2048,
            intermediate_size=6144,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            head_dim=128,
            vocab_size=151936,
            max_position_embeddings=max(seq_len, 4096),
            rms_norm_eps=1e-6,
            rope_theta=1_000_000,
            tie_word_embeddings=True,
            attn_implementation="flash_attention_2",
        )
    else:
        # Tiny config: fast smoke test of the pipeline on any GPU.
        cfg = Qwen3Config(
            hidden_size=512,
            intermediate_size=1536,
            num_hidden_layers=4,
            num_attention_heads=8,
            num_key_value_heads=4,
            head_dim=64,
            vocab_size=4096,
            max_position_embeddings=max(seq_len, 4096),
            tie_word_embeddings=True,
        )

    model = Qwen3ForCausalLM(cfg).to(device=device, dtype=torch.bfloat16)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

    vocab = cfg.vocab_size

    def data_iter():
        g = torch.Generator(device=device).manual_seed(0)
        while True:
            ids = torch.randint(0, vocab, (batch, seq_len), device=device, generator=g)
            yield ids

    return model, opt, data_iter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--compile", action="store_true", help="torch.compile the model")
    p.add_argument("--tiny", action="store_true", help="tiny config for CPU/small-GPU smoke test")
    p.add_argument("--max-steps", type=int, default=64)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, opt, data = build_model_and_data(args.seq_len, args.batch, device, full_size=not args.tiny)
    if args.compile:
        model = torch.compile(model)

    def steps():
        for i in range(args.max_steps):
            yield next(data)

    for ids in profiled_loop(steps()):
        with nvtx_range("forward"):
            out = model(input_ids=ids, labels=ids)
            loss = out.loss
        with nvtx_range("backward"):
            loss.backward()
        with nvtx_range("optimizer"):
            opt.step()
            opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"loss={loss.item():.4f}", flush=True)


if __name__ == "__main__":
    main()
