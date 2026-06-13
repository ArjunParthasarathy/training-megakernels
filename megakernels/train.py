"""Unified training entrypoint for all variants.

    python -m megakernels.train --variant baseline   --tiny --max-steps 20
    python -m megakernels.train --variant modded      --tiny --max-steps 20
    python -m megakernels.train --variant custom_backward --tiny --max-steps 20

Same data/seed across variants (megakernels.data.synthetic_loader), so loss curves
are directly comparable. Emits per-step metrics (loss, step time, tokens/s, peak
mem) and supports NVTX/cudaProfilerApi capture so it runs under nsys/ncu (see
profiling/). Returns the metrics list so megakernels.compare can drive it in-proc.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from .config import ModelConfig, RunConfig, qwen3_1p7b, tiny
from .data import synthetic_loader
from .variants import get_variant, list_variants


def _device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    return "cpu", torch.float32


def train(run: RunConfig, *, profile: bool = False, verbose: bool = True,
          require_cute: bool = False):
    torch.manual_seed(run.seed)
    device, dtype = _device_dtype()

    variant = get_variant(run.variant)
    backend = variant.setup(require_cute=require_cute)

    model = variant.build_model(run.model).to(device=device, dtype=dtype)
    model.train()
    optimizers = variant.build_optimizers(model, run)
    loader = synthetic_loader(run.model.vocab_size, run.batch_size, run.seq_len,
                              device=device, seed=run.seed)

    # torch.compile mode. The variant may request one (e.g. the cudagraph variant
    # -> 'reduce-overhead', which replays the eager kernels via CUDAGraphs); the
    # --compile flag is a manual override using default mode. Variant wins if both set.
    compile_mode = variant.compile_mode or ("default" if run.compile else None)
    if compile_mode is not None:
        model = torch.compile(model, mode=compile_mode)

    if verbose:
        print(f"[{run.variant}] backend={backend} device={device} dtype={dtype} "
              f"params={model.num_params()/1e6:.1f}M "
              f"opt={'Muon+AdamW' if variant.use_muon else 'AdamW'} "
              f"fused_ce={variant.fused_ce} custom_bwd={variant.custom_backward} "
              f"compile={compile_mode or 'off'}",
              flush=True)

    metrics = []
    log_f = open(run.log_path, "w") if run.log_path else None
    capturing = False
    tokens_per_step = run.batch_size * run.seq_len

    for step in range(run.max_steps):
        if profile and device == "cuda" and step == run.warmup_steps:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()       # nsys --capture-range=cudaProfilerApi
            capturing = True

        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        # constant range name so `ncu --nvtx-include "profile_step/"` matches.
        torch.cuda.nvtx.range_push("profile_step") if device == "cuda" else None
        batch = next(loader)
        m = variant.training_step(model, optimizers, batch)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()

        dt = time.perf_counter() - t0
        m.update(step=step, step_ms=dt * 1e3, tokens_per_s=tokens_per_step / dt)
        if device == "cuda":
            m["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
        metrics.append(m)
        if log_f:
            log_f.write(json.dumps(m) + "\n")
        if verbose and (step < 3 or step % 10 == 0):
            print(f"  step {step:4d}  loss {m['loss']:.4f}  "
                  f"{m['step_ms']:.1f} ms  {m['tokens_per_s']:.0f} tok/s", flush=True)

        if capturing and step == run.warmup_steps + 2:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            break

    if log_f:
        log_f.close()
    return metrics


def _build_run(args) -> RunConfig:
    model_cfg: ModelConfig = tiny() if args.tiny else qwen3_1p7b()
    return RunConfig(
        variant=args.variant, kernels=args.kernels, seq_len=args.seq_len,
        batch_size=args.batch, max_steps=args.max_steps, warmup_steps=args.warmup,
        seed=args.seed, compile=args.compile, tiny=args.tiny,
        log_path=args.log, model=model_cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="baseline", choices=list_variants())
    p.add_argument("--kernels", default="auto", choices=["auto", "eager", "cute"])
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--profile", action="store_true", help="cudaProfilerApi capture for nsys/ncu")
    p.add_argument("--require-cute", action="store_true",
                   help="hard-fail if a cute/auto variant falls back to eager (no silent degrade)")
    p.add_argument("--log", default="")
    args = p.parse_args()
    run = _build_run(args)
    # kernel backend override (variant default unless explicitly set)
    if args.kernels != "auto":
        get_variant  # variants set their own backend; --kernels forces it below
    train(run, profile=args.profile, require_cute=args.require_cute)


if __name__ == "__main__":
    main()
