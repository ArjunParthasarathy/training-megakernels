"""Compare training runs head-to-head on identical data/seed.

    python -m megakernels.compare --variants baseline modded --tiny --max-steps 30

Runs each variant in-process with the same seed (so they see the same batches) and
prints a table of final loss, median step time, tokens/s, and peak memory — the
apples-to-apples comparison of baseline vs modded vs custom_backward/dev.
"""

from __future__ import annotations

import argparse
import statistics

from .config import RunConfig, qwen3_1p7b, tiny
from .train import train
from .variants import list_variants


def run_all(variants, *, tiny_cfg: bool, max_steps: int, seq_len: int,
            batch: int, seed: int):
    results = {}
    for v in variants:
        run = RunConfig(variant=v, seq_len=seq_len, batch_size=batch,
                        max_steps=max_steps, seed=seed, tiny=tiny_cfg,
                        model=tiny() if tiny_cfg else qwen3_1p7b())
        metrics = train(run, verbose=True)
        # skip step 0 (warmup/compile) for timing
        timed = metrics[1:] if len(metrics) > 1 else metrics
        results[v] = {
            "final_loss": metrics[-1]["loss"],
            "median_step_ms": statistics.median(m["step_ms"] for m in timed),
            "median_tok_s": statistics.median(m["tokens_per_s"] for m in timed),
            "peak_mem_mb": max((m.get("peak_mem_mb", 0) for m in metrics), default=0),
        }
    return results


def print_table(results: dict):
    cols = ["variant", "final_loss", "median_step_ms", "median_tok_s", "peak_mem_mb"]
    print("\n" + "  ".join(f"{c:>15}" for c in cols))
    print("  ".join("-" * 15 for _ in cols))
    base = next(iter(results.values()))["median_step_ms"]
    for v, r in results.items():
        speedup = base / r["median_step_ms"] if r["median_step_ms"] else 0
        print(f"{v:>15}  {r['final_loss']:>15.4f}  {r['median_step_ms']:>15.1f}  "
              f"{r['median_tok_s']:>15.0f}  {r['peak_mem_mb']:>15.1f}"
              f"   ({speedup:.2f}x vs first)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=["baseline", "modded"],
                   choices=list_variants())
    p.add_argument("--tiny", action="store_true")
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    results = run_all(args.variants, tiny_cfg=args.tiny, max_steps=args.max_steps,
                      seq_len=args.seq_len, batch=args.batch, seed=args.seed)
    print_table(results)


if __name__ == "__main__":
    main()
