#!/usr/bin/env bash
# Nsight Systems timeline (kernel launches, bubbles/gaps, CPU<->GPU overlap) for a
# chosen training run. Cheap (~5-20% overhead); run this FIRST, then run_ncu.sh on
# the kernels it flags. Produces <out>.nsys-rep -> open in Nsight Systems UI.
#
# Usage: ./run_nsys.sh <variant> [out_basename] [-- extra args to megakernels.train]
#   variant: baseline (eager) | cudagraph | modded | dev   (which run to profile)
#   PROFILE knobs come from megakernels.train flags (--warmup, --max-steps, ...).
#   WARMUP=<n> sets the pre-capture warmup (default below; clears the compile cost).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:-baseline}"; shift || true
OUT="${1:-timeline-$VARIANT}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true

# Warmup steps to run BEFORE the nsys capture opens (megakernels.train starts the
# cudaProfilerApi range at step == warmup). This must clear the one-time
# torch.compile cost: the `cudagraph` variant compiles on step 0 and Inductor's
# CUDAGraph trees then run a few warmup/record iterations before steady-state
# replay, so a too-small warmup would capture the compile spike, not the graphed
# steps. We set it uniformly for EVERY variant (eager included) — eager has no
# compile, but using the same warmup means all variants capture the same
# steady-state window, so their timelines are directly comparable. Override with
# `WARMUP=<n> ./run_nsys.sh ...` or a trailing `--warmup <n>` (last value wins).
WARMUP="${WARMUP:-15}"

# --capture-range=cudaProfilerApi pairs with --profile in megakernels.train
# (torch.cuda.profiler.start/stop) so only the post-warmup steps are recorded.
# NOTE: no --gpu-metrics-device — that add-on needs GPU perf counters (SYS_ADMIN)
# which multi-tenant hosts deny (ERR_NVGPUCTRPERM), and it would make nsys exit
# with a usage error. The `-t cuda` trace already captures kernel launches, gaps,
# and cudaMemcpy host<->device transfers, none of which need counters.
nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o "$OUT" -f true -x true \
  python -m megakernels.train --variant "$VARIANT" --profile --warmup "$WARMUP" "$@"

echo "Wrote ${OUT}.nsys-rep  (variant=$VARIANT)"
nsys stats --report cuda_gpu_kern_sum "${OUT}.nsys-rep" 2>/dev/null | head -20 || true
# host<->device transfer summaries (the other thing we care about): time + bytes
# moved by HtoD/DtoH/DtoD memcpy, so transfer overhead shows up next to kernels.
echo "--- host<->device transfers ---"
nsys stats --report cuda_gpu_mem_time_sum "${OUT}.nsys-rep" 2>/dev/null | head -15 || true
nsys stats --report cuda_gpu_mem_size_sum "${OUT}.nsys-rep" 2>/dev/null | head -15 || true
echo "Open locally:  nsys-ui ${OUT}.nsys-rep"
