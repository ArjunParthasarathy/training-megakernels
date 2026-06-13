#!/usr/bin/env bash
# Nsight Systems timeline: kernel launches, bubbles/gaps, CPU<->GPU overlap, host
# overhead. Cheap (~5-20% overhead) and traces the whole captured region. Run this
# FIRST to find where the GPU is idle and which kernels are expensive, THEN use
# run_ncu.sh on just those kernels.
#
# Produces <out>.nsys-rep -> open locally in the Nsight Systems UI (nsys-ui).
# Your local Nsight Systems version must be >= the nsys version used here.
#
# Usage: ./run_nsys.sh [output_basename] [-- extra args to train_qwen3.py]
#   PROFILE_WARMUP / PROFILE_STEPS env vars control warmup + #steps captured.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-timeline}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true

# --capture-range=cudaProfilerApi pairs with torch.cuda.profiler.start/stop in
# profile_utils.py so only the post-warmup steps are recorded.
nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-device=0 \
  -o "$OUT" -f true -x true \
  python train_qwen3.py "$@"

echo "Wrote ${OUT}.nsys-rep"
echo "Quick text summary:"
nsys stats --report cuda_gpu_kern_sum "${OUT}.nsys-rep" || true
echo "Open locally:  nsys-ui ${OUT}.nsys-rep"
