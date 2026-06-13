#!/usr/bin/env bash
# Nsight Systems timeline (kernel launches, bubbles/gaps, CPU<->GPU overlap) for a
# chosen training run. Cheap (~5-20% overhead); run this FIRST, then run_ncu.sh on
# the kernels it flags. Produces <out>.nsys-rep -> open in Nsight Systems UI.
#
# Usage: ./run_nsys.sh <variant> [out_basename] [-- extra args to megakernels.train]
#   variant: baseline | modded | dev   (which training run to profile)
#   PROFILE knobs come from megakernels.train flags (--warmup, --max-steps, ...).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:-baseline}"; shift || true
OUT="${1:-timeline-$VARIANT}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true

# --capture-range=cudaProfilerApi pairs with --profile in megakernels.train
# (torch.cuda.profiler.start/stop) so only the post-warmup steps are recorded.
nsys profile \
  -w true \
  -t cuda,nvtx,osrt,cudnn,cublas \
  -s none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --gpu-metrics-device=0 \
  -o "$OUT" -f true -x true \
  python -m megakernels.train --variant "$VARIANT" --profile "$@"

echo "Wrote ${OUT}.nsys-rep  (variant=$VARIANT)"
nsys stats --report cuda_gpu_kern_sum "${OUT}.nsys-rep" 2>/dev/null | head -20 || true
echo "Open locally:  nsys-ui ${OUT}.nsys-rep"
