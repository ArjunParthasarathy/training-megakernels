#!/usr/bin/env bash
# Nsight Compute per-kernel deep dive: warp utilization, occupancy, memory
# throughput, stall reasons, roofline. SLOW: ncu --set full replays each kernel
# 30-60x to read every counter, so you MUST scope it to a few kernels or it runs
# for GPU-hours. We scope via the "profile_step" NVTX range emitted by
# profile_utils.py plus --launch-count.
#
# Produces <out>.ncu-rep -> open locally in the Nsight Compute UI (ncu-ui).
# Your local Nsight Compute version must be >= the ncu version used here.
# Requires GPU perf-counter permission (see fix_profiling_perms.sh for
# ERR_NVGPUCTRPERM).
#
# Usage: ./run_ncu.sh [output_basename] [-- extra args to train_qwen3.py]
#   Env knobs:
#     NCU_SET           metric set: default (fast, ~1 pass) | full (everything)  [default: full]
#     NCU_LAUNCH_COUNT  max kernels to profile                                   [default: 40]
#     NCU_KERNEL_REGEX  optional -k regex to target named kernels (e.g. gemm|flash_attn|norm)
#   Tip: export PROFILE_STEPS=1 so only ONE step's kernels are in scope.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-kernels}"; shift || true
[[ "${1:-}" == "--" ]] && shift || true

SET="${NCU_SET:-full}"
COUNT="${NCU_LAUNCH_COUNT:-40}"

ARGS=(
  --target-processes all
  --nvtx --nvtx-include "profile_step/"
  --launch-count "$COUNT"
  --set "$SET"
  -o "$OUT" -f
)
if [[ -n "${NCU_KERNEL_REGEX:-}" ]]; then
  ARGS+=(-k "regex:${NCU_KERNEL_REGEX}")
fi

PROFILE_STEPS="${PROFILE_STEPS:-1}" ncu "${ARGS[@]}" python train_qwen3.py "$@"

echo "Wrote ${OUT}.ncu-rep"
echo "Open locally:  ncu-ui ${OUT}.ncu-rep"
