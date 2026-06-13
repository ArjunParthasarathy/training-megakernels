#!/usr/bin/env bash
# Nsight Compute per-kernel deep dive (warp util, occupancy, memory, stalls) for a
# chosen training run. SLOW: ncu --set full replays each kernel 30-60x, so it is
# scoped to the "profile_step" NVTX range + --launch-count. Produces <out>.ncu-rep
# -> open in Nsight Compute UI. Needs GPU perf-counter perms (fix_profiling_perms.sh).
#
# Usage: ./run_ncu.sh <variant> [out_basename] [-- extra args to megakernels.train]
#   variant: baseline | modded | dev
#   Env: NCU_SET (full|default), NCU_LAUNCH_COUNT (default 40), NCU_KERNEL_REGEX
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

VARIANT="${1:-baseline}"; shift || true
OUT="${1:-kernels-$VARIANT}"; shift || true
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
[[ -n "${NCU_KERNEL_REGEX:-}" ]] && ARGS+=(-k "regex:${NCU_KERNEL_REGEX}")

# --warmup 3 keeps the ncu run short (profiles a couple of steps after warmup).
ncu "${ARGS[@]}" python -m megakernels.train --variant "$VARIANT" --profile --warmup 3 "$@"

echo "Wrote ${OUT}.ncu-rep  (variant=$VARIANT)"
echo "Open locally:  ncu-ui ${OUT}.ncu-rep"
