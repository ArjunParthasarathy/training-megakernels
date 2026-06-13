#!/usr/bin/env bash
# Fix ERR_NVGPUCTRPERM ("user does not have permission to access NVIDIA GPU
# Performance Counters"). Nsight Compute (and nsys --gpu-metrics-device) read
# hardware perf counters, which are admin-gated by default. Run this on the
# rented H100 BEFORE run_ncu.sh.
#
# nsys *timeline* tracing (cuda/nvtx/cudnn/cublas) does NOT need this — only ncu
# and nsys GPU-metrics sampling do. So even on a locked-down host you still get
# the Nsight Systems timeline.
set -uo pipefail

echo "== current perf-counter permission =="
cat /proc/driver/nvidia/params 2>/dev/null | grep -i RmProfilingAdminOnly || \
  echo "(could not read /proc/driver/nvidia/params)"

# Persistent host-level fix (needs host root; survives reboot, applies to all users).
if [[ -w /etc/modprobe.d ]] || [[ "$(id -u)" == "0" ]]; then
  echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
    > /etc/modprobe.d/nvidia-profiler.conf 2>/dev/null \
    && echo "wrote /etc/modprobe.d/nvidia-profiler.conf (reload module or reboot to apply)"
fi

# Try to reload the param live without reboot (works if no process holds the GPU).
modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0 2>/dev/null \
  && echo "reloaded nvidia module with profiling enabled" \
  || echo "could not live-reload nvidia module (GPU busy or not root) — may need reboot"

echo "== after =="
cat /proc/driver/nvidia/params 2>/dev/null | grep -i RmProfilingAdminOnly || true
echo
echo "RmProfilingAdminOnly: 0  -> all users allowed (good)"
echo "RmProfilingAdminOnly: 1  -> still admin-only."
echo
echo "In Docker you ALSO need the container launched with --cap-add=SYS_ADMIN"
echo "(or --privileged). On Vast.ai, set that via a Template's 'Docker options'."
echo "Many shared/multi-tenant hosts will NOT grant counters regardless — if ncu"
echo "still fails, rent a verified whole-machine host, or fall back to nsys-only."
