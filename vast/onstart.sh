#!/usr/bin/env bash
# Vast.ai onstart script (passed via `vastai create instance --onstart`).
# Runs at container boot. Its only jobs here are:
#   1. enable GPU perf counters for ncu (best-effort; needs SYS_ADMIN from template),
#   2. arm a watchdog that self-destroys the instance after MAX_LIFETIME_SECS so a
#      dead launcher can never leak billing,
#   3. signal readiness so the launcher can rsync code up and drive the run.
# The actual profiling is driven by the launcher over SSH (so it works with
# uncommitted local code and can copy .nsys-rep/.ncu-rep back before destroy).
set -uo pipefail

MAX_LIFETIME_SECS="${MAX_LIFETIME_SECS:-3600}"  # hard cap: never bill past 1h by default

# (1) best-effort perf-counter enable for Nsight Compute
echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \
  > /etc/modprobe.d/nvidia-profiler.conf 2>/dev/null || true

# (2) self-destruct watchdog — uses the per-instance CONTAINER_ID/CONTAINER_API_KEY
# that Vast injects into every instance. This is the billing safety net.
( sleep "$MAX_LIFETIME_SECS"
  pip install -q vastai 2>/dev/null || true
  vastai destroy instance "$CONTAINER_ID" --api-key "$CONTAINER_API_KEY"
) >/var/log/vast_watchdog.log 2>&1 &

# (3) tools the launcher's SSH run will need
pip install -q vastai 2>/dev/null || true
mkdir -p /workspace/out

echo "ONSTART_READY"  # launcher greps vastai logs for this marker
