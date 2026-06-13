#!/usr/bin/env bash
# Install the CuTeDSL "cute stack" and VERIFY it actually imports — with everything
# logged. The old launcher ran `pip install -q A B C >/dev/null 2>&1 || true`, which
# hid three real failure modes behind a silent eager fallback:
#   * a multi-package resolve leaving a broken `cutlass` module      (NVIDIA/cutlass#3132)
#   * the base image's bundled `cutlass` shadowing the DSL's `cute`  (NVIDIA/cutlass#2446)
#   * a host driver too old for the wheels (nvidia-smi CUDA < toolkit; cute stack needs CUDA>=12.3)
# All three packages DO exist on PyPI (nvidia-cutlass-dsl, quack-kernels, flash-attn-4);
# the failure is environmental, so we log each step and run the exact import the
# dispatch layer gates on (megakernels/cute/__init__.py: `import cutlass.cute`).
#
# Exit 0  -> `import cutlass.cute` works; the cute backend will be used.
# Exit 3  -> import failed; the run will fall back to eager (the log says why).
#
# Usage: bash install_cute.sh [logfile]   (default /workspace/out/install-cute.log)
set -uo pipefail   # NOT -e: we handle failures explicitly so the log always completes
LOG="${1:-/workspace/out/install-cute.log}"
mkdir -p "$(dirname "$LOG")"
log() { echo "$@" | tee -a "$LOG"; }

log "=== cute-stack install $(date -u +%FT%TZ) ==="
log "## python : $(python --version 2>&1)"
log "## pip    : $(pip --version 2>&1)"
python - 2>&1 <<'PY' | tee -a "$LOG"
import torch
print(f"## torch  : {torch.__version__}  built-for-cuda {torch.version.cuda}")
print(f"## cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"## device : {torch.cuda.get_device_name(0)}")
PY
log "## driver : $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>&1 | head -1)"
log "## nvidia-smi CUDA (driver max): $(nvidia-smi 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1)"

# Pre-clean any pre-existing cutlass/DSL so a stale or bundled module can't shadow
# the freshly-installed DSL (NVIDIA/cutlass#2446, #3132). Harmless if absent.
log "=== pre-clean any pre-existing cutlass DSL ==="
pip uninstall -y nvidia-cutlass-dsl nvidia-cutlass \
  nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 \
  2>&1 | tee -a "$LOG" || true

# Install each package SEPARATELY so one failure can't abort the others (bundling
# them is exactly what hid the original bug: flash-attn-4 failing took the whole
# transaction down with it). The import-critical package is nvidia-cutlass-dsl — it
# alone provides `cutlass.cute`, so the cute backend activates as long as IT lands.
# quack-kernels + flash-attn-4 only back the "wired" ops (RMSNorm/CE/attention);
# missing, those ops just stay eager while the cute backend is still used elsewhere.
# NO -q, NO >/dev/null: we want the real pip output in the log.
log "=== pip install nvidia-cutlass-dsl (import-critical) ==="
pip install nvidia-cutlass-dsl 2>&1 | tee -a "$LOG" || log "## nvidia-cutlass-dsl: install FAILED (cute backend will be unavailable)"
log "=== pip install quack-kernels (optional: wired RMSNorm/cross-entropy) ==="
pip install quack-kernels 2>&1 | tee -a "$LOG" || log "## quack-kernels: install FAILED (non-fatal; those ops stay eager)"
# flash-attn-4 publishes ONLY pre-release wheels (4.0.0bN) — a bare
# `pip install flash-attn-4` errors "No matching distribution found", so it needs --pre.
log "=== pip install --pre flash-attn-4 (optional: wired attention; pre-release only) ==="
pip install --pre flash-attn-4 2>&1 | tee -a "$LOG" || log "## flash-attn-4: install FAILED (non-fatal; attention stays eager)"

# nvidia-cutlass-dsl has NO cuda pin, so the installs above greedily pull the LATEST
# cuda-python (13.x). But cuda-python 13's *base* wheel ships without the
# cuda.bindings.driver/.runtime .so, so `import cutlass.cute` dies with
# "No module named 'cuda.bindings.driver'" on a CUDA-12 host. Pin cuda-python to the
# MAJOR that matches this container's CUDA (torch.version.cuda): the cu12 wheel
# (cuda-python 12.9.x) bundles those bindings and imports cleanly. Done LAST so it
# overrides whatever quack/flash dragged in. (Verified 2026-06-13: cuda-python<13 ->
# import cutlass.cute OK on a 12.8 host.)
CUDA_MAJOR=$(python -c "import torch; print((torch.version.cuda or '12').split('.')[0])" 2>/dev/null || echo 12)
log "=== pin cuda-python to CUDA major $CUDA_MAJOR (matches the container toolkit) ==="
if [ "$CUDA_MAJOR" = "13" ]; then
  pip install "cuda-python>=13,<14" 2>&1 | tee -a "$LOG" || log "## cuda-python cu13 pin FAILED"
else
  pip install "cuda-python>=12,<13" 2>&1 | tee -a "$LOG" || log "## cuda-python cu12 pin FAILED"
fi

log "=== installed versions ==="
pip list 2>/dev/null | grep -iE "cutlass|quack|flash" | tee -a "$LOG" || log "(none of cutlass/quack/flash installed)"

# The decisive check: the import that megakernels/cute/__init__.py runs. pipefail
# makes the pipeline return python's exit code (not tee's 0), so the `if` is honest.
log "=== verify: import cutlass.cute ==="
if python - 2>&1 <<'PY' | tee -a "$LOG"
import sys
try:
    import cutlass.cute  # noqa: F401  (the exact gate in megakernels/cute/__init__.py)
    import cutlass
    print(f"## import cutlass.cute: OK  (cutlass at {cutlass.__file__})")
except Exception as e:
    print(f"## import cutlass.cute: FAILED -> {type(e).__name__}: {e}")
    sys.exit(3)
PY
then
  log "## RESULT: cute backend AVAILABLE"
  exit 0
else
  log "## RESULT: cute backend UNAVAILABLE — run will fall back to eager (reason above)"
  exit 3
fi
