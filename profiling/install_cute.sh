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
# cut-cross-entropy backs the fused linear-CE (modded/dev fused_ce path); missing -> CE
# stays eager (materializes logits). Separate line so its failure can't abort the others.
# --no-deps is CRITICAL: cut-cross-entropy depends on a BARE `triton` (no pin), so a
# plain install greedily pulls the LATEST triton (3.7.0), which OVERWRITES the NGC
# image's torch-2.6-compatible triton (~3.2). torch 2.6's inductor then dies importing
# `AttrsDescriptor` from triton.compiler.compiler (removed in 3.7) -> EVERY torch.compile
# breaks: the modded variant's reduce-overhead CUDAGraph path AND CCE's own
# torch.compile-backed backward (sort_logit_avg). The container already ships a
# compatible triton + torch, so install CCE with NO deps and keep that triton.
# (Verified 2026-06-15: a bare install bumped triton 3.7.0 and broke compile;
# --no-deps keeps the run green.)
log "=== pip install --no-deps cut-cross-entropy (optional: wired fused linear-cross-entropy) ==="
pip install --no-deps cut-cross-entropy 2>&1 | tee -a "$LOG" || log "## cut-cross-entropy: install FAILED (non-fatal; linear-CE stays eager)"
# Belt-and-suspenders: --no-deps prevents a FRESH triton bump, but the NGC image ships
# its triton as `pytorch_triton` (not `triton`), so torch has NO triton pin in its
# metadata to restore from, and a triton already broken by an EARLIER run persists on a
# KEEP=1 host. So FUNCTIONALLY test torch.compile's inductor backend: import the exact
# module that the bad triton breaks (torch._inductor.runtime.hints, which does
# `from triton.compiler.compiler import AttrsDescriptor` — removed in triton>=3.4). If it
# fails, pin triton to 3.2.0 (the torch-2.6 pairing; still has AttrsDescriptor) and
# re-check. (Verified 2026-06-15 on H100: triton 3.7.0 -> ImportError; ==3.2.0 -> OK.)
log "=== verify torch.compile/inductor still imports (triton compat) ==="
if ! python -c "import torch._inductor.runtime.hints" 2>>"$LOG"; then
  log "## inductor import BROKEN (triton $(python -c 'import triton;print(triton.__version__)' 2>/dev/null) incompatible) -> pinning triton==3.2.0"
  pip install --no-deps "triton==3.2.0" 2>&1 | tee -a "$LOG" || log "## triton==3.2.0 pin FAILED"
  if python -c "import torch._inductor.runtime.hints" 2>>"$LOG"; then
    log "## inductor import OK after triton==3.2.0"
  else
    log "## inductor STILL broken after triton==3.2.0 — torch.compile variants will fail"
  fi
else
  log "## inductor import OK (triton $(python -c 'import triton;print(triton.__version__)' 2>/dev/null) compatible)"
fi

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
pip list 2>/dev/null | grep -iE "cutlass|quack|flash|cut-cross-entropy|cut_cross_entropy|^triton " | tee -a "$LOG" || log "(none of cutlass/quack/flash/cce installed)"

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
