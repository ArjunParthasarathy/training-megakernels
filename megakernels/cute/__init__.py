"""CuTeDSL kernel backend (GPU-only).

Everything here imports ``cutlass.cute`` and requires an NVIDIA GPU (Hopper /
Blackwell), so it is **import-guarded**: on a CPU/macOS box ``HAVE_CUTE`` is False
and the dispatch layer (``megakernels.kernels``) silently falls back to the eager
reference. The kernels are validated on a rented H100 via ``vast/launch.sh``, not
locally.

Implementation status (see docs/PLAN.md for the rationale):
  P0  newton_schulz   Gram-NS symmetric-GEMM   -> cute/newton_schulz.py  (skeleton)
  P1  linear_ce       fused unembed + CE       -> reuse Dao-AILab/quack  (wired)
  P2  rms_norm        RMSNorm (+residual)      -> reuse quack            (wired)
  P3  rope_qk_norm    fused QK-norm + RoPE      -> cute/rope_qk_norm.py   (skeleton)
  P5  swiglu          fused SwiGLU MLP          -> cute/swiglu.py         (skeleton)
  attention            GQA fwd/bwd              -> reuse flash-attn-4     (wired)
"""

from __future__ import annotations

HAVE_CUTE = False
_IMPORT_ERROR = None

try:  # pragma: no cover - exercised only on GPU
    import cutlass.cute  # noqa: F401
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CuTeDSL requires a CUDA device")
    HAVE_CUTE = True
except Exception as e:  # noqa: BLE001
    _IMPORT_ERROR = e
    HAVE_CUTE = False


# Registry of *implemented* cute ops, keyed by op name. Kernel modules call
# register(name, fn) when they import successfully on a GPU. Anything not present
# here is transparently served by the eager reference instead — so a half-finished
# kernel set still runs end to end.
OPS: dict = {}


def register(name: str, fn) -> None:  # pragma: no cover - GPU only
    OPS[name] = fn


def available() -> bool:
    """True iff the CuTeDSL backend can actually run on this machine."""
    return HAVE_CUTE


def why_unavailable() -> str:
    return "" if HAVE_CUTE else f"CuTeDSL backend unavailable: {_IMPORT_ERROR!r}"


if HAVE_CUTE:  # pragma: no cover - GPU only
    # Import kernel modules so they register into OPS. Each is wrapped so one
    # broken/unfinished kernel doesn't disable the whole backend.
    for _mod in ("newton_schulz", "rope_qk_norm", "swiglu", "wired"):
        try:
            __import__(f"{__name__}.{_mod}")
        except Exception:  # noqa: BLE001
            pass

