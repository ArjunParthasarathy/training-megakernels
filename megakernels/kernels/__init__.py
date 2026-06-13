"""Kernel dispatch: route each fusible op to the eager reference or a CuTeDSL
kernel, chosen by the active backend.

The model code calls these functions and never imports a backend directly, so a
variant flips behaviour just by selecting a backend (``set_backend``). Any op the
cute backend hasn't implemented falls back to eager automatically.
"""

from __future__ import annotations

import threading

from .. import cute
from . import eager

_state = threading.local()


def resolve_backend(name: str) -> str:
    """'auto' -> 'cute' if a GPU + CuTeDSL are present, else 'eager'."""
    if name == "auto":
        return "cute" if cute.available() else "eager"
    if name not in ("eager", "cute"):
        raise ValueError(f"unknown kernel backend {name!r}")
    return name


def set_backend(name: str) -> str:
    backend = resolve_backend(name)
    _state.backend = backend
    return backend


def get_backend() -> str:
    return getattr(_state, "backend", "eager")


def _impl(op: str):
    """Return the callable for ``op`` under the current backend (cute or eager)."""
    if get_backend() == "cute" and op in cute.OPS:
        return cute.OPS[op]
    return getattr(eager, op)


# --- public ops (signatures mirror megakernels.kernels.eager) ---------------

def rms_norm(*a, **k):
    return _impl("rms_norm")(*a, **k)


def qk_norm(*a, **k):
    return _impl("qk_norm")(*a, **k)


def apply_rope(*a, **k):
    return _impl("apply_rope")(*a, **k)


def swiglu(*a, **k):
    return _impl("swiglu")(*a, **k)


def attention(*a, **k):
    return _impl("attention")(*a, **k)


def linear_cross_entropy(*a, **k):
    return _impl("linear_cross_entropy")(*a, **k)


# re-export helpers that have no backend variant
rotary_cos_sin = eager.rotary_cos_sin
