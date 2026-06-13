"""P5 — fused SwiGLU MLP (GPU only, skeleton).

gate/up GEMMs -> silu(gate)*up fused in the GEMM epilogue -> down GEMM (blockscaled
FP8 possible on Blackwell). GEMM-epilogue fusion is CuTeDSL's sweet spot.

STATUS: skeleton — registers nothing until implemented, so swiglu uses the eager
reference. Implement the activation as a CUTLASS epilogue on the up/gate GEMM.
"""

from __future__ import annotations

# TODO(gpu): implement fused swiglu cute kernel and register("swiglu", ...)
