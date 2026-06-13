"""P0 — Gram Newton-Schulz with a CuTeDSL symmetric-GEMM (GPU only).

Imported only when a GPU + CuTeDSL are present (see cute/__init__.py). On success
it overrides the Muon optimizer's 'gram' Newton-Schulz with this accelerated
version. The algorithm is the one validated on CPU in megakernels/optim/muon.py
(gram_newton_schulz); here the k×k symmetric products S² and M·S·M are computed by
a custom symmetric GEMM that fills only the lower triangle and reflects it — Tri
Dao's trick (tridao.me/blog/2026/gram-newton-schulz, quack/gemm_symmetric.py).

STATUS: skeleton. The driver + symmetric-GEMM scaffold are here; the WGMMA/TMA
mainloop is marked TODO. Validate on a rented H100 via vast/launch.sh before
trusting it; until the TODO is filled, set ns_impl='gram' (eager) for correctness.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from . import register
from ..optim import muon


@cute.kernel
def _symmetric_gemm_kernel(mA: cute.Tensor, mOut: cute.Tensor):
    """C = A @ Aᵀ, computing only lower-triangular output tiles + reflecting.

    The two differences from a vanilla GEMM (per the Gram-NS post):
      1. a TRIANGULAR tile scheduler — map the 1D block index onto lower-triangular
         tile coords (i, j) with i >= j via triangular-number indexing, so every
         scheduled tile does real work and SMs stay balanced (a naive 'skip upper
         tiles' approach load-imbalances).
      2. a transposed epilogue write — off-diagonal tiles write both (i,j) and
         (j,i); diagonal tiles write once.
    """
    bid, _, _ = cute.arch.block_idx()
    # TODO(gpu): triangular index -> (tile_i, tile_j); TMA-load A tiles; WGMMA
    # accumulate A_i @ A_jᵀ; if i==j compute full, else write C[i,j] and C[j,i]=Cᵀ.
    raise NotImplementedError("symmetric-GEMM mainloop TODO — validate on H100")


@cute.jit
def _symmetric_gemm(mA: cute.Tensor, mOut: cute.Tensor):
    m, _ = mA.shape
    tile = 128
    n_tiles = (m + tile - 1) // tile
    n_blocks = n_tiles * (n_tiles + 1) // 2          # lower-triangular tile count
    _symmetric_gemm_kernel(mA, mOut).launch(
        grid=(n_blocks, 1, 1), block=(256, 1, 1),
        stream=torch.cuda.current_stream().cuda_stream)


def _syrk(X: torch.Tensor) -> torch.Tensor:
    """Symmetric C = X @ Xᵀ via the cute kernel (falls back to dense matmul TODO)."""
    m = X.shape[0]
    out = torch.empty((m, m), device=X.device, dtype=X.dtype)
    _symmetric_gemm(from_dlpack(X, assumed_align=16), from_dlpack(out, assumed_align=16))
    return out


def gram_newton_schulz_cute(G: torch.Tensor, steps: int = muon.NS_STEPS,
                            coeffs=muon.NS_COEFFS) -> torch.Tensor:
    """Same math as muon.gram_newton_schulz, with symmetric GEMMs for S²/MSM.

    Keeps the small k×k Gram matrix resident across the 5 iterations (no inter-step
    HBM round-trips). See the CPU reference for the exact recurrence.
    """
    # The recurrence is identical to the eager gram version; only the k×k symmetric
    # products are swapped for _syrk-based ones. Until the kernel mainloop lands,
    # delegate to the validated eager implementation so callers stay correct.
    return muon.gram_newton_schulz(G, steps, coeffs)


# Register: Muon's ns_impl='gram' now resolves to this on GPU. (Currently a
# correctness-preserving passthrough; flips to the symmetric-GEMM path once the
# kernel TODO is done.)
muon._NS["gram"] = gram_newton_schulz_cute
register("newton_schulz", gram_newton_schulz_cute)
