"""Muon optimizer + Newton-Schulz orthogonalization (standard and Gram).

Muon applies the orthogonal polar factor UVᵀ of each 2D gradient (never 1D — those
go to AdamW; see optim/__init__.py). The polar factor is approximated by a 5-step
quintic Newton-Schulz iteration.

Two equivalent implementations are provided, both pure-PyTorch and CPU-testable:

  newton_schulz_standard(G)  — the classic form, computes A = XXᵀ each step:
        X <- aX + b(XXᵀ)X + c(XXᵀ)²X

  gram_newton_schulz(G)      — the reformulation behind Tri Dao's CuTeDSL kernel.
        Key identity:  (XXᵀ)X = X(XᵀX).  So each step is a *right*-multiply by a
        polynomial in the SMALL Gram matrix S = XᵀX (k×k, k=min(m,n)):
              X_{t+1} = X_t (aI + bS_t + cS_t²),   M_t := aI + bS_t + cS_t²
        and S itself updates without touching the big matrix:
              S_{t+1} = X_{t+1}ᵀX_{t+1} = M_t S_t M_t      (M, S symmetric)
        Accumulate Q = M_0 M_1 ... M_{T-1}; the only big GEMMs are S_0 = XᵀX once
        and the final X_0 Q once. Everything in between is k×k — and S², M S M are
        symmetric, which is exactly where the CuTeDSL symmetric-GEMM trick halves
        the work (see megakernels/cute/newton_schulz.py).

In exact arithmetic the two are identical; tests/test_muon.py checks they agree
numerically and that the output is ~orthogonal.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Keller Jordan's coefficients (modded-nanoGPT), one quintic repeated 5x.
NS_COEFFS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5


def _acc(dtype: torch.dtype) -> torch.dtype:
    """fp32 for low precision (bf16/fp16 production), else keep (fp64 exact tests)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _normalize(G: Tensor) -> Tensor:
    return G / (G.norm() + 1e-7)


def newton_schulz_standard(G: Tensor, steps: int = NS_STEPS,
                           coeffs=NS_COEFFS) -> Tensor:
    """Classic Newton-Schulz. Orthogonalizes the 2D matrix G -> ~UVᵀ."""
    a, b, c = coeffs
    transpose = G.shape[0] < G.shape[1]              # iterate on the smaller XXᵀ
    X = G.T if transpose else G
    X = _normalize(X.to(_acc(G.dtype)))
    for _ in range(steps):
        A = X @ X.transpose(-2, -1)
        X = a * X + (b * A + c * (A @ A)) @ X
    return (X.T if transpose else X).to(G.dtype)


def gram_newton_schulz(G: Tensor, steps: int = NS_STEPS,
                       coeffs=NS_COEFFS) -> Tensor:
    """Gram reformulation: iterate on the small k×k Gram matrix S = XᵀX.

    Mathematically equivalent to newton_schulz_standard; far fewer FLOPs when G is
    rectangular. This is the algorithm the CuTeDSL symmetric-GEMM kernel accelerates.
    """
    a, b, c = coeffs
    # Orient so columns are the smaller dim -> S = XᵀX is k×k, k=min(m,n).
    transpose = G.shape[0] < G.shape[1]
    X0 = (G.T if transpose else G).to(_acc(G.dtype))
    X0 = _normalize(X0)
    k = X0.shape[1]
    I = torch.eye(k, device=X0.device, dtype=X0.dtype)

    S = X0.transpose(-2, -1) @ X0                    # big GEMM #1 (symmetric, SYRK)
    Q = I.clone()
    for _ in range(steps):
        M = a * I + b * S + c * (S @ S)              # k×k; S@S symmetric
        Q = Q @ M                                    # accumulate right-transform
        S = M @ S @ M                                # k×k; symmetric update
    X = X0 @ Q                                        # big GEMM #2: apply once
    return (X.T if transpose else X).to(G.dtype)


# Backend hook: 'gram' default (matches the CuTeDSL kernel path); on GPU the cute
# symmetric-GEMM version registers itself and is used instead.
_NS = {"standard": newton_schulz_standard, "gram": gram_newton_schulz}


def orthogonalize(G: Tensor, impl: str = "gram") -> Tensor:
    fn = _NS.get(impl)
    if fn is None:
        raise ValueError(f"unknown newton-schulz impl {impl!r}")
    return fn(G)


class Muon(torch.optim.Optimizer):
    """Muon for 2D parameters. 1D params must NOT be put in this optimizer.

    Update: Nesterov momentum on the grad, orthogonalize via Newton-Schulz, scale
    by sqrt(max(1, fan_out/fan_in)) (Keller's shape scaling), then SGD step.
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_impl: str = "gram", ns_steps: int = NS_STEPS):
        for p in (params if isinstance(params, list) else list(params)):
            if p.ndim != 2:
                raise ValueError(
                    f"Muon only accepts 2D params; got {p.ndim}D. Route 1D params "
                    f"(norms, biases) and embeddings to AdamW.")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_impl=ns_impl, ns_steps=ns_steps)
        super().__init__(params if isinstance(params, list) else list(params), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                buf = st.get("momentum_buffer")
                if buf is None:
                    buf = st["momentum_buffer"] = torch.zeros_like(g)
                buf.mul_(group["momentum"]).add_(g)
                update = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                ortho = orthogonalize(update, group["ns_impl"])
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5
                p.add_(ortho, alpha=-group["lr"] * scale)
        return loss
