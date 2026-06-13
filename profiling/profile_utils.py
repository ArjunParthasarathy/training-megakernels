"""Profiling instrumentation for the training loop.

This module provides a single helper, ``profiled_loop``, that wraps a training
step iterator so the **same** ``train.py`` works under both Nsight Systems
(``nsys``, timeline / bubbles / launches) and Nsight Compute (``ncu``, per-kernel
warp utilization / occupancy / memory). The CLI tool you launch with decides what
gets recorded; this code just:

  * warms up ``WARMUP`` steps (let cuDNN autotune, the caching allocator, and any
    JIT-compiled CuTeDSL kernels settle) so you don't profile cold-start noise,
  * opens a ``cudaProfilerApi`` capture range (``torch.cuda.profiler.start/stop``)
    that ``nsys --capture-range=cudaProfilerApi`` keys off,
  * annotates each step + sub-phase (forward / backward / optimizer) with NVTX
    ranges so they are filterable via ``ncu --nvtx-include`` and visible as named
    bars on the nsys timeline,
  * profiles only ``PROFILE_STEPS`` steps and then stops the iterator.

Why warmup+few-steps: ``ncu --set full`` replays *each* kernel 30-60x to read all
counters, so profiling a whole training run takes GPU-hours. You profile a couple
of steps, not the run.

torch.profiler note: PyTorch's built-in ``torch.profiler`` emits Chrome/TensorBoard
traces, NOT ``.ncu-rep`` / ``.nsys-rep``. It will not open in the Nsight Compute UI.
For Nsight you must wrap ``python train.py`` with the external ``ncu`` / ``nsys``
CLIs, which is exactly what this harness is designed for.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

import torch

# Tunable via env so the SAME train.py serves nsys (more steps OK) and ncu (1 step).
WARMUP = int(os.environ.get("PROFILE_WARMUP", "10"))
PROFILE_STEPS = int(os.environ.get("PROFILE_STEPS", "3"))

T = TypeVar("T")


@contextmanager
def nvtx_range(name: str):
    """Named NVTX range; filterable by ``ncu --nvtx-include`` and shown by nsys."""
    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def profiled_loop(steps: Iterable[T]) -> Iterator[T]:
    """Wrap a step iterable so warmup/capture/stop is handled automatically.

    Usage::

        from profile_utils import profiled_loop, nvtx_range
        for step, batch in enumerate(profiled_loop(dataloader)):
            with nvtx_range("forward"):
                loss = model(batch).loss
            with nvtx_range("backward"):
                loss.backward()
            with nvtx_range("optimizer"):
                opt.step(); opt.zero_grad(set_to_none=True)

    The outer ``profile_step`` NVTX range and the warmup/capture/stop logic are
    inserted here so the caller only annotates sub-phases.
    """
    capturing = False
    for i, item in enumerate(steps):
        if i == WARMUP:
            torch.cuda.synchronize()
            torch.cuda.profiler.start()  # nsys --capture-range=cudaProfilerApi hooks here
            capturing = True

        torch.cuda.nvtx.range_push("profile_step")
        try:
            yield item
        finally:
            torch.cuda.nvtx.range_pop()  # profile_step

        if capturing and i == WARMUP + PROFILE_STEPS - 1:
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
            break
