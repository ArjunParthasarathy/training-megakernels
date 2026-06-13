"""Data loading. Synthetic by default so every variant sees identical batches
(seeded) for apples-to-apples comparison; swap in a real FineWeb/tokenized loader
later behind the same interface.
"""

from __future__ import annotations

import torch


def synthetic_loader(vocab_size: int, batch_size: int, seq_len: int, *,
                     device: str, seed: int = 0):
    """Deterministic stream of random token batches: yields int64 [B, T].

    Same (seed, vocab, batch, seq_len) -> same batches across variants, so loss
    curves are directly comparable.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    while True:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        yield ids.to(device, non_blocking=True)


# Placeholder for the real pipeline (tokenized FineWeb-EDU shards, etc.).
def fineweb_loader(*args, **kwargs):  # pragma: no cover - not implemented yet
    raise NotImplementedError(
        "Real-data loader not wired yet; use synthetic_loader for now. "
        "Plan: tokenized FineWeb-EDU shards, document-packed to seq_len.")
