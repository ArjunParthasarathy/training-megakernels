"""Optimizer construction and the Muon / AdamW parameter split.

The split rule (modded-nanoGPT / nanochat): hidden 2D weight *matrices* -> Muon;
everything else (1D norms/biases/scalars AND the token embedding / lm_head, which
are 2D but token-indexed) -> AdamW. See docs/PLAN.md and the 1D-vector discussion.
"""

from __future__ import annotations

import torch
from torch import nn

from .muon import Muon, gram_newton_schulz, newton_schulz_standard, orthogonalize

__all__ = ["Muon", "gram_newton_schulz", "newton_schulz_standard",
           "orthogonalize", "build_optimizers", "split_params"]


def split_params(model: nn.Module):
    """Return (muon_params, adamw_params).

    Muon: 2D params that are NOT the embedding/lm_head. AdamW: the rest.
    """
    # identify token-matrix params by object identity so tied weights aren't double-counted
    embed_ids = set()
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Embedding,)):
            embed_ids.add(id(mod.weight))
        if isinstance(mod, nn.Linear) and getattr(mod, "out_features", None) is not None:
            # lm_head (untied) is hidden->vocab; flag by name
            if name.endswith("lm_head"):
                embed_ids.add(id(mod.weight))

    muon, adamw, seen = [], [], set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        if p.ndim == 2 and id(p) not in embed_ids:
            muon.append(p)
        else:
            adamw.append(p)
    return muon, adamw


def build_optimizers(model: nn.Module, *, use_muon: bool, lr: float, muon_lr: float,
                     weight_decay: float, ns_impl: str = "gram"):
    """Build the optimizer set for a variant.

    use_muon=False (baseline): a single AdamW over all params.
    use_muon=True  (modded):   Muon over 2D hidden matrices + AdamW over the rest.
    Returns a list of optimizers (call .step()/.zero_grad() on each).
    """
    if not use_muon:
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  betas=(0.9, 0.95))]
    muon_params, adamw_params = split_params(model)
    opts = [
        Muon(muon_params, lr=muon_lr, ns_impl=ns_impl),
        torch.optim.AdamW(adamw_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)),
    ]
    return opts
