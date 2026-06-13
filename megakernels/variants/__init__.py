"""Variant registry. ``get_variant(name)`` -> a configured TrainVariant."""

from __future__ import annotations

from .base import TrainVariant
from . import baseline, cudagraph, custom_backward, modded

_REGISTRY = {
    "baseline": baseline.make,
    "cudagraph": cudagraph.make,
    "modded": modded.make,
    "custom_backward": custom_backward.make,
}

# Friendly aliases. 'eager' names the pure-eager baseline (no graphs/compile);
# 'dev' is the short name for the custom-backward run (matches the dev branch).
_ALIASES = {"eager": "baseline", "dev": "custom_backward"}


def get_variant(name: str) -> TrainVariant:
    name = _ALIASES.get(name, name)
    if name not in _REGISTRY:
        raise ValueError(f"unknown variant {name!r}; choose from {list_variants()}")
    return _REGISTRY[name]()


def list_variants():
    return list(_REGISTRY) + list(_ALIASES)
