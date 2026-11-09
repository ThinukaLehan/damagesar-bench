"""Encoder plug-in interface (FR5, NFR5: a new encoder is one class in one file).

An encoder turns one preprocessed SAR tile into a grid of token features that covers the
whole tile. Everything else (resampling, pooling, probes) is shared, so encoders are compared
on identical inputs (FYP Master §5.4).

Input contract for ``encode``: float32 array (H, W) in [0, 1] (8-bit SAR / 255; no-data 0),
with H and W multiples of ``patch_px``. Output: float32 array (H / patch, W / patch, dim).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

_REGISTRY: dict[str, Callable[..., "Encoder"]] = {}


class EncoderError(RuntimeError):
    pass


class Encoder(ABC):
    name: str = ""
    patch_px: int = 16
    dim: int = 0

    @abstractmethod
    def _encode(self, image: np.ndarray) -> np.ndarray:
        """(H, W) float32 in [0, 1] -> (H/patch, W/patch, dim)."""

    def encode(self, image: np.ndarray) -> np.ndarray:
        img = np.asarray(image, dtype=np.float32)
        if img.ndim != 2:
            raise EncoderError(f"{self.name}: expected a single-channel (H, W) image, got shape {img.shape}")
        h, w = img.shape
        if h % self.patch_px or w % self.patch_px:
            raise EncoderError(f"{self.name}: image {h}x{w} is not a multiple of the patch size {self.patch_px}")
        if img.min() < 0 or img.max() > 1:
            raise EncoderError(f"{self.name}: image values must be in [0, 1]")
        out = np.asarray(self._encode(img), dtype=np.float32)
        expected = (h // self.patch_px, w // self.patch_px, self.dim)
        if out.shape != expected:
            raise EncoderError(f"{self.name}: output shape {out.shape} != expected {expected}")
        if not np.isfinite(out).all():
            raise EncoderError(f"{self.name}: output contains NaN or inf")
        return out


def register(name: str):
    """Class decorator: ``@register("dinov3_vitb16")`` makes the encoder available by name."""
    def deco(factory):
        if name in _REGISTRY:
            raise EncoderError(f"encoder {name!r} registered twice")
        _REGISTRY[name] = factory
        return factory
    return deco


def _load_builtins() -> None:
    from . import hf, patchstats  # noqa: F401  (registration side effect)


def available() -> list[str]:
    _load_builtins()
    return sorted(_REGISTRY)


def get_encoder(name: str, **kwargs) -> Encoder:
    _load_builtins()
    if name not in _REGISTRY:
        raise EncoderError(f"unknown encoder {name!r}; available: {sorted(_REGISTRY)}")
    enc = _REGISTRY[name](**kwargs)
    if not enc.name:
        enc.name = name
    return enc
