"""Patch-statistics encoder: four hand-made features per 16 x 16 patch, no learning.

Used for tests and as a simple image baseline next to the image-free controls:
mean, standard deviation, maximum, and mean absolute horizontal + vertical gradient.
"""

from __future__ import annotations

import numpy as np

from .base import Encoder, register


@register("patchstats")
class PatchStats(Encoder):
    name = "patchstats"
    patch_px = 16
    dim = 4

    def _encode(self, image: np.ndarray) -> np.ndarray:
        p = self.patch_px
        h, w = image.shape[0] // p, image.shape[1] // p
        blocks = image.reshape(h, p, w, p).transpose(0, 2, 1, 3).reshape(h, w, p * p)
        gy = np.abs(np.diff(image, axis=0, append=image[-1:, :]))
        gx = np.abs(np.diff(image, axis=1, append=image[:, -1:]))
        grad = (gx + gy).reshape(h, p, w, p).transpose(0, 2, 1, 3).reshape(h, w, p * p)
        return np.stack([blocks.mean(-1), blocks.std(-1), blocks.max(-1), grad.mean(-1)], axis=-1)
