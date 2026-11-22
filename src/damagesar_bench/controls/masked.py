"""Masked-building control (exploratory; added 22 Nov 2026, after the freeze, on development events only).

The ring control (``pooling/footprint.py``, ``ring:<m>``) pools only tokens that contain no building pixel. In a
Vision Transformer every token attends to the whole tile, so those tokens can still carry information about the
building. This control removes the buildings from the image *before* encoding:

* every building pixel of the tile (all footprints in the building table, excluded ones too) is masked, as drawn
  and, where the tile has a confident footprint-to-SAR offset (``c2_offsets.matched_shifts``), also at the shifted
  position, because the radar may record the building there; both masks are grown by ``dilate_m`` metres;
* masked pixels are replaced by the median of the tile's valid pixels (value > 0) outside the mask; no-data pixels
  (value 0) stay 0, so the no-data rule is unchanged;
* the masked tile is encoded and pooled over the ring as usual; the rows are renamed to pooling
  ``ring<m>m_masked`` so they can sit next to the unmasked ring in one benchmark run.

No label is read. The masked region's shape still shows where buildings are (like the image-free control), but no
building pixel value reaches the encoder.
"""

from __future__ import annotations

import json
from typing import Callable

import numpy as np
import pandas as pd

from ..pooling.footprint import TileGeometry, building_mask, shift_polygons


def masked_name(ring_m: float) -> str:
    return f"ring{ring_m:g}m_masked"


def building_pixel_mask(tile_buildings: pd.DataFrame, geom: TileGeometry, dilate_m: float = 1.0,
                        shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """Union of all footprints of one tile (H x W bool), as drawn and, if ``shift`` is non-zero, also shifted."""
    full = np.zeros((geom.height, geom.width), dtype=bool)
    offsets = [(0.0, 0.0)] + ([tuple(map(float, shift))] if any(abs(v) > 0 for v in shift) else [])
    for seg in tile_buildings["segmentation_px"]:
        base = json.loads(seg)
        for dy, dx in offsets:
            polys = shift_polygons(base, dy, dx, geom) if (dy or dx) else base
            m, r0, c0 = building_mask(polys, geom, dilate_m)
            if m.size == 0 or not m.any():
                continue
            h, w = m.shape
            r1, c1 = max(r0, 0), max(c0, 0)
            r2, c2 = min(r0 + h, geom.height), min(c0 + w, geom.width)
            if r2 > r1 and c2 > c1:
                full[r1:r2, c1:c2] |= m[r1 - r0:r2 - r0, c1 - c0:c2 - c0]
    return full


def mask_tile(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace masked pixels by the median of the valid unmasked pixels; keep no-data (0) as 0."""
    img = np.asarray(image, dtype=np.float32)
    if img.shape != mask.shape:
        raise ValueError(f"image {img.shape} and mask {mask.shape} differ")
    valid = img > 0
    background = img[valid & ~mask]
    fill = float(np.median(background)) if background.size else 0.0
    out = img.copy()
    out[mask & valid] = fill
    return out


def masked_reader(read_tile: Callable[[str], tuple[np.ndarray, TileGeometry]], buildings: pd.DataFrame,
                  dilate_m: float = 1.0, shifts: dict[str, tuple[float, float]] | None = None,
                  ) -> Callable[[str], tuple[np.ndarray, TileGeometry]]:
    """Wrap a tile reader so that it returns the tile with every building masked out."""
    by_tile = {t: g for t, g in buildings.groupby("tile_id", sort=False)}

    def read(tile_id: str) -> tuple[np.ndarray, TileGeometry]:
        image, geom = read_tile(tile_id)
        g = by_tile.get(tile_id)
        if g is None or g.empty:
            return image, geom
        shift = (shifts or {}).get(tile_id, (0.0, 0.0))
        return mask_tile(image, building_pixel_mask(g, geom, dilate_m, shift)), geom

    return read
