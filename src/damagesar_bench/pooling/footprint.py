"""Building pooling (FR6): token features -> one vector per building.

Methods (pre-registered names):
* ``outline``      the footprint polygon as drawn (BRIGHT instance labels);
* ``dilated:<m>``  the footprint grown by <m> metres on the ground, e.g. ``dilated:3``
                   (FYP Master §7: 3 m). Projected pooling (C2) is added later.
* ``ring:<m>``     neighbourhood control (RQ4): the area within <m> metres of the footprint, **minus every
                   token that contains a building pixel**, so the vector carries the surroundings but none
                   of the building itself. If no such token is left (the ring stays inside the building's
                   tokens), the 8 tokens around the building's centre token are used and the row is flagged
                   (``pool_pixels`` = 0).

How a building's vector is formed (ADR 0001 leaning: area-weighted pooling):
1. Rasterise the polygon on the tile's native pixel grid (pixel centres, same rule as the
   building table). For ``dilated``, pixels whose centre lies within <m> metres of a footprint
   pixel centre are added; distances use the true ground pixel size on each axis, so
   non-square pixels (EPSG:4326 tiles) are handled.
2. Map every pixel to the token that contains its centre: token row = floor((r + 0.5) * h / H),
   where the encoder's h x w token grid covers the whole H x W tile.
3. Vector = mean of the token features weighted by the number of building pixels in each token.
4. A footprint with no pixel centre inside (smaller than a pixel) uses the token that contains
   the centre of its bounding box; such rows are flagged (``pool_pixels`` = 0).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

from ..config import EventsConfig
from ..contracts import feature_name, validate_building_features
from ..data.buildings import rasterize_window

LABEL_COLS = ("event", "tile_id", "instance_id", "role", "damaged", "labels_locked", "area_m2")


class PoolingError(ValueError):
    pass


@dataclass(frozen=True)
class TileGeometry:
    height: int          # native pixels
    width: int
    px_x_m: float        # ground size of one pixel (metres), per axis
    px_y_m: float

    def __post_init__(self):
        if self.height <= 0 or self.width <= 0:
            raise PoolingError("tile size must be positive")
        if not (self.px_x_m > 0 and self.px_y_m > 0):
            raise PoolingError("pixel size must be positive")


@dataclass(frozen=True)
class Method:
    kind: str            # "outline" | "dilated" | "ring"
    metres: float = 0.0

    @classmethod
    def parse(cls, text: str) -> "Method":
        if text == "outline":
            return cls("outline")
        if text.startswith("dilated:"):
            m = float(text.split(":", 1)[1])
            if not m > 0:
                raise PoolingError("dilation must be > 0 metres")
            return cls("dilated", m)
        if text.startswith("ring:"):
            m = float(text.split(":", 1)[1])
            if not m > 0:
                raise PoolingError("ring width must be > 0 metres")
            return cls("ring", m)
        raise PoolingError(f"unknown pooling {text!r} (use 'outline', 'dilated:<metres>' or 'ring:<metres>')")

    @property
    def name(self) -> str:
        if self.kind == "outline":
            return "outline"
        return f"{self.kind}{self.metres:g}m"


def building_mask(polys: list[list[float]], geom: TileGeometry, dilate_m: float = 0.0) -> tuple[np.ndarray, int, int]:
    """Footprint mask (optionally dilated in metres) inside a window; returns (mask, row0, col0)."""
    if dilate_m <= 0:
        return rasterize_window(polys, geom.height, geom.width)
    ry = int(math.ceil(dilate_m / geom.px_y_m)) + 1
    rx = int(math.ceil(dilate_m / geom.px_x_m)) + 1
    core, r0, c0 = rasterize_window(polys, geom.height, geom.width)
    if core.size == 0 or not core.any():
        return core, r0, c0
    top, left = max(r0 - ry, 0), max(c0 - rx, 0)
    bottom = min(r0 + core.shape[0] + ry, geom.height)
    right = min(c0 + core.shape[1] + rx, geom.width)
    big = np.zeros((bottom - top, right - left), dtype=bool)
    big[r0 - top:r0 - top + core.shape[0], c0 - left:c0 - left + core.shape[1]] = core
    dist = distance_transform_edt(~big, sampling=(geom.px_y_m, geom.px_x_m))
    return dist <= dilate_m + 1e-9, top, left


def _bbox_centre(polys) -> tuple[float, float]:
    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def _token_index(rr: np.ndarray, cc: np.ndarray, geom: TileGeometry, h: int, w: int) -> np.ndarray:
    tr = np.floor((rr + 0.5) * h / geom.height).astype(int)
    tc = np.floor((cc + 0.5) * w / geom.width).astype(int)
    return tr * w + tc


def _centre_token(polys, geom: TileGeometry, h: int, w: int) -> tuple[int, int]:
    x, y = _bbox_centre(polys)
    tr = min(max(int(math.floor(y * h / geom.height)), 0), h - 1)
    tc = min(max(int(math.floor(x * w / geom.width)), 0), w - 1)
    return tr, tc


def _weighted_mean(tokens: np.ndarray, flat: np.ndarray) -> np.ndarray:
    h, w, _ = tokens.shape
    idx, counts = np.unique(flat, return_counts=True)
    vecs = tokens.reshape(h * w, -1)[idx].astype(np.float64)
    return (vecs * counts[:, None]).sum(axis=0) / counts.sum()


def pool_building(tokens: np.ndarray, polys: list[list[float]], geom: TileGeometry,
                  method: Method) -> tuple[np.ndarray, int]:
    """Area-weighted token mean for one building; returns (vector, pooled pixel count)."""
    if tokens.ndim != 3:
        raise PoolingError("tokens must have shape (rows, cols, dim)")
    h, w, _ = tokens.shape
    if method.kind == "ring":
        return _pool_ring(tokens, polys, geom, method.metres)
    mask, r0, c0 = building_mask(polys, geom, method.metres if method.kind == "dilated" else 0.0)
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        tr, tc = _centre_token(polys, geom, h, w)
        return tokens[tr, tc].astype(np.float64), 0
    return _weighted_mean(tokens, _token_index(rr + r0, cc + c0, geom, h, w)), int(rr.size)


def _pool_ring(tokens: np.ndarray, polys, geom: TileGeometry, metres: float) -> tuple[np.ndarray, int]:
    """Surroundings within ``metres`` of the footprint, excluding every token that holds a building pixel."""
    h, w, _ = tokens.shape
    core, r0, c0 = building_mask(polys, geom, 0.0)
    rr, cc = np.nonzero(core)
    if rr.size:
        building_tokens = set(np.unique(_token_index(rr + r0, cc + c0, geom, h, w)).tolist())
    else:
        tr, tc = _centre_token(polys, geom, h, w)
        building_tokens = {tr * w + tc}
    big, top, left = building_mask(polys, geom, metres)
    br, bc = np.nonzero(big)
    flat = _token_index(br + top, bc + left, geom, h, w) if br.size else np.empty(0, dtype=int)
    keep = ~np.isin(flat, list(building_tokens))
    if keep.any():
        return _weighted_mean(tokens, flat[keep]), int(keep.sum())
    # Fallback: the 8 tokens around the building's centre token (equal weights).
    tr, tc = _centre_token(polys, geom, h, w)
    nb = [(r, c) for r in range(tr - 1, tr + 2) for c in range(tc - 1, tc + 2)
          if (r, c) != (tr, tc) and 0 <= r < h and 0 <= c < w and r * w + c not in building_tokens]
    if not nb:
        nb = [(r, c) for r in range(tr - 1, tr + 2) for c in range(tc - 1, tc + 2)
              if (r, c) != (tr, tc) and 0 <= r < h and 0 <= c < w]
    return np.mean([tokens[r, c].astype(np.float64) for r, c in nb], axis=0), 0


def pool_features(
    buildings: pd.DataFrame,
    tokens_for_tile: Callable[[str], np.ndarray],
    geometry_for_tile: Callable[[str], TileGeometry],
    encoder: str,
    method: str,
    events: EventsConfig,
    today: date | None = None,
) -> pd.DataFrame:
    """Building-features table (ADR 0002) for one encoder and one pooling method.

    ``buildings`` is the building table; excluded buildings (no-data rule) are skipped.
    ``tokens_for_tile(tile_id)`` returns the encoder's (rows, cols, dim) token grid.
    """
    m = Method.parse(method)
    kept = buildings[buildings["excluded_reason"].isna()]
    if kept.empty:
        raise PoolingError("no buildings left after the no-data rule")
    rows, vecs, dim = [], [], None
    for tile_id, g in kept.groupby("tile_id", sort=True):
        tokens = np.asarray(tokens_for_tile(tile_id))
        geom = geometry_for_tile(tile_id)
        if tokens.ndim != 3:
            raise PoolingError(f"{tile_id}: tokens must have shape (rows, cols, dim)")
        if dim is None:
            dim = tokens.shape[2]
        elif tokens.shape[2] != dim:
            raise PoolingError(f"{tile_id}: feature size {tokens.shape[2]} != {dim}")
        for _, b in g.iterrows():
            vec, npix = pool_building(tokens, json.loads(b["segmentation_px"]), geom, m)
            rows.append({**{c: b[c] for c in LABEL_COLS}, "encoder": encoder, "pooling": m.name,
                         "pool_pixels": npix})
            vecs.append(vec)
    df = pd.DataFrame(rows)
    feats = pd.DataFrame(np.vstack(vecs).astype(np.float32), columns=[feature_name(i) for i in range(dim)])
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    df["damaged"] = df["damaged"].astype("boolean")
    df["labels_locked"] = df["labels_locked"].astype(bool)
    validate_building_features(df, events, today)
    return df
