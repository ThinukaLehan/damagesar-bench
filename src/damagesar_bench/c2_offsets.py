"""C2 groundwork: how far, and in which direction, is the SAR signal from the optical footprints? (per tile)

For each tile, the footprints of its buildings are rasterised into one mask and cross-correlated with the SAR
image (both block-averaged by ``factor`` and made zero-mean; no-data pixels count as neither). The shift with the
highest correlation within ``max_shift_m`` is the tile's offset: the SAR signal sits ``(dy_m, dx_m)`` metres from
the footprints (+x = east/right, +y = south/down in image rows).

Two scores say whether the shift is real:

* ``peak_z``  how far the best correlation stands above the rest of the search window (in SDs);
* ``gain_z``  how much the best shift improves on no shift, in the same SD units.

The permutation null (``null_gains``) repeats this with the SAR image of *other* tiles of the same event: a real
offset should improve the match much more for the true pair than for a mismatched one. This mirrors the 31 Oct
audit, now in the package. Only footprints and SAR are read; damage labels are never used.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .data.buildings import rasterize_window


class OffsetError(ValueError):
    pass


@dataclass(frozen=True)
class TileOffset:
    tile_id: str
    n_buildings: int
    dy_m: float
    dx_m: float
    magnitude_m: float
    angle_deg: float      # direction of the shift in image coordinates, 0 = east (+x), 90 = south (+y)
    peak_z: float
    gain_z: float


def footprint_mask(polygon_sets: list[list[list[float]]], height: int, width: int) -> np.ndarray:
    """One boolean mask of all footprints in a tile (each item is one building's list of polygons)."""
    mask = np.zeros((height, width), dtype=bool)
    for polys in polygon_sets:
        if not polys:
            continue
        m, r0, c0 = rasterize_window(polys, height, width)
        if m.size:
            mask[r0:r0 + m.shape[0], c0:c0 + m.shape[1]] |= m
    return mask


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average over non-overlapping ``factor`` x ``factor`` blocks (edges that do not fill a block are dropped)."""
    if factor < 1:
        raise OffsetError("factor must be >= 1")
    h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
    return a[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def correlation_surface(mask: np.ndarray, sar: np.ndarray, valid: np.ndarray, max_shift: int) -> np.ndarray:
    """corr[dy, dx] = sum_y,x m0(y, x) * s0(y + dy, x + dx) for |dy|, |dx| <= max_shift (zero padded, no wrap).

    ``m0`` and ``s0`` are the mask and SAR made zero-mean over the valid pixels and set to 0 elsewhere.
    Returns a (2 max_shift + 1) square array indexed by ``dy + max_shift, dx + max_shift``.
    """
    if mask.shape != sar.shape or valid.shape != sar.shape:
        raise OffsetError("mask, sar and valid must have the same shape")
    if max_shift < 1:
        raise OffsetError("max_shift must be >= 1 block")
    m = mask.astype(np.float64)
    s = sar.astype(np.float64)
    if valid.sum() < 4:
        raise OffsetError("too few valid pixels")
    m0 = np.where(valid, m - m[valid].mean(), 0.0)
    s0 = np.where(valid, s - s[valid].mean(), 0.0)
    h, w = m.shape
    ph, pw = h + max_shift, w + max_shift
    fm = np.fft.rfft2(m0, s=(ph, pw))
    fs = np.fft.rfft2(s0, s=(ph, pw))
    full = np.fft.irfft2(np.conj(fm) * fs, s=(ph, pw))     # full[dy, dx] for dy, dx >= 0; negatives wrap to the end
    idx_y = np.arange(-max_shift, max_shift + 1) % ph
    idx_x = np.arange(-max_shift, max_shift + 1) % pw
    return full[np.ix_(idx_y, idx_x)]


def _scores(surface: np.ndarray, max_shift: int) -> tuple[int, int, float, float]:
    flat = surface.ravel()
    k = int(np.argmax(flat))
    by, bx = divmod(k, surface.shape[1])
    sd = float(flat.std())
    if sd == 0:
        return 0, 0, 0.0, 0.0
    peak_z = float((flat[k] - flat.mean()) / sd)
    gain_z = float((flat[k] - surface[max_shift, max_shift]) / sd)
    return by - max_shift, bx - max_shift, peak_z, gain_z


def tile_offset(tile_id: str, polygon_sets: list[list[list[float]]], sar: np.ndarray, px_y_m: float, px_x_m: float,
                factor: int = 4, max_shift_m: float = 40.0) -> TileOffset:
    """Offset of the SAR signal from the tile's footprints (``sar`` in [0, 1]; 0 = no-data)."""
    if not polygon_sets:
        raise OffsetError(f"{tile_id}: no footprints")
    h, w = sar.shape
    mask = block_mean(footprint_mask(polygon_sets, h, w).astype(np.float32), factor)
    s = block_mean(sar, factor)
    valid = block_mean((sar > 0).astype(np.float32), factor) >= 0.5
    step = factor * min(px_y_m, px_x_m)
    max_shift = max(1, int(math.ceil(max_shift_m / step)))
    surf = correlation_surface(mask, s, valid, max_shift)
    dy, dx, peak_z, gain_z = _scores(surf, max_shift)
    dy_m, dx_m = dy * factor * px_y_m, dx * factor * px_x_m
    return TileOffset(tile_id, len(polygon_sets), float(dy_m), float(dx_m), float(math.hypot(dy_m, dx_m)),
                      float(math.degrees(math.atan2(dy_m, dx_m)) % 360.0), peak_z, gain_z)


def null_gains(polygon_sets: list[list[list[float]]], other_sars: list[np.ndarray], px_y_m: float, px_x_m: float,
               factor: int = 4, max_shift_m: float = 40.0) -> list[float]:
    """``gain_z`` of this tile's footprints against the SAR of other tiles (the permutation null)."""
    out = []
    for o in other_sars:
        try:
            out.append(tile_offset("null", polygon_sets, o, px_y_m, px_x_m, factor, max_shift_m).gain_z)
        except OffsetError:
            continue
    return out


def mean_resultant_length(angles_deg: np.ndarray) -> float:
    """1 = every shift points the same way; near 0 = directions spread evenly (circular statistics)."""
    a = np.radians(np.asarray(angles_deg, dtype=float))
    if a.size == 0:
        return float("nan")
    return float(np.hypot(np.cos(a).mean(), np.sin(a).mean()))


def event_summary(offsets: pd.DataFrame, peak_z_min: float = 3.0) -> pd.DataFrame:
    """Per event: tiles, median offset (m), share of confident tiles, share beating every null, direction
    concentration (R) and mean direction of the confident tiles.

    ``offsets`` needs columns event, magnitude_m, angle_deg, peak_z and (optional) beats_null.
    """
    rows = []
    for ev, g in offsets.groupby("event", sort=True):
        conf = g[g["peak_z"] >= peak_z_min]
        a = np.radians(conf["angle_deg"].to_numpy(float))
        mean_dir = float(math.degrees(math.atan2(np.sin(a).mean(), np.cos(a).mean())) % 360.0) if a.size else float("nan")
        rows.append({
            "event": ev, "tiles": len(g),
            "median_offset_m": float(g["magnitude_m"].median()),
            "median_offset_confident_m": float(conf["magnitude_m"].median()) if len(conf) else float("nan"),
            "share_confident": float(len(conf) / len(g)),
            "share_beats_null": float(g["beats_null"].mean()) if "beats_null" in g else float("nan"),
            "direction_R": mean_resultant_length(conf["angle_deg"].to_numpy(float)),
            "mean_direction_deg": mean_dir,
        })
    return pd.DataFrame(rows)


def measure_offsets(buildings: pd.DataFrame, read_tile, events: list[str], factor: int = 4,
                    max_shift_m: float = 40.0, n_null: int = 3, seed: int = 0,
                    log=lambda s: None) -> pd.DataFrame:
    """Offsets for every tile of ``events`` with at least one kept building, with a permutation null.

    ``read_tile(tile_id)`` returns ``(sar in [0, 1], TileGeometry)``. Each tile is compared with ``n_null``
    other tiles of the same event; ``beats_null`` is True when its gain exceeds all of theirs.
    """
    rng = np.random.default_rng(seed)
    b = buildings[buildings["event"].isin(events) & buildings["excluded_reason"].isna()]
    if b.empty:
        raise OffsetError(f"no kept buildings in {events}")
    rows = []
    for ev, ge in b.groupby("event", sort=True):
        tiles = sorted(ge["tile_id"].unique())
        cache: dict[str, tuple] = {}

        def get(t):
            if t not in cache:
                cache[t] = read_tile(t)
            return cache[t]

        for t in tiles:
            polys = [json.loads(s) for s in ge.loc[ge["tile_id"] == t, "segmentation_px"]]
            sar, geom = get(t)
            try:
                off = tile_offset(t, polys, sar, geom.px_y_m, geom.px_x_m, factor, max_shift_m)
            except OffsetError as ex:
                log(f"skip {t}: {ex}")
                continue
            others = [o for o in tiles if o != t]
            pick = rng.choice(others, size=min(n_null, len(others)), replace=False) if others else []
            nulls = null_gains(polys, [get(o)[0] for o in pick if get(o)[0].shape == sar.shape],
                               geom.px_y_m, geom.px_x_m, factor, max_shift_m)
            rows.append({"event": ev, **asdict(off), "null_gain_max": max(nulls) if nulls else float("nan"),
                         "beats_null": bool(nulls) and off.gain_z > max(nulls)})
        log(f"{ev}: {len(tiles)} tiles")
        cache.clear()
    return pd.DataFrame(rows)
