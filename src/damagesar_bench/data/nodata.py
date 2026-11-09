"""No-data rule (FR2).

- A tile is excluded if at least ``tile_exclude_frac`` (95%) of its SAR pixels are no-data.
- In the remaining tiles, a building is dropped if at least ``building_exclude_frac`` (50%) of its
  footprint pixels are no-data.

SAR value 0 is treated as no-data in every event, because Türkiye's files carry no no-data tag.
"""

from __future__ import annotations

import numpy as np

from damagesar_bench.config import NoDataRule


def nodata_mask(sar: np.ndarray, rule: NoDataRule) -> np.ndarray:
    """Boolean mask, True where every band equals the no-data value."""
    arr = np.asarray(sar)
    if arr.ndim == 3:  # (bands, H, W)
        return np.all(arr == rule.sar_value, axis=0)
    if arr.ndim == 2:
        return arr == rule.sar_value
    raise ValueError(f"expected a 2-D or 3-D SAR array, got shape {arr.shape}")


def nodata_fraction(mask: np.ndarray, region: np.ndarray | None = None) -> float:
    """Share of no-data pixels, over the whole tile or inside ``region`` (a boolean footprint)."""
    if region is None:
        return float(mask.mean()) if mask.size else 1.0
    region = np.asarray(region, dtype=bool)
    if region.shape != mask.shape:
        raise ValueError("region and mask must have the same shape")
    n = int(region.sum())
    if n == 0:
        return 1.0  # an empty footprint is unusable
    return float(mask[region].mean())


def tile_is_excluded(mask: np.ndarray, rule: NoDataRule) -> bool:
    return nodata_fraction(mask) >= rule.tile_exclude_frac


def building_is_excluded(mask: np.ndarray, footprint: np.ndarray, rule: NoDataRule) -> bool:
    return nodata_fraction(mask, footprint) >= rule.building_exclude_frac
