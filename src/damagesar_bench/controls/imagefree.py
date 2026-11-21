"""Image-free control (FR9, RQ4): predict damage from building geometry alone, without any image.

Four features per building, all from the building table (no pixels, no labels):

* ``f0000``  log building area (m²);
* ``f0001``  log(1 + number of buildings in the same tile);
* ``f0002``  log(1 + number of other buildings whose bounding-box centre lies within ``radius_m``);
* ``f0003``  log distance (m) to the nearest other building in the tile (``radius_m`` × 4 if alone).

The table has the same shape as an encoder's building features (encoder ``imagefree``, pooling ``geometry``), so
it goes through the same probes, folds and statistics. A model whose AUROC does not clearly exceed this
control's is not credited with recognising damage. Every building in a tile (also those excluded by the
no-data rule) counts as a neighbour, because they are part of the scene's layout; only kept buildings get a row.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..config import EventsConfig
from ..contracts import feature_name, validate_building_features

ENCODER = "imagefree"
POOLING = "geometry"
LABEL_COLS = ("event", "tile_id", "instance_id", "role", "damaged", "labels_locked", "area_m2")


class ControlError(ValueError):
    pass


def _tile_features(g: pd.DataFrame, radius_m: float) -> np.ndarray:
    res = g["res_m"].to_numpy(float)
    cx = (g["bbox_x"].to_numpy(float) + g["bbox_w"].to_numpy(float) / 2) * res
    cy = (g["bbox_y"].to_numpy(float) + g["bbox_h"].to_numpy(float) / 2) * res
    d = np.hypot(cx[:, None] - cx[None, :], cy[:, None] - cy[None, :])
    np.fill_diagonal(d, np.inf)
    within = (d <= radius_m).sum(axis=1)
    nearest = d.min(axis=1) if len(g) > 1 else np.full(len(g), np.inf)
    nearest = np.where(np.isfinite(nearest), nearest, radius_m * 4)
    return np.column_stack([
        np.log(g["area_m2"].to_numpy(float)),
        np.full(len(g), np.log1p(len(g))),
        np.log1p(within),
        np.log(np.maximum(nearest, 0.1)),
    ])


def image_free_features(buildings: pd.DataFrame, events: EventsConfig, today: date | None = None,
                        radius_m: float = 50.0) -> pd.DataFrame:
    """Building-features table of the image-free control for every kept building."""
    need = {"tile_id", "excluded_reason", "res_m", "bbox_x", "bbox_y", "bbox_w", "bbox_h", *LABEL_COLS}
    missing = need - set(buildings.columns)
    if missing:
        raise ControlError(f"building table is missing {sorted(missing)}")
    if not radius_m > 0:
        raise ControlError("radius_m must be > 0")
    parts = []
    for _, g in buildings.groupby("tile_id", sort=True):
        g = g.dropna(subset=["res_m", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "area_m2"])
        g = g[g["area_m2"] > 0]
        if g.empty:
            continue
        x = _tile_features(g, radius_m)
        keep = g["excluded_reason"].isna().to_numpy()
        if not keep.any():
            continue
        rows = g.loc[keep, list(LABEL_COLS)].reset_index(drop=True)
        rows["encoder"] = ENCODER
        rows["pooling"] = POOLING
        feats = pd.DataFrame(x[keep].astype(np.float32), columns=[feature_name(i) for i in range(x.shape[1])])
        parts.append(pd.concat([rows, feats], axis=1))
    if not parts:
        raise ControlError("no kept buildings with geometry")
    df = pd.concat(parts, ignore_index=True)
    df["damaged"] = df["damaged"].astype("boolean")
    df["labels_locked"] = df["labels_locked"].astype(bool)
    validate_building_features(df, events, today)
    return df
