"""``bench extract``: tiles -> encoder tokens -> pooled building features (FR5, FR6, NFR2, NFR3).

Design (ADR 0003):
* Token grids are **not** stored (a 64 x 64 x 768 grid is ~6 MB per tile, ~26 GB for BRIGHT).
  Each tile is encoded once and immediately pooled with every requested method; only the
  building vectors are kept.
* One Parquet shard per tile, written atomically (temporary file, then rename). A rerun skips
  tiles whose shard exists, so a Colab disconnect loses at most one tile (kill-and-resume).
* ``manifest.json`` records encoder, pooling methods, resolution and patch size; resuming
  with different settings stops instead of mixing incompatible shards.
* Before the freeze, confirmatory tiles are skipped unless asked for explicitly (SAR only, no
  labels, but it keeps every confirmatory result untouched until the plan is registered).
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

from .. import __version__
from ..config import EventsConfig
from ..contracts import validate_building_features
from ..data.events import labels_unlocked
from ..pooling.footprint import Method, TileGeometry, pool_features
from .base import Encoder, EncoderError

MANIFEST = "manifest.json"


def resample(image: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize with pixel centres aligned (output centre i+0.5 <- input (i+0.5)*H/out_h)."""
    h, w = image.shape
    if (out_h, out_w) == (h, w):
        return image.astype(np.float32, copy=False)
    rr = (np.arange(out_h) + 0.5) * h / out_h - 0.5
    cc = (np.arange(out_w) + 0.5) * w / out_w - 0.5
    grid = np.meshgrid(rr, cc, indexing="ij")
    return map_coordinates(image.astype(np.float32), grid, order=1, mode="nearest").astype(np.float32)


def encoder_input_size(geom: TileGeometry, patch_px: int, target_res_m: float | None) -> tuple[int, int]:
    """Encoder input size: tile at ``target_res_m`` (or native), rounded to whole patches."""
    if target_res_m is None:
        h, w = geom.height, geom.width
    else:
        h = geom.height * geom.px_y_m / target_res_m
        w = geom.width * geom.px_x_m / target_res_m
    return max(1, round(h / patch_px)) * patch_px, max(1, round(w / patch_px)) * patch_px


def read_bright_tile(root: str | Path, tile_id: str) -> tuple[np.ndarray, TileGeometry]:
    """8-bit BRIGHT SAR tile -> float32 in [0, 1] and its true ground pixel size."""
    import rasterio

    from ..data.buildings import pixel_size_m

    path = Path(root) / "post-event" / f"{tile_id}_post_disaster.tif"
    with rasterio.open(path) as ds:
        if ds.count != 1:
            raise EncoderError(f"{tile_id}: expected 1 SAR band, found {ds.count}")
        arr = ds.read(1)
        px_x, px_y = pixel_size_m(ds.transform, ds.crs, ds.bounds)
    if arr.dtype != np.uint8:
        raise EncoderError(f"{tile_id}: expected 8-bit SAR, found {arr.dtype}")
    return arr.astype(np.float32) / 255.0, TileGeometry(arr.shape[0], arr.shape[1], px_x, px_y)


@dataclass
class ExtractReport:
    tiles_total: int = 0
    tiles_done_before: int = 0
    tiles_encoded: int = 0
    tiles_skipped_confirmatory: int = 0
    seconds: float = 0.0
    shards: list[Path] = field(default_factory=list)


def _check_manifest(out: Path, manifest: dict) -> None:
    p = out / MANIFEST
    if p.exists():
        old = json.loads(p.read_text())
        keys = ("encoder", "methods", "target_res_m", "patch_px", "dim")
        diff = {k: (old.get(k), manifest[k]) for k in keys if old.get(k) != manifest[k]}
        if diff:
            raise EncoderError(f"{out} holds shards made with other settings {diff}; use a new folder")
    else:
        out.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(manifest, indent=2) + "\n")


def extract(
    buildings: pd.DataFrame,
    read_tile: Callable[[str], tuple[np.ndarray, TileGeometry]],
    encoder: Encoder,
    methods: list[str],
    out_dir: str | Path,
    events: EventsConfig,
    today: date | None = None,
    target_res_m: float | None = None,
    include_confirmatory: bool = False,
    log: Callable[[str], None] = lambda s: None,
) -> tuple[pd.DataFrame, ExtractReport]:
    out = Path(out_dir)
    parsed = [Method.parse(m) for m in methods]
    if not parsed:
        raise EncoderError("give at least one pooling method")
    _check_manifest(out, {
        "encoder": encoder.name, "methods": [m.name for m in parsed], "target_res_m": target_res_m,
        "patch_px": encoder.patch_px, "dim": encoder.dim, "tool_version": __version__,
    })
    shard_dir = out / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    kept = buildings[buildings["excluded_reason"].isna()]
    rep = ExtractReport()
    if not include_confirmatory and not labels_unlocked(events, today):
        conf = kept["role"] == "confirmatory"
        rep.tiles_skipped_confirmatory = kept.loc[conf, "tile_id"].nunique()
        kept = kept[~conf]
    tiles = sorted(kept["tile_id"].unique())
    rep.tiles_total = len(tiles)
    t0 = time.time()
    for i, tile_id in enumerate(tiles, 1):
        shard = shard_dir / f"{tile_id}.parquet"
        rep.shards.append(shard)
        if shard.exists():
            rep.tiles_done_before += 1
            continue
        image, geom = read_tile(tile_id)
        size = encoder_input_size(geom, encoder.patch_px, target_res_m)
        tokens = encoder.encode(resample(image, *size))
        tb = kept[kept["tile_id"] == tile_id]
        parts = [pool_features(tb, lambda _t: tokens, lambda _t: geom, encoder.name, m, events, today)
                 for m in methods]
        tmp = shard.with_suffix(".parquet.tmp")
        pd.concat(parts, ignore_index=True).to_parquet(tmp, index=False)
        os.replace(tmp, shard)
        rep.tiles_encoded += 1
        if rep.tiles_encoded % 50 == 0:
            el = time.time() - t0
            log(f"{i}/{len(tiles)} tiles, {rep.tiles_encoded / el:.2f} tiles/s")
    rep.seconds = time.time() - t0
    if not rep.shards:
        raise EncoderError("no tiles to extract")
    df = pd.concat([pd.read_parquet(s) for s in rep.shards], ignore_index=True)
    df["damaged"] = df["damaged"].astype("boolean")
    validate_building_features(df, events, today)
    return df, rep


def tiles_per_second(rep: ExtractReport) -> float:
    return rep.tiles_encoded / rep.seconds if rep.seconds > 0 and rep.tiles_encoded else math.nan
