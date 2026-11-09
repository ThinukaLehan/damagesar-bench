"""Building table (``buildings.parquet``): one row per labelled building instance.

Source: BRIGHT instance labels, one COCO-style JSON per tile
(``target_instance_level/<tile>_instance_damage.json``) with ``images`` (one entry, tile size and
georeference), ``annotations`` (one per building: ``category_id``, ``segmentation`` polygons in pixel
coordinates, ``area``, ``bbox``) and ``categories`` (id -> name).

Rules applied here (pre-registration §3):
- The damage class comes from the JSON category **name**, mapped with ``CATEGORY_MAP``. Names that are not in
  the map stop the run: nothing is guessed.
- Confirmatory events before the freeze: geometry only. Their category is read from the file (it is in the same
  record) but is discarded at once and never stored, so the table holds no confirmatory label information.
- No-data rule: a building is excluded if >= 50% of its footprint pixels are SAR no-data; the whole tile is
  excluded if >= 95% of its pixels are no-data.
- Polygons are rasterised in pixel space (pixel centres), the same way the label rasters are drawn.
"""

from __future__ import annotations

import io
import json
import math
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from damagesar_bench.config import EventsConfig, NoDataRule
from damagesar_bench.data.events import event_key_from_filename, labels_unlocked

# Damage classes used everywhere in the toolkit.
INTACT, DAMAGED, DESTROYED = 0, 1, 2
CLASS_NAMES = {INTACT: "intact", DAMAGED: "damaged", DESTROYED: "destroyed"}

# JSON category name (lower-case, spaces/underscores -> hyphens) -> damage class.
CATEGORY_MAP = {
    "intact": INTACT,
    "no-damage": INTACT,
    "undamaged": INTACT,
    "damaged": DAMAGED,
    "minor-damage": DAMAGED,
    "major-damage": DAMAGED,
    "destroyed": DESTROYED,
}

INSTANCE_SUFFIX = "_instance_damage.json"
# Reported COCO area vs our pixel count: allowed relative difference before we flag the building.
AREA_TOLERANCE = 0.25


class BuildingTableError(ValueError):
    """Raised when an instance file does not match the expected format."""


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "-").replace("_", "-")


def category_lookup(categories: list[dict]) -> dict[int, int]:
    """Map category ids in one file to damage classes, refusing unknown names."""
    lookup = {}
    for c in categories:
        key = _norm(str(c.get("name", "")))
        if key not in CATEGORY_MAP:
            raise BuildingTableError(
                f"unknown category name {c.get('name')!r} (id {c.get('id')}); add it to CATEGORY_MAP after checking"
            )
        lookup[int(c["id"])] = CATEGORY_MAP[key]
    return lookup


def tile_id_from_instance_name(name: str) -> str:
    base = Path(name).name
    if not base.endswith(INSTANCE_SUFFIX):
        raise BuildingTableError(f"{name!r} is not an instance-label file ({INSTANCE_SUFFIX})")
    return base[: -len(INSTANCE_SUFFIX)]


def polygon_area(xs: list[float], ys: list[float]) -> float:
    """Shoelace area of one ring."""
    n = len(xs)
    return 0.5 * abs(sum(xs[i] * ys[i - 1] - xs[i - 1] * ys[i] for i in range(n)))


def _shapes(polys: list[list[float]]) -> list:
    shapes = []
    for p in polys:
        ring = [(float(p[i]), float(p[i + 1])) for i in range(0, len(p), 2)]
        if len(ring) >= 3:
            if ring[0] != ring[-1]:
                ring.append(ring[0])
            shapes.append(({"type": "Polygon", "coordinates": [ring]}, 1))
    return shapes


def rasterize_polygons(polys: list[list[float]], height: int, width: int) -> np.ndarray:
    """Boolean full-tile mask of the polygons (COCO flat [x1, y1, x2, y2, ...] in pixel coordinates).

    A pixel is inside if its centre is inside (rasterio's default), as for the label rasters.
    """
    from rasterio import features
    from rasterio.transform import Affine

    shapes = _shapes(polys)
    if not shapes:
        return np.zeros((height, width), dtype=bool)
    return features.rasterize(shapes, out_shape=(height, width), transform=Affine.identity(), fill=0,
                              dtype="uint8").astype(bool)


def rasterize_window(polys: list[list[float]], height: int, width: int) -> tuple[np.ndarray, int, int]:
    """Same pixels as :func:`rasterize_polygons`, computed only inside the polygons' bounding box.

    Returns ``(mask, row0, col0)``; ``mask`` covers rows ``row0:row0+h`` and columns ``col0:col0+w`` of the tile.
    """
    from rasterio import features
    from rasterio.transform import Affine

    xs = [v for p in polys for v in p[0::2]]
    ys = [v for p in polys for v in p[1::2]]
    c0, r0 = max(int(math.floor(min(xs))), 0), max(int(math.floor(min(ys))), 0)
    c1, r1 = min(int(math.ceil(max(xs))), width), min(int(math.ceil(max(ys))), height)
    shapes = _shapes(polys)
    if not shapes or c1 <= c0 or r1 <= r0:
        return np.zeros((0, 0), dtype=bool), r0, c0
    mask = features.rasterize(shapes, out_shape=(r1 - r0, c1 - c0), transform=Affine.translation(c0, r0),
                              fill=0, dtype="uint8").astype(bool)
    return mask, r0, c0


EARTH_RADIUS_WEB_MERCATOR = 6_378_137.0


def pixel_size_m(transform, crs, bounds) -> tuple[float, float]:
    """Ground size of one pixel in metres, ``(x, y)``, at the tile's centre.

    - Projected CRS in metres (UTM etc.): the geotransform values.
    - Geographic CRS (EPSG:4326, degrees): x scaled by cos(latitude); y by metres per degree of latitude.
      The two differ, so pixels square in degrees are not square on the ground.
    - Web Mercator (EPSG:3857): its "metres" are stretched by 1/cos(latitude); both sides are scaled back.
    """
    ax, ay = abs(transform.a), abs(transform.e)
    if crs is not None and crs.is_geographic:
        lat = math.radians((bounds.top + bounds.bottom) / 2)
        m_per_deg_lat = 111_132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
        m_per_deg_lon = 111_412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
        return ax * m_per_deg_lon, ay * m_per_deg_lat
    if crs is not None and crs.to_epsg() == 3857:
        y_mid = (bounds.top + bounds.bottom) / 2
        lat = math.atan(math.sinh(y_mid / EARTH_RADIUS_WEB_MERCATOR))
        return ax * math.cos(lat), ay * math.cos(lat)
    return ax, ay


def metres_per_pixel(transform, crs, bounds) -> float:
    """Nominal ground resolution: sqrt of the ground pixel area (equals the side for square pixels)."""
    rx, ry = pixel_size_m(transform, crs, bounds)
    return math.sqrt(rx * ry)


@dataclass
class TileRaster:
    """Optional rasters for one tile: SAR (for no-data) and label raster (for the agreement check)."""
    sar_path: Path | None = None
    label_path: Path | None = None


@dataclass
class BuildStats:
    files: int = 0
    tiles_excluded_nodata: int = 0
    buildings: int = 0
    buildings_excluded_nodata: int = 0
    area_flags: int = 0
    sample_id_mismatches: int = 0
    events: dict = field(default_factory=dict)


def _iter_instance_files(source: Path) -> Iterator[tuple[str, bytes]]:
    """Yield (name, bytes) from a folder of JSON files or a zip archive."""
    if source.is_file() and zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as z:
            for n in sorted(z.namelist()):
                if n.endswith(INSTANCE_SUFFIX):
                    yield n, z.read(n)
    elif source.is_dir():
        for p in sorted(source.rglob(f"*{INSTANCE_SUFFIX}")):
            yield p.name, p.read_bytes()
    else:
        raise FileNotFoundError(f"instance labels not found: {source}")


def _tile_rows(name: str, raw: bytes, events: EventsConfig, unlocked: bool, rule: NoDataRule,
               rasters: TileRaster | None, target_res_m: float, patch_px: int, stats: BuildStats) -> list[dict]:
    tile_id = tile_id_from_instance_name(name)
    event = event_key_from_filename(tile_id)
    if event not in events.events:
        raise BuildingTableError(f"event {event!r} (from {name}) is not in events.yaml")
    info = events.events[event]
    keep_labels = info.role != "confirmatory" or unlocked

    doc = json.loads(raw)
    images = doc.get("images") or []
    if len(images) != 1:
        raise BuildingTableError(f"{name}: expected exactly one image entry, found {len(images)}")
    im = images[0]
    width, height = int(im["width"]), int(im["height"])
    if str(im.get("sample_id", tile_id)) != tile_id:
        stats.sample_id_mismatches += 1
    lookup = category_lookup(doc.get("categories") or [])

    res_m = None
    nodata = None
    tile_excluded = False
    label_raster = None
    if rasters is not None and rasters.sar_path is not None:
        import rasterio
        with rasterio.open(rasters.sar_path) as s:
            if (s.height, s.width) != (height, width):
                raise BuildingTableError(f"{name}: JSON size {width}x{height} != SAR size {s.width}x{s.height}")
            res_m = metres_per_pixel(s.transform, s.crs, s.bounds)   # sqrt of ground pixel area
            sar = s.read()
        nodata = np.all(sar == rule.sar_value, axis=0)
        tile_excluded = float(nodata.mean()) >= rule.tile_exclude_frac
        if tile_excluded:
            stats.tiles_excluded_nodata += 1
        if keep_labels and rasters.label_path is not None:
            with rasterio.open(rasters.label_path) as lab_src:
                label_raster = lab_src.read(1)

    rows = []
    for ann in doc.get("annotations") or []:
        polys = ann.get("segmentation") or []
        if not isinstance(polys, list) or not polys or not all(isinstance(p, list) for p in polys):
            raise BuildingTableError(f"{name}: annotation {ann.get('id')} has no polygon segmentation (RLE?)")
        xs = [v for p in polys for v in p[0::2]]
        ys = [v for p in polys for v in p[1::2]]
        if min(xs) < -0.5 or min(ys) < -0.5 or max(xs) > width + 0.5 or max(ys) > height + 0.5:
            raise BuildingTableError(f"{name}: annotation {ann.get('id')} lies outside the {width}x{height} tile")
        cat = int(ann["category_id"])
        if cat not in lookup:
            raise BuildingTableError(f"{name}: annotation {ann.get('id')} uses undeclared category {cat}")

        mask, r0, c0 = rasterize_window(polys, height, width)
        area_px = int(mask.sum())
        win = (slice(r0, r0 + mask.shape[0]), slice(c0, c0 + mask.shape[1]))
        poly_area = sum(polygon_area(p[0::2], p[1::2]) for p in polys)
        reported = float(ann.get("area", poly_area))
        area_flag = reported > 0 and abs(area_px - reported) / reported > AREA_TOLERANCE
        stats.area_flags += int(area_flag)

        # Order matters: a blank tile excludes everything; an empty footprint is not a no-data case.
        nodata_frac = None
        if tile_excluded:
            excluded = "tile_nodata"
        elif area_px == 0:
            excluded = "empty_footprint"
        else:
            excluded = None
        if nodata is not None and area_px:
            nodata_frac = float(nodata[win][mask].mean())
            if excluded is None and nodata_frac >= rule.building_exclude_frac:
                excluded = "building_nodata"
        stats.buildings_excluded_nodata += int(excluded in ("tile_nodata", "building_nodata"))

        row = {
            "event": event,
            "role": info.role,
            "hazard": info.hazard,
            "tile_id": tile_id,
            "instance_id": int(ann["id"]),
            "n_polygons": len(polys),
            "area_px": area_px,
            "area_px_reported": reported,
            "area_flag": bool(area_flag),
            "bbox_x": float(ann["bbox"][0]) if ann.get("bbox") else None,
            "bbox_y": float(ann["bbox"][1]) if ann.get("bbox") else None,
            "bbox_w": float(ann["bbox"][2]) if ann.get("bbox") else None,
            "bbox_h": float(ann["bbox"][3]) if ann.get("bbox") else None,
            "res_m": res_m,
            "area_m2": area_px * res_m * res_m if res_m else None,
            "tokens_native": area_px / patch_px**2,
            "tokens_at_target": (area_px * res_m * res_m) / (patch_px * target_res_m) ** 2 if res_m else None,
            "nodata_frac": nodata_frac,
            "excluded_reason": excluded,
            "labels_locked": not keep_labels,
            "damage_class": lookup[cat] if keep_labels else None,
            "raster_majority_class": None,
            "segmentation_px": json.dumps(polys, separators=(",", ":")),
        }
        if keep_labels and label_raster is not None and area_px:
            vals = label_raster[win][mask]
            vals = vals[vals > 0]  # 1 intact, 2 damaged, 3 destroyed in the raster
            if vals.size:
                row["raster_majority_class"] = int(np.bincount(vals, minlength=4)[1:].argmax())
        rows.append(row)
    stats.buildings += len(rows)
    stats.events[event] = stats.events.get(event, 0) + len(rows)
    return rows


def build_building_table(
    labels: str | Path,
    events: EventsConfig,
    rule: NoDataRule | None = None,
    bright_root: str | Path | None = None,
    today: date | None = None,
    target_res_m: float = 0.3,
    patch_px: int = 16,
    only_events: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, BuildStats]:
    """Build the table from instance labels (zip or folder).

    ``bright_root`` (folder with ``post-event/`` and ``target/``) is optional; without it, no-data, metric
    areas and the raster-majority check are left empty.
    """
    rule = rule or NoDataRule()
    unlocked = labels_unlocked(events, today)
    wanted = set(only_events) if only_events else None
    root = Path(bright_root) if bright_root else None
    stats = BuildStats()
    rows: list[dict] = []
    for name, raw in _iter_instance_files(Path(labels)):
        tile_id = tile_id_from_instance_name(name)
        if wanted and event_key_from_filename(tile_id) not in wanted:
            continue
        stats.files += 1
        rasters = None
        if root is not None:
            sar = root / "post-event" / f"{tile_id}_post_disaster.tif"
            lab = root / "target" / f"{tile_id}_building_damage.tif"
            rasters = TileRaster(sar if sar.exists() else None, lab if lab.exists() else None)
        rows.extend(_tile_rows(name, raw, events, unlocked, rule, rasters, target_res_m, patch_px, stats))

    df = pd.DataFrame(rows)
    if len(df):
        df["damage_class"] = df["damage_class"].astype("Int8")
        df["raster_majority_class"] = df["raster_majority_class"].astype("Int8")
        df["damaged"] = (df["damage_class"] >= DAMAGED).astype("boolean")
        df.loc[df["damage_class"].isna(), "damaged"] = pd.NA
    return df, stats


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-event counts. Class counts are shown only where labels are not locked."""
    out = []
    for event, g in df.groupby("event", sort=True):
        kept = g[g["excluded_reason"].isna()]
        row = {"event": event, "role": g["role"].iloc[0], "buildings": len(g), "kept": len(kept),
               "excluded_nodata": int(g["excluded_reason"].isin(["tile_nodata", "building_nodata"]).sum()),
               "area_flags": int(g["area_flag"].sum())}
        if not g["labels_locked"].any():
            for k, v in CLASS_NAMES.items():
                row[v] = int((kept["damage_class"] == k).sum())
            row["damaged_or_destroyed"] = int(kept["damaged"].sum())
            both = kept.dropna(subset=["raster_majority_class"])
            row["raster_agreement"] = round(float((both["raster_majority_class"] == both["damage_class"]).mean()), 4) \
                if len(both) else None
        else:
            row["class_counts"] = "locked until freeze"
        for k in (1, 4, 9):
            row[f"pct_lt_{k}_tokens_native"] = round(float((kept["tokens_native"] < k).mean() * 100), 1) if len(kept) else None
        if kept["tokens_at_target"].notna().any():
            for k in (1, 4, 9):
                row[f"pct_lt_{k}_tokens_target"] = round(float((kept["tokens_at_target"] < k).mean() * 100), 1)
        out.append(row)
    return pd.DataFrame(out)


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p
