"""Data validation (FR1, FR2, DR1).

Checks every tile pair: pairing, shape, CRS and geotransform agreement, label value range, and the
SAR no-data share. Writes two CSV files:

- ``tile_report.csv``: one row per tile.
- ``event_summary.csv``: one row per event.

Before the plan freeze, confirmatory labels are read only to check their *schema* (shape and value
range). Their class counts are not reported, so validation never reveals new confirmatory label
information.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from damagesar_bench.config import BenchConfig, EventsConfig
from damagesar_bench.data.discover import TilePair, discover_pairs
from damagesar_bench.data.events import labels_unlocked
from damagesar_bench.data.nodata import nodata_fraction, nodata_mask, tile_is_excluded

CLASS_NAMES = {1: "intact", 2: "damaged", 3: "destroyed"}


@dataclass
class TileResult:
    tile_id: str
    event: str
    role: str
    paired: bool
    height: int | None = None
    width: int | None = None
    sar_bands: int | None = None
    crs: str | None = None
    shape_match: bool | None = None
    crs_match: bool | None = None
    transform_match: bool | None = None
    labels_in_range: bool | None = None
    nodata_frac: float | None = None
    excluded_nodata: bool | None = None
    counts_reported: bool = False
    px_intact: int | None = None
    px_damaged: int | None = None
    px_destroyed: int | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _check_pair(pair: TilePair, role: str, cfg: BenchConfig, report_counts: bool) -> TileResult:
    res = TileResult(tile_id=pair.tile_id, event=pair.event, role=role, paired=pair.paired)
    if not pair.paired:
        res.problems.append("missing SAR file" if pair.sar_path is None else "missing label file")
        return res

    with rasterio.open(pair.sar_path) as s, rasterio.open(pair.label_path) as l:
        res.height, res.width, res.sar_bands = s.height, s.width, s.count
        res.crs = s.crs.to_string() if s.crs else None
        res.shape_match = (s.height, s.width) == (l.height, l.width)
        res.crs_match = s.crs == l.crs
        res.transform_match = s.transform.almost_equals(l.transform)
        sar = s.read()
        lab = l.read(1)

    if not res.shape_match:
        res.problems.append("SAR and label shapes differ")
    if res.crs is None:
        res.problems.append("SAR has no CRS")
    if not res.crs_match:
        res.problems.append("SAR and label CRS differ")
    if not res.transform_match:
        res.problems.append("SAR and label geotransforms differ")

    allowed = np.asarray(cfg.data.label_values)
    res.labels_in_range = bool(np.isin(lab, allowed).all())
    if not res.labels_in_range:
        res.problems.append(f"label values outside {cfg.data.label_values}")

    mask = nodata_mask(sar, cfg.nodata)
    res.nodata_frac = round(nodata_fraction(mask), 6)
    res.excluded_nodata = tile_is_excluded(mask, cfg.nodata)

    if report_counts:
        res.counts_reported = True
        res.px_intact = int((lab == 1).sum())
        res.px_damaged = int((lab == 2).sum())
        res.px_destroyed = int((lab == 3).sum())
    return res


def validate_dataset(
    cfg: BenchConfig, events: EventsConfig, today: date | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate every tile; return (tile report, event summary)."""
    unlocked = labels_unlocked(events, today)
    rows = []
    for pair in discover_pairs(cfg.data):
        info = events.events.get(pair.event)
        if info is None:
            r = TileResult(tile_id=pair.tile_id, event=pair.event, role="unknown", paired=pair.paired)
            r.problems.append(f"event {pair.event!r} is not in events.yaml")
            rows.append(r)
            continue
        report_counts = info.role != "confirmatory" or unlocked
        rows.append(_check_pair(pair, info.role, cfg, report_counts))

    tiles = pd.DataFrame(
        [{**{k: v for k, v in asdict(r).items() if k != "problems"}, "problems": "; ".join(r.problems), "ok": r.ok}
         for r in rows]
    )
    summary = summarise_events(tiles, events)
    return tiles, summary


def summarise_events(tiles: pd.DataFrame, events: EventsConfig) -> pd.DataFrame:
    out = []
    for key, info in sorted(events.events.items()):
        t = tiles[tiles["event"] == key] if len(tiles) else tiles
        row = {
            "event": key,
            "name": info.name,
            "hazard": info.hazard,
            "role": info.role,
            "units": info.units,
            "n_tiles": int(len(t)),
            "n_problem_tiles": int((~t["ok"]).sum()) if len(t) else 0,
            "n_excluded_nodata": int(t["excluded_nodata"].fillna(False).astype(bool).sum()) if len(t) else 0,
        }
        reported = t[t["counts_reported"]] if len(t) else t
        if len(t) and len(reported) == len(t):
            tot = {c: int(reported[f"px_{c}"].sum()) for c in CLASS_NAMES.values()}
            n = sum(tot.values())
            for c, v in tot.items():
                row[f"share_{c}"] = round(v / n, 4) if n else None
            row["class_counts"] = "reported"
        else:
            row["class_counts"] = "locked until freeze" if len(t) else "no tiles"
        out.append(row)
    return pd.DataFrame(out)


def write_reports(tiles: pd.DataFrame, summary: pd.DataFrame, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_path, s_path = out / "tile_report.csv", out / "event_summary.csv"
    tiles.to_csv(t_path, index=False)
    summary.to_csv(s_path, index=False)
    return t_path, s_path
