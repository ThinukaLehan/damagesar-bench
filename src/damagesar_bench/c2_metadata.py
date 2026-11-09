"""C2: find the source SAR scene of each BRIGHT event and record its viewing geometry.

BRIGHT ships terrain-corrected 8-bit tiles without the scene metadata, so incidence angle,
look direction and orbit direction have to be recovered from the providers' open-data
STAC catalogs (Capella, Umbra; CC BY 4.0). Matching rule, decided before looking:

* a STAC item is a candidate for an event if its footprint bbox intersects the event's
  tile area (union of BRIGHT tile bounds, WGS84) and it was acquired between
  ``start - 30 days`` and ``end + 90 days`` (BRIGHT uses post-event scenes; some events
  span months, e.g. La Palma, Ukraine);
* candidates are ranked by covered fraction of the event area (high first), then by the
  number of days after the event (low first), then before-event scenes last;
* the best candidate's geometry is taken as the event's geometry. Several good candidates
  with different geometry are flagged (``geometry_consistent = False``), because BRIGHT
  tiles of one event may come from more than one scene.

Status per event:
    geometry-ready  scene matched and incidence angle, look direction and orbit state found
    match-only      scene matched but at least one of those three is missing
    dilated-only    no scene matched: C2 falls back to dilated footprints for this event

This module is pure (no network) so it can be unit-tested; the C06 notebook does the
fetching in Colab.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable

# Event dates from BRIGHT Table 2 (Chen et al., ESSD 2025 / arXiv 2501.06019) and the
# SAR provider(s) named there. Ranges are inclusive.
EVENT_DATES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "beirut": ("2020-08-04", "2020-08-04", ("capella",)),
    "bata": ("2021-03-07", "2021-03-07", ("capella",)),
    "congo": ("2021-05-22", "2021-05-22", ("capella",)),
    "haiti": ("2021-08-14", "2021-08-14", ("capella",)),
    "la_palma": ("2021-09-19", "2021-12-13", ("capella",)),
    "marshall": ("2021-12-30", "2022-01-01", ("capella",)),
    "ukraine": ("2022-03-22", "2022-09-21", ("capella",)),
    "turkey": ("2023-02-06", "2023-02-06", ("capella", "umbra")),
    "myanmar": ("2023-05-14", "2023-05-14", ("capella",)),
    "hawaii": ("2023-08-08", "2023-08-09", ("capella",)),
    "morocco": ("2023-09-08", "2023-09-08", ("capella",)),
    "libya": ("2023-09-10", "2023-09-10", ("capella",)),
    "mexico": ("2023-10-25", "2023-10-25", ("capella",)),
    "noto": ("2024-01-01", "2024-01-01", ("umbra",)),
}

DAYS_BEFORE = 30
DAYS_AFTER = 90

# STAC property names that carry each quantity, in order of preference. Covers the STAC
# sar/sat/view extensions and the provider-specific names seen in Capella/Umbra metadata.
FIELD_KEYS: dict[str, tuple[str, ...]] = {
    "incidence_deg": (
        "view:incidence_angle", "sar:incidence_angle", "capella:incidence_angle",
        "incidence_angle", "umbra:incidence_angle",
    ),
    "look_angle_deg": ("view:off_nadir", "capella:look_angle", "look_angle", "sar:look_angle"),
    "look_direction": (
        "sar:observation_direction", "capella:observation_direction",
        "observation_direction", "look_direction", "umbra:look_direction",
    ),
    "orbit_state": ("sat:orbit_state", "capella:orbit_state", "orbit_state", "orbit_direction"),
    "squint_deg": ("capella:squint_angle", "squint_angle", "umbra:squint_angle"),
    "azimuth_deg": ("view:azimuth", "capella:azimuth", "azimuth_angle"),
    "layover_deg": ("capella:layover_angle", "layover_angle"),
    "mode": ("sar:instrument_mode", "capella:instrument_mode", "instrument_mode"),
    "platform": ("platform", "constellation"),
    "product_type": ("sar:product_type", "capella:product_type", "product_type"),
    "resolution_m": ("sar:resolution_range", "gsd", "sar:pixel_spacing_range"),
}

GEOMETRY_FIELDS = ("incidence_deg", "look_direction", "orbit_state")


@dataclass(frozen=True)
class BBox:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def of(cls, seq: Iterable[float]) -> "BBox":
        w, s, e, n = (float(v) for v in seq)
        if w > e or s > n:
            raise ValueError(f"bad bbox {w, s, e, n}")
        return cls(w, s, e, n)

    @property
    def area(self) -> float:
        return (self.east - self.west) * (self.north - self.south)

    def intersection(self, other: "BBox") -> "BBox | None":
        w, s = max(self.west, other.west), max(self.south, other.south)
        e, n = min(self.east, other.east), min(self.north, other.north)
        if w >= e or s >= n:
            return None
        return BBox(w, s, e, n)

    def union(self, other: "BBox") -> "BBox":
        return BBox(min(self.west, other.west), min(self.south, other.south),
                    max(self.east, other.east), max(self.north, other.north))


def union_bbox(boxes: Iterable[BBox]) -> BBox:
    out = None
    for b in boxes:
        out = b if out is None else out.union(b)
    if out is None:
        raise ValueError("no boxes")
    return out


def coverage(event_area: BBox, item_bbox: BBox) -> float:
    """Fraction of the event's bbox covered by the item's bbox (0..1)."""
    inter = event_area.intersection(item_bbox)
    if inter is None or event_area.area == 0:
        return 0.0
    return inter.area / event_area.area


def parse_datetime(value: str) -> dt.date:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def item_date(props: dict) -> dt.date | None:
    for key in ("datetime", "start_datetime", "capella:collect_time"):
        if props.get(key):
            return parse_datetime(str(props[key]))
    return None


def in_window(event: str, when: dt.date) -> bool:
    start, end, _ = EVENT_DATES[event]
    lo = dt.date.fromisoformat(start) - dt.timedelta(days=DAYS_BEFORE)
    hi = dt.date.fromisoformat(end) + dt.timedelta(days=DAYS_AFTER)
    return lo <= when <= hi


def days_after_event(event: str, when: dt.date) -> int:
    """Days after the event start (negative = before the event)."""
    return (when - dt.date.fromisoformat(EVENT_DATES[event][0])).days


def first_present(props: dict, keys: Iterable[str]):
    for k in keys:
        v = props.get(k)
        if v not in (None, "", []):
            return v[0] if isinstance(v, list) and len(v) == 1 else v
    return None


def normalise_direction(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s in ("left", "l"):
        return "left"
    if s in ("right", "r"):
        return "right"
    return s


def normalise_orbit(v) -> str | None:
    if v is None:
        return None
    s = str(v).strip().lower()
    if s.startswith("asc"):
        return "ascending"
    if s.startswith("desc"):
        return "descending"
    return s


def extract_geometry(props: dict) -> dict:
    """Pull the C2-relevant fields out of a STAC item's properties."""
    out = {name: first_present(props, keys) for name, keys in FIELD_KEYS.items()}
    out["look_direction"] = normalise_direction(out["look_direction"])
    out["orbit_state"] = normalise_orbit(out["orbit_state"])
    for k in ("incidence_deg", "look_angle_deg", "squint_deg", "azimuth_deg", "layover_deg"):
        if out[k] is not None:
            try:
                out[k] = float(out[k])
            except (TypeError, ValueError):
                out[k] = None
    # Off-nadir/look angle is not the incidence angle; never substitute one for the other.
    return out


def rank_key(row: dict) -> tuple:
    after = row["days_after"]
    return (-round(row["coverage"], 3), 0 if after >= 0 else 1, abs(after))


def match_event(event: str, area: BBox, items: Iterable[dict]) -> list[dict]:
    """Return candidate rows for one event, best first.

    ``items`` are STAC item dicts (``id``, ``bbox``, ``properties``, optional ``provider``).
    """
    rows = []
    for it in items:
        bbox = it.get("bbox")
        when = item_date(it.get("properties", {}))
        if not bbox or when is None or not in_window(event, when):
            continue
        # 3-D STAC bbox is [w, s, zmin, e, n, zmax]
        flat = [bbox[0], bbox[1], bbox[3], bbox[4]] if len(bbox) == 6 else bbox
        cov = coverage(area, BBox.of(flat))
        if cov <= 0:
            continue
        geo = extract_geometry(it.get("properties", {}))
        rows.append({
            "event": event, "item_id": it.get("id"), "provider": it.get("provider"),
            "date": when.isoformat(), "days_after": days_after_event(event, when),
            "coverage": cov, "href": it.get("href"), **geo,
        })
    rows.sort(key=rank_key)
    return rows


def status_of(best: dict | None) -> str:
    if best is None:
        return "dilated-only"
    if all(best.get(f) not in (None, "") for f in GEOMETRY_FIELDS):
        return "geometry-ready"
    return "match-only"


def summarise_event(event: str, candidates: list[dict], min_cov: float = 0.5) -> dict:
    """One row per event: best candidate plus a consistency flag over the good candidates."""
    best = candidates[0] if candidates else None
    good = [c for c in candidates if c["coverage"] >= min_cov]
    looks = {(c.get("look_direction"), c.get("orbit_state")) for c in good}
    inc = [c["incidence_deg"] for c in good if c.get("incidence_deg") is not None]
    row = {
        "event": event,
        "event_start": EVENT_DATES[event][0],
        "event_end": EVENT_DATES[event][1],
        "providers_in_paper": "+".join(EVENT_DATES[event][2]),
        "n_candidates": len(candidates),
        "n_good_candidates": len(good),
        "status": status_of(best),
        "geometry_consistent": len(looks) <= 1 and (not inc or max(inc) - min(inc) <= 5.0),
        "incidence_range_good": f"{min(inc):.1f}-{max(inc):.1f}" if inc else "",
    }
    keep = ("item_id", "provider", "date", "days_after", "coverage", "incidence_deg",
            "look_angle_deg", "look_direction", "orbit_state", "squint_deg", "azimuth_deg",
            "layover_deg", "mode", "platform", "product_type", "resolution_m", "href")
    for k in keep:
        row[f"best_{k}"] = best.get(k) if best else None
    return row
