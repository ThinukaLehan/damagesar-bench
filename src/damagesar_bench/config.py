"""Load the YAML configuration files.

Two files drive everything:
- ``configs/events.yaml``: event roles, hazards, building-unit type and the freeze date.
- ``configs/bench.yaml``: data paths, file-name suffixes and the no-data rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

ROLES = ("confirmatory", "development", "secondary")
UNIT_TYPES = ("instance", "fallback", "none")

# Repository root = two levels above this file's package directory (src/damagesar_bench/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVENTS = REPO_ROOT / "configs" / "events.yaml"
DEFAULT_BENCH = REPO_ROOT / "configs" / "bench.yaml"


@dataclass(frozen=True)
class EventInfo:
    key: str
    name: str
    hazard: str
    role: str
    units: str


@dataclass(frozen=True)
class EventsConfig:
    freeze_date: date
    events: dict[str, EventInfo]

    def keys_with_role(self, role: str) -> list[str]:
        return sorted(k for k, e in self.events.items() if e.role == role)


@dataclass(frozen=True)
class NoDataRule:
    sar_value: float = 0
    tile_exclude_frac: float = 0.95
    building_exclude_frac: float = 0.50


@dataclass(frozen=True)
class DataConfig:
    root: Path
    sar_dir: str = "post-event"
    label_dir: str = "target"
    sar_suffix: str = "_post_disaster.tif"
    label_suffix: str = "_building_damage.tif"
    label_values: tuple[int, ...] = (0, 1, 2, 3)


@dataclass(frozen=True)
class BenchConfig:
    data: DataConfig
    nodata: NoDataRule
    outputs_dir: Path = field(default=Path("outputs"))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        content = yaml.safe_load(fh)
    if not isinstance(content, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return content


def load_events(path: str | Path = DEFAULT_EVENTS) -> EventsConfig:
    raw = _read_yaml(Path(path))
    freeze = raw.get("freeze_date")
    if freeze is None:
        raise ValueError("events config needs a freeze_date")
    freeze_date = freeze if isinstance(freeze, date) else date.fromisoformat(str(freeze))

    events: dict[str, EventInfo] = {}
    for key, spec in (raw.get("events") or {}).items():
        role = spec.get("role")
        units = spec.get("units")
        if role not in ROLES:
            raise ValueError(f"event {key!r}: role must be one of {ROLES}, got {role!r}")
        if units not in UNIT_TYPES:
            raise ValueError(f"event {key!r}: units must be one of {UNIT_TYPES}, got {units!r}")
        events[key] = EventInfo(
            key=key,
            name=spec.get("name", key),
            hazard=spec.get("hazard", "unknown"),
            role=role,
            units=units,
        )
    if not events:
        raise ValueError("events config lists no events")
    return EventsConfig(freeze_date=freeze_date, events=events)


def load_bench(path: str | Path = DEFAULT_BENCH, root_override: str | Path | None = None) -> BenchConfig:
    raw = _read_yaml(Path(path))
    d = raw.get("data", {})
    n = raw.get("nodata", {})
    o = raw.get("outputs", {})
    data = DataConfig(
        root=Path(root_override if root_override is not None else d.get("root", ".")),
        sar_dir=d.get("sar_dir", "post-event"),
        label_dir=d.get("label_dir", "target"),
        sar_suffix=d.get("sar_suffix", "_post_disaster.tif"),
        label_suffix=d.get("label_suffix", "_building_damage.tif"),
        label_values=tuple(d.get("label_values", (0, 1, 2, 3))),
    )
    rule = NoDataRule(
        sar_value=n.get("sar_value", 0),
        tile_exclude_frac=float(n.get("tile_exclude_frac", 0.95)),
        building_exclude_frac=float(n.get("building_exclude_frac", 0.50)),
    )
    if not 0 < rule.tile_exclude_frac <= 1 or not 0 < rule.building_exclude_frac <= 1:
        raise ValueError("no-data fractions must be in (0, 1]")
    return BenchConfig(data=data, nodata=rule, outputs_dir=Path(o.get("dir", "outputs")))
