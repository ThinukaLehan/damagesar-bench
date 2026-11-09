"""Shared fixtures: a tiny synthetic dataset laid out like BRIGHT.

Layout: <root>/post-event/<event>-<hazard>_<id>_post_disaster.tif and
        <root>/target/<event>-<hazard>_<id>_building_damage.tif
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from damagesar_bench.config import BenchConfig, DataConfig, EventInfo, EventsConfig, NoDataRule

SIZE = 32
CRS = "EPSG:32637"


def write_tif(path: Path, arr: np.ndarray, crs: str | None = CRS, transform=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    transform = transform or from_origin(500000, 4000000, 0.5, 0.5)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
        dtype=arr.dtype, crs=crs, transform=transform,
    ) as dst:
        dst.write(arr)


def make_tile(root: Path, tile_id: str, sar: np.ndarray, label: np.ndarray, **kw) -> None:
    write_tif(root / "post-event" / f"{tile_id}_post_disaster.tif", sar, **kw)
    write_tif(root / "target" / f"{tile_id}_building_damage.tif", label, **kw)


def simple_label() -> np.ndarray:
    lab = np.zeros((SIZE, SIZE), dtype=np.uint8)
    lab[2:10, 2:10] = 1   # intact building, 64 px
    lab[12:16, 12:16] = 2  # damaged building, 16 px
    lab[20:22, 20:22] = 3  # destroyed building, 4 px
    return lab


def good_sar() -> np.ndarray:
    return np.full((SIZE, SIZE), 120, dtype=np.uint8)


@pytest.fixture
def events_cfg() -> EventsConfig:
    ev = {
        "beirut": EventInfo("beirut", "Beirut explosion", "explosion", "development", "instance"),
        "hawaii": EventInfo("hawaii", "Hawaii wildfire", "wildfire", "development", "instance"),
        "haiti": EventInfo("haiti", "Haiti earthquake", "earthquake", "confirmatory", "instance"),
        "marshall": EventInfo("marshall", "Marshall wildfire", "wildfire", "secondary", "instance"),
        "mexico": EventInfo("mexico", "Mexico hurricane", "storm", "secondary", "none"),
    }
    return EventsConfig(freeze_date=date(2027, 1, 10), events=ev)


@pytest.fixture
def before_freeze() -> date:
    return date(2026, 11, 9)


@pytest.fixture
def after_freeze() -> date:
    return date(2027, 1, 11)


@pytest.fixture(autouse=True)
def _no_unlock_env(monkeypatch):
    monkeypatch.delenv("DAMAGESAR_UNLOCK_CONFIRMATORY", raising=False)


@pytest.fixture
def bright_root(tmp_path: Path) -> Path:
    root = tmp_path / "bright"
    make_tile(root, "beirut-explosion_00000000", good_sar(), simple_label())
    make_tile(root, "beirut-explosion_00000001", good_sar(), simple_label())
    blank = np.zeros((SIZE, SIZE), dtype=np.uint8)  # 100% no-data -> excluded
    make_tile(root, "hawaii-wildfire_00000000", blank, simple_label())
    make_tile(root, "haiti-earthquake_00000000", good_sar(), simple_label())
    return root


@pytest.fixture
def bench_cfg(bright_root: Path) -> BenchConfig:
    return BenchConfig(data=DataConfig(root=bright_root), nodata=NoDataRule())
