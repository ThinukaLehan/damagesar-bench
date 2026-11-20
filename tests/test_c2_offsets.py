import json

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.c2_offsets import (
    OffsetError, block_mean, correlation_surface, event_summary, footprint_mask, mean_resultant_length,
    measure_offsets, null_gains, tile_offset)
from damagesar_bench.pooling.footprint import TileGeometry


def rect(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def scene(seed=0, n=40, size=256):
    rng = np.random.default_rng(seed)
    polys = []
    for _ in range(n):
        x, y = rng.integers(20, size - 40, 2)
        polys.append(rect(int(x), int(y), int(x) + int(rng.integers(6, 16)), int(y) + int(rng.integers(6, 16))))
    return polys


def sar_from(polys, dy, dx, size=256, noise=0.05, seed=1):
    """SAR that is bright where the footprints are, shifted by (dy, dx) pixels; never exactly 0 (no no-data)."""
    m = footprint_mask(polys, size, size).astype(float)
    m = np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    rng = np.random.default_rng(seed)
    return np.clip(0.2 + 0.6 * m + noise * rng.standard_normal(m.shape), 0.01, 1.0)


def test_block_mean_and_mask():
    a = np.arange(16, dtype=float).reshape(4, 4)
    assert block_mean(a, 2).tolist() == [[2.5, 4.5], [10.5, 12.5]]
    m = footprint_mask([rect(1, 1, 3, 3), rect(2, 2, 4, 4)], 6, 6)
    assert m.dtype == bool and m[1, 1] and m[3, 3] and not m[0, 0] and not m[5, 5]
    with pytest.raises(OffsetError):
        block_mean(a, 0)


def test_correlation_surface_finds_a_known_shift():
    rng = np.random.default_rng(0)
    m = (rng.random((40, 40)) > 0.8).astype(float)
    s = np.roll(np.roll(m, 3, axis=0), -2, axis=1)                  # signal 3 down, 2 left of the mask
    surf = correlation_surface(m, s, np.ones_like(m, bool), 5)
    by, bx = np.unravel_index(np.argmax(surf), surf.shape)
    assert (by - 5, bx - 5) == (3, -2)


@pytest.mark.parametrize("dy,dx", [(12, -8), (0, 16), (-20, 4)])
def test_tile_offset_recovers_the_shift_in_metres(dy, dx):
    polys = scene()
    off = tile_offset("t", polys, sar_from(polys, dy, dx), px_y_m=0.5, px_x_m=0.5, factor=4, max_shift_m=20)
    # One block = 4 px = 2 m: the estimate is within one block of the truth.
    assert off.dy_m == pytest.approx(dy * 0.5, abs=2.0) and off.dx_m == pytest.approx(dx * 0.5, abs=2.0)
    assert off.peak_z > 5 and off.n_buildings == len(polys)
    if (dy, dx) != (0, 0):
        assert off.gain_z > 0


def test_no_shift_gives_zero_gain_and_null_is_lower():
    polys = scene()
    off = tile_offset("t", polys, sar_from(polys, 0, 0), 0.5, 0.5, factor=4, max_shift_m=20)
    assert off.magnitude_m == pytest.approx(0.0) and off.gain_z == pytest.approx(0.0)
    shifted = tile_offset("t", polys, sar_from(polys, 16, 0), 0.5, 0.5, factor=4, max_shift_m=20)
    other = sar_from(scene(seed=7), 16, 0, seed=3)                   # a different scene: no real match
    nulls = null_gains(polys, [other], 0.5, 0.5, factor=4, max_shift_m=20)
    assert shifted.gain_z > max(nulls)


def test_no_data_pixels_are_ignored():
    polys = scene()
    s = sar_from(polys, 8, 8)
    s[:, :64] = 0.0                                                  # a no-data strip
    off = tile_offset("t", polys, s, 0.5, 0.5, factor=4, max_shift_m=20)
    assert off.dy_m == pytest.approx(4.0, abs=2.0) and off.dx_m == pytest.approx(4.0, abs=2.0)


def test_direction_statistics():
    assert mean_resultant_length(np.array([10.0, 12.0, 8.0])) == pytest.approx(1.0, abs=0.01)
    assert mean_resultant_length(np.array([0.0, 90.0, 180.0, 270.0])) == pytest.approx(0.0, abs=1e-9)
    df = pd.DataFrame({"event": ["a"] * 4, "magnitude_m": [5.0, 6.0, 7.0, 1.0], "angle_deg": [90, 92, 88, 0],
                       "peak_z": [6, 6, 6, 1], "beats_null": [True, True, False, False]})
    s = event_summary(df).iloc[0]
    assert s["tiles"] == 4 and s["share_confident"] == 0.75 and s["share_beats_null"] == 0.5
    assert s["median_offset_confident_m"] == 6.0 and s["direction_R"] == pytest.approx(1.0, abs=0.01)
    assert s["mean_direction_deg"] == pytest.approx(90.0, abs=0.5)


def test_measure_offsets_end_to_end():
    tiles = {f"beirut_{i}": scene(seed=i) for i in range(4)}
    rows = []
    for t, polys in tiles.items():
        for j, p in enumerate(polys):
            rows.append({"event": "beirut", "tile_id": t, "instance_id": j, "excluded_reason": None,
                         "segmentation_px": json.dumps(p)})
    b = pd.DataFrame(rows)
    sars = {t: sar_from(p, 12, 0, seed=k) for k, (t, p) in enumerate(tiles.items())}
    geom = TileGeometry(256, 256, 0.5, 0.5)
    df = measure_offsets(b, lambda t: (sars[t], geom), ["beirut"], factor=4, max_shift_m=20, n_null=3)
    assert len(df) == 4 and df["beats_null"].all()
    assert df["dy_m"].between(4.0, 8.0).all()                        # 12 px x 0.5 m = 6 m
    s = event_summary(df).iloc[0]
    assert s["share_beats_null"] == 1.0 and s["direction_R"] > 0.95
    with pytest.raises(OffsetError):
        measure_offsets(b, lambda t: (sars[t], geom), ["haiti"])
