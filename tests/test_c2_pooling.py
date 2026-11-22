import math

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.c2_offsets import OffsetError, matched_shifts
from damagesar_bench.pooling.footprint import (Method, PoolingError, TileGeometry, pool_building, shift_polygons,
                                               tile_shift)

GEOM = TileGeometry(64, 64, 0.5, 0.5)


def test_methods_parse_and_name():
    assert Method.parse("matched").name == "matched"
    assert Method.parse("rshift:3").name == "rshift3"
    assert Method.parse("matched").shifted and Method.parse("rshift:0").shifted
    assert not Method.parse("outline").shifted
    with pytest.raises(PoolingError):
        Method.parse("rshift:-1")


def test_shift_polygons_moves_in_metres_per_axis():
    g = TileGeometry(64, 64, 0.5, 0.25)          # non-square pixels
    out = shift_polygons([[0, 0, 4, 0, 4, 4]], dy_m=1.0, dx_m=2.0, geom=g)
    assert out == [[4.0, 4.0, 8.0, 4.0, 8.0, 8.0]]  # dx 2 m / 0.5 = 4 px, dy 1 m / 0.25 = 4 px


def test_matched_uses_the_tile_shift_and_rshift_keeps_its_length():
    shifts = {"t1": (3.0, 4.0), "t2": (0.0, 0.0)}
    assert tile_shift(Method.parse("matched"), "t1", shifts) == (3.0, 4.0)
    assert tile_shift(Method.parse("outline"), "t1", None) == (0.0, 0.0)
    dy, dx = tile_shift(Method.parse("rshift:0"), "t1", shifts)
    assert math.isclose(math.hypot(dy, dx), 5.0)
    assert tile_shift(Method.parse("rshift:0"), "t1", shifts) == (dy, dx)          # same draw, same shift
    assert tile_shift(Method.parse("rshift:1"), "t1", shifts) != (dy, dx)          # another draw differs
    assert tile_shift(Method.parse("rshift:5"), "t2", shifts) == (0.0, 0.0)        # not moved under matched
    assert tile_shift(Method.parse("matched"), "unknown", shifts) == (0.0, 0.0)
    with pytest.raises(PoolingError):
        tile_shift(Method.parse("matched"), "t1", None)


def test_random_directions_are_spread():
    shifts = {f"t{i}": (0.0, 10.0) for i in range(400)}
    ang = [math.atan2(*tile_shift(Method.parse("rshift:0"), t, shifts)) for t in shifts]
    r = math.hypot(np.cos(ang).mean(), np.sin(ang).mean())
    assert r < 0.15                                                                  # no preferred direction


def test_shifted_pooling_reads_the_moved_tokens():
    tokens = np.zeros((4, 4, 2))
    tokens[0, 0] = [1, 0]                       # token under the footprint
    tokens[0, 2] = [0, 1]                       # token 16 m (32 px at 0.5 m) to the east
    poly = [[2, 2, 10, 2, 10, 10, 2, 10]]
    v0, _ = pool_building(tokens, poly, GEOM, Method("outline"))
    v1, _ = pool_building(tokens, shift_polygons(poly, 0.0, 16.0, GEOM), GEOM, Method("outline"))
    assert v0.tolist() == [1, 0] and v1.tolist() == [0, 1]


def test_matched_shifts_only_for_real_offsets():
    off = pd.DataFrame({"tile_id": ["a", "b", "c"], "dy_m": [1.0, 2.0, 3.0], "dx_m": [4.0, 5.0, 6.0],
                        "peak_z": [5.0, 5.0, 1.0], "beats_null": [True, False, True]})
    assert matched_shifts(off) == {"a": (1.0, 4.0), "b": (0.0, 0.0), "c": (0.0, 0.0)}
    with pytest.raises(OffsetError):
        matched_shifts(off.drop(columns="beats_null"))
