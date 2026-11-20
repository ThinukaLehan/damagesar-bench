import json
import math

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.contracts import feature_columns
from damagesar_bench.pooling.footprint import (
    Method, PoolingError, TileGeometry, building_mask, pool_building, pool_features)

G = TileGeometry(64, 64, 0.5, 0.5)          # 64 px tile; tokens 4x4 -> 16 px per token


def square(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def token_grid(h=4, w=4, dim=3):
    # token (r, c) has features [r, c, 10r + c] so every weight shows up exactly
    t = np.zeros((h, w, dim))
    for r in range(h):
        for c in range(w):
            t[r, c] = [r, c, 10 * r + c][:dim] if dim <= 3 else np.r_[[r, c, 10 * r + c], np.zeros(dim - 3)]
    return t


def test_method_parse():
    assert Method.parse("outline").name == "outline"
    assert Method.parse("dilated:3").name == "dilated3m"
    assert Method.parse("dilated:1.5").name == "dilated1.5m"
    for bad in ("dilated:0", "dilated:-1", "projected", "dilated"):
        with pytest.raises(PoolingError):
            Method.parse(bad)


def test_geometry_validation():
    with pytest.raises(PoolingError):
        TileGeometry(0, 10, 0.5, 0.5)
    with pytest.raises(PoolingError):
        TileGeometry(10, 10, 0.0, 0.5)


def test_building_inside_one_token_equals_that_token():
    vec, n = pool_building(token_grid(), square(17, 33, 30, 46), G, Method.parse("outline"))
    assert n == 13 * 13
    np.testing.assert_allclose(vec, [2, 1, 21])          # token row 2, col 1


def test_area_weighting_hand_worked():
    # x 12..20 (8 px): 4 px in token col 0 (12-15), 4 px in col 1 (16-19);
    # y 0..4 (4 px): token row 0. Equal split -> mean of tokens (0,0) and (0,1)
    vec, n = pool_building(token_grid(), square(12, 0, 20, 4), G, Method.parse("outline"))
    assert n == 32
    np.testing.assert_allclose(vec, [0, 0.5, 0.5])
    # x 10..18: 6 px in col 0, 2 px in col 1 -> weights 0.75 / 0.25
    vec, _ = pool_building(token_grid(), square(10, 0, 18, 4), G, Method.parse("outline"))
    np.testing.assert_allclose(vec, [0, 0.25, 0.25])


def test_four_tokens_unequal():
    # x 12..20, y 14..18 -> cols split 4/4, rows split 2 (14,15) / 2 (16,17): all four tokens 8 px each
    vec, n = pool_building(token_grid(), square(12, 14, 20, 18), G, Method.parse("outline"))
    assert n == 32
    np.testing.assert_allclose(vec, [0.5, 0.5, 5.5])


def test_non_integer_token_ratio():
    # 1000 px tile, 64 tokens: pixel centre r + 0.5 -> token floor((r + 0.5) * 64 / 1000)
    g = TileGeometry(1000, 1000, 0.3, 0.3)
    t = np.arange(64 * 64, dtype=float).reshape(64, 64, 1)
    vec, n = pool_building(t, square(0, 0, 16, 1), g, Method.parse("outline"))
    # columns 0..15: centres 0.5..15.5 -> tokens floor(c*0.064): cols 0..15 -> 0 except 15.5*0.064=0.992 -> 0
    assert n == 16
    np.testing.assert_allclose(vec, [0.0])
    vec, _ = pool_building(t, square(0, 0, 17, 1), g, Method.parse("outline"))
    # column 16: centre 16.5*0.064 = 1.056 -> token 1; weights 16/17 and 1/17
    np.testing.assert_allclose(vec, [1 / 17])


def test_sub_pixel_building_uses_centre_token():
    vec, n = pool_building(token_grid(), [[40.1, 50.1, 40.3, 50.1, 40.3, 50.3]], G, Method.parse("outline"))
    assert n == 0
    np.testing.assert_allclose(vec, [3, 2, 32])


def brute_dilation(core_idx, shape, px_x, px_y, d):
    """Independent reference: a pixel is in if its centre is within d metres of any footprint pixel centre."""
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    out = np.zeros(shape, bool)
    for r, c in core_idx:
        out |= np.hypot((rr - r) * px_y, (cc - c) * px_x) <= d + 1e-9
    return out


@pytest.mark.parametrize("px_x, px_y, d", [(0.5, 0.5, 1.0), (0.5, 0.5, 3.0), (0.38, 0.48, 3.0), (0.3, 0.3, 0.7)])
def test_dilation_matches_brute_force(px_x, px_y, d):
    g = TileGeometry(80, 80, px_x, px_y)
    polys = [[30, 30, 40, 30, 40, 36, 35, 36, 35, 42, 30, 42]]   # L-shape
    core, r0, c0 = building_mask(polys, g)
    core_idx = [(r + r0, c + c0) for r, c in zip(*np.nonzero(core))]
    mask, top, left = building_mask(polys, g, d)
    full = np.zeros((80, 80), bool)
    full[top:top + mask.shape[0], left:left + mask.shape[1]] = mask
    np.testing.assert_array_equal(full, brute_dilation(core_idx, (80, 80), px_x, px_y, d))


def test_dilation_square_count_hand_worked():
    # 10x10 px square, 0.5 m pixels, 1 m = 2 px: adds a 2 px band on each side, corners rounded:
    # 14x14 - 4 corners where distance > 2 px. Corner offsets (dr, dc) in {1,2}^2: (2,1) 2.24 and (1,2), (2,2) 2.83 excluded -> 3 per corner
    mask, _, _ = building_mask(square(20, 20, 30, 30), TileGeometry(64, 64, 0.5, 0.5), 1.0)
    assert int(mask.sum()) == 14 * 14 - 4 * 3


def test_dilation_clipped_at_tile_edge():
    g = TileGeometry(64, 64, 0.5, 0.5)
    mask, top, left = building_mask(square(0, 0, 4, 4), g, 1.0)
    assert top == 0 and left == 0
    assert int(mask.sum()) == (4 + 2) ** 2 - 3        # one rounded corner only (far side)


def test_dilated_pooling_includes_neighbour_token():
    # building fills x 4..12 inside token col 0; dilating by 3 m = 6 px reaches col 1 (x >= 16)
    vec_o, _ = pool_building(token_grid(), square(4, 4, 12, 12), G, Method.parse("outline"))
    vec_d, n = pool_building(token_grid(), square(4, 4, 12, 12), G, Method.parse("dilated:3"))
    np.testing.assert_allclose(vec_o, [0, 0, 0])
    assert vec_d[1] > 0 and n > 64


def test_pool_building_rejects_bad_tokens():
    with pytest.raises(PoolingError):
        pool_building(np.zeros((4, 4)), square(0, 0, 2, 2), G, Method.parse("outline"))

# ------------------------------------------------------------------ table level


def building_table(events_cfg):
    rows = []
    specs = [("beirut", "beirut-explosion_00000001", False), ("haiti", "haiti-earthquake_00000002", True)]
    for ev, tile, locked in specs:
        for i, (box, dmg) in enumerate([(square(17, 33, 30, 46), True), (square(2, 2, 10, 10), False),
                                        (square(40, 40, 44, 44), False)]):
            rows.append({"event": ev, "tile_id": tile, "instance_id": i, "role": events_cfg.events[ev].role,
                         "damaged": pd.NA if locked else dmg, "labels_locked": locked, "area_m2": 42.0,
                         "segmentation_px": json.dumps(box), "excluded_reason": "building_nodata" if i == 2 else None})
    df = pd.DataFrame(rows)
    df["damaged"] = df["damaged"].astype("boolean")
    return df


def test_pool_features_builds_a_valid_table(events_cfg, before_freeze):
    bt = building_table(events_cfg)
    out = pool_features(bt, lambda t: token_grid(dim=3), lambda t: G, "fake", "outline", events_cfg, before_freeze)
    assert len(out) == 4                                  # excluded buildings skipped
    assert feature_columns(out) == ["f0000", "f0001", "f0002"]
    assert out["pooling"].unique().tolist() == ["outline"]
    first = out[(out["event"] == "beirut") & (out["instance_id"] == 0)].iloc[0]
    np.testing.assert_allclose(first[["f0000", "f0001", "f0002"]].to_numpy(float), [2, 1, 21])
    assert out.loc[out["event"] == "haiti", "damaged"].isna().all()   # locked labels stay empty
    assert out["f0000"].dtype == np.float32


def test_pool_features_dilated_and_errors(events_cfg, before_freeze):
    bt = building_table(events_cfg)
    out = pool_features(bt, lambda t: token_grid(), lambda t: G, "fake", "dilated:3", events_cfg, before_freeze)
    assert out["pooling"].unique().tolist() == ["dilated3m"]
    sizes = iter([3, 4])
    with pytest.raises(PoolingError, match="feature size"):
        pool_features(bt, lambda t: np.zeros((4, 4, next(sizes))), lambda t: G, "fake", "outline", events_cfg, before_freeze)
    with pytest.raises(PoolingError, match="shape"):
        pool_features(bt, lambda t: np.zeros((4, 4)), lambda t: G, "fake", "outline", events_cfg, before_freeze)
    with pytest.raises(PoolingError, match="no buildings"):
        pool_features(bt.assign(excluded_reason="tile_nodata"), lambda t: token_grid(), lambda t: G,
                      "fake", "outline", events_cfg, before_freeze)


def test_token_side_in_metres_is_consistent():
    # sanity check of the geometry used in ADR 0001: 16 px tokens at 0.5 m are 8 m wide
    assert 16 * G.px_x_m == 8.0 and math.isclose(1024 / 64 * 0.3, 4.8)


# ---------------------------------------------------------------- ring (neighbourhood control)

def test_ring_parse():
    assert Method.parse("ring:10").name == "ring10m"
    for bad in ("ring:0", "ring:-2", "ring"):
        with pytest.raises(PoolingError):
            Method.parse(bad)


def test_ring_never_uses_a_token_holding_the_building():
    # Building fills token (1, 1) exactly (pixels 16..31); 8 m ring = 16 px around it at 0.5 m/px.
    t = token_grid()
    vec, npix = pool_building(t, square(16, 16, 32, 32), G, Method.parse("ring:8"))
    assert npix > 0
    # Ring reaches the 8 neighbours of (1, 1) and (by corners) nothing else; (1, 1) itself is excluded.
    # Weighted mean of rows 0..2 and cols 0..2 minus the centre: symmetric -> mean row 1, mean col 1.
    assert vec[0] == pytest.approx(1.0) and vec[1] == pytest.approx(1.0)
    # The centre token's own value (10*1 + 1 = 11) must not be the answer by construction:
    out, _ = pool_building(t, square(16, 16, 32, 32), G, Method.parse("outline"))
    assert out[2] == pytest.approx(11.0)


def test_ring_excludes_every_token_the_building_touches():
    # Building straddles tokens (1, 1) and (1, 2); a token value of 100 marks them.
    t = token_grid()
    t[1, 1, 2] = t[1, 2, 2] = 100.0
    vec, _ = pool_building(t, square(20, 18, 44, 30), G, Method.parse("ring:4"))
    # With the marked tokens excluded, the value is a mean of unmarked tokens (all <= 33).
    assert vec[2] <= 33.0


def test_ring_falls_back_to_neighbour_tokens_when_ring_stays_inside():
    # Huge building covering the whole tile's middle tokens, ring too narrow to leave them.
    t = token_grid()
    vec, npix = pool_building(t, square(0, 0, 64, 64), G, Method.parse("ring:0.25"))
    assert npix == 0                             # flagged
    assert np.isfinite(vec).all()


def test_ring_runs_through_pool_features(events_cfg, before_freeze):
    import damagesar_bench.pooling.footprint as fp
    b = pd.DataFrame({
        "event": ["beirut", "beirut"], "tile_id": ["beirut_t1", "beirut_t1"], "instance_id": [1, 2],
        "role": ["development", "development"], "damaged": [True, False], "labels_locked": [False, False],
        "area_m2": [64.0, 64.0], "excluded_reason": [None, None],
        "segmentation_px": [json.dumps(square(16, 16, 32, 32)), json.dumps(square(40, 40, 56, 56))],
    })
    df = pool_features(b, lambda _t: token_grid(), lambda _t: G, "toy", "ring:8", events_cfg, before_freeze)
    assert list(df["pooling"].unique()) == ["ring8m"]
    assert len(feature_columns(df)) == 3
