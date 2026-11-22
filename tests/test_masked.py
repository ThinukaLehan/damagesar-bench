import json

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.controls.masked import building_pixel_mask, mask_tile, masked_name, masked_reader
from damagesar_bench.pooling.footprint import TileGeometry, building_mask, shift_polygons

G = TileGeometry(64, 64, 0.5, 0.5)


def square(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def table(*polys, tile="t1"):
    return pd.DataFrame({"tile_id": [tile] * len(polys), "segmentation_px": [json.dumps(p) for p in polys]})


def test_name():
    assert masked_name(10) == "ring10m_masked"


def test_mask_is_union_of_footprints():
    a, b = square(4, 4, 12, 12), square(40, 40, 50, 52)
    m = building_pixel_mask(table(a, b), G, dilate_m=0.0)
    ma, ra, ca = building_mask(a, G)
    mb, rb, cb = building_mask(b, G)
    assert m.sum() == ma.sum() + mb.sum()
    assert m[ra:ra + ma.shape[0], ca:ca + ma.shape[1]][ma].all()
    assert not m[30, 30]


def test_dilation_grows_the_mask():
    t = table(square(20, 20, 30, 30))
    assert building_pixel_mask(t, G, dilate_m=1.0).sum() > building_pixel_mask(t, G, dilate_m=0.0).sum()


def test_shift_adds_the_shifted_footprint():
    p = square(20, 20, 30, 30)
    plain = building_pixel_mask(table(p), G, dilate_m=0.0)
    both = building_pixel_mask(table(p), G, dilate_m=0.0, shift=(4.0, 0.0))
    shifted, r0, c0 = building_mask(shift_polygons(p, 4.0, 0.0, G), G)
    assert both.sum() > plain.sum()
    assert both[r0:r0 + shifted.shape[0], c0:c0 + shifted.shape[1]][shifted].all()
    assert (both | plain).sum() == both.sum()


def test_footprint_at_the_edge_is_clipped():
    m = building_pixel_mask(table(square(-5, 58, 6, 70)), G, dilate_m=1.0)
    assert m.shape == (64, 64) and m[63, 0] and not m[0, 63]


def test_mask_tile_fills_with_background_median_and_keeps_nodata():
    img = np.full((64, 64), 0.2, dtype=np.float32)
    img[10:20, 10:20] = 0.9                      # the building
    img[0:4, :] = 0.0                            # no-data strip
    mask = np.zeros((64, 64), bool)
    mask[10:20, 10:20] = True
    mask[0:2, 0:5] = True                        # mask touching no-data
    out = mask_tile(img, mask)
    assert np.allclose(out[10:20, 10:20], 0.2)   # building replaced by the background median
    assert (out[0:4, :] == 0).all()              # no-data stays no-data
    assert np.allclose(out[30:, 30:], 0.2)       # rest unchanged
    assert img[10, 10] == pytest.approx(0.9)     # input not modified


def test_mask_tile_shape_mismatch():
    with pytest.raises(ValueError):
        mask_tile(np.zeros((4, 4)), np.zeros((5, 5), bool))


def test_reader_masks_only_tiles_with_buildings():
    img = np.full((64, 64), 0.3, dtype=np.float32)
    img[20:30, 20:30] = 1.0

    def read(t):
        return img.copy(), G

    r = masked_reader(read, table(square(20, 20, 30, 30)), dilate_m=0.0, shifts={"t1": (0.0, 0.0)})
    out, geom = r("t1")
    assert geom == G and out.max() == pytest.approx(0.3)
    same, _ = r("other")
    assert same.max() == pytest.approx(1.0)


def test_masked_ring_end_to_end_sees_no_building_pixels(tmp_path, events_cfg, before_freeze):
    from damagesar_bench.contracts import feature_columns
    from damagesar_bench.encoders import get_encoder
    from damagesar_bench.encoders.extract import extract

    g = TileGeometry(64, 64, 0.5, 0.5)
    rows = []
    for i, (box, dmg) in enumerate([(square(8, 8, 20, 20), True), (square(36, 36, 52, 52), False)]):
        rows.append({"event": "beirut", "tile_id": "beirut-explosion_00000000", "instance_id": i,
                     "role": events_cfg.events["beirut"].role, "damaged": dmg, "labels_locked": False,
                     "area_m2": 36.0, "segmentation_px": json.dumps(box), "excluded_reason": None})
    b = pd.DataFrame(rows)
    b["damaged"] = b["damaged"].astype("boolean")
    img = np.full((64, 64), 0.2, dtype=np.float32)
    img[8:20, 8:20] = 0.9
    img[36:52, 36:52] = 0.7

    def read(t):
        return img.copy(), g

    enc = get_encoder("patchstats")
    plain, _ = extract(b, read, enc, ["ring:10"], tmp_path / "plain", events_cfg, before_freeze)
    masked, _ = extract(b, masked_reader(read, b, dilate_m=1.0), enc, ["ring:10"], tmp_path / "masked",
                        events_cfg, before_freeze)
    cols = feature_columns(plain)
    background = enc.encode(np.full((64, 64), 0.2, dtype=np.float32)).reshape(-1, len(cols))[0]
    # every masked-ring vector equals the vector of a building-free tile; the unmasked ring does not
    np.testing.assert_allclose(masked[cols].to_numpy(), np.tile(background, (2, 1)), atol=1e-6)
    assert not np.allclose(plain[cols].to_numpy(), np.tile(background, (2, 1)), atol=1e-3)
