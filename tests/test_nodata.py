import numpy as np
import pytest

from damagesar_bench.config import NoDataRule
from damagesar_bench.data.nodata import (
    building_is_excluded,
    nodata_fraction,
    nodata_mask,
    tile_is_excluded,
)

RULE = NoDataRule()  # 0 = no-data, tile >= 95%, building >= 50%


def test_mask_2d_and_3d():
    sar = np.array([[0, 5], [0, 0]], dtype=np.uint8)
    assert nodata_mask(sar, RULE).tolist() == [[True, False], [True, True]]
    three = np.stack([sar, sar * 0 + 7])  # second band non-zero everywhere
    assert not nodata_mask(three, RULE).any()


def test_mask_rejects_1d():
    with pytest.raises(ValueError):
        nodata_mask(np.zeros(4), RULE)


def test_tile_threshold_is_inclusive():
    mask = np.zeros(100, dtype=bool)
    mask[:95] = True
    assert tile_is_excluded(mask.reshape(10, 10), RULE)  # exactly 95% -> excluded
    mask[94] = False
    assert not tile_is_excluded(mask.reshape(10, 10), RULE)  # 94% -> kept


def test_building_threshold():
    mask = np.zeros((10, 10), dtype=bool)
    fp = np.zeros((10, 10), dtype=bool)
    fp[0:2, 0:5] = True  # 10-pixel footprint
    mask[0, 0:6] = True  # 5 of its pixels are no-data (plus one outside)
    assert nodata_fraction(mask, fp) == pytest.approx(0.5)
    assert building_is_excluded(mask, fp, RULE)  # 50% -> dropped
    mask[0, 4] = False
    assert not building_is_excluded(mask, fp, RULE)  # 40% -> kept


def test_empty_footprint_is_excluded():
    mask = np.zeros((4, 4), dtype=bool)
    assert building_is_excluded(mask, np.zeros((4, 4), dtype=bool), RULE)


def test_region_shape_must_match():
    with pytest.raises(ValueError):
        nodata_fraction(np.zeros((4, 4), dtype=bool), np.zeros((3, 3), dtype=bool))
