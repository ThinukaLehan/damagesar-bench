import json

import numpy as np
import pandas as pd
import pytest

from damagesar_bench.baselines.texture import (ENCODER, GLCM_NAMES, LEVELS, building_features, feature_columns,
                                               glcm, glcm_stats, leave_one_event_out, texture_table)
from damagesar_bench.pooling.footprint import TileGeometry


def test_glcm_matches_scikit_image_on_a_full_window():
    skf = pytest.importorskip("skimage.feature")
    rng = np.random.default_rng(1)
    q = rng.integers(0, LEVELS, size=(30, 40))
    ours = glcm(q, np.ones(q.shape, bool))
    sk = skf.graycomatrix(q.astype(np.uint8), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=LEVELS,
                          symmetric=True, normed=False)[:, :, 0, :].sum(axis=2).astype(float)
    assert np.allclose(ours, sk / sk.sum())
    for name in ("contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"):
        p4 = skf.graycomatrix(q.astype(np.uint8), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=LEVELS,
                              symmetric=True, normed=False).sum(axis=3, keepdims=True).astype(float)
        p4 /= p4.sum()
        assert glcm_stats(ours)[name.lower()] == pytest.approx(float(skf.graycoprops(p4, name)[0, 0]), rel=1e-9)


def test_glcm_counts_only_pairs_inside_the_mask():
    q = np.array([[1, 1, 5], [1, 1, 5]])
    m = np.array([[True, True, False], [True, True, False]])
    p = glcm(q, m, levels=8)
    assert p[1, 1] == pytest.approx(1.0)               # the 5s are outside: only 1-1 pairs remain
    assert glcm_stats(glcm(q, np.zeros_like(m), 8))["contrast"] != glcm_stats(p)["contrast"]   # empty -> NaN
    assert np.isnan(glcm_stats(glcm(q, np.zeros_like(m), 8))["entropy"])


def test_flat_region_has_no_contrast_and_full_homogeneity():
    s = glcm_stats(glcm(np.full((6, 6), 3), np.ones((6, 6), bool)))
    assert s["contrast"] == 0 and s["homogeneity"] == pytest.approx(1) and s["asm"] == pytest.approx(1)
    assert s["entropy"] == pytest.approx(0)


def test_building_features_separate_footprint_and_context():
    img = np.full((100, 100), 0.1, np.float32)
    img[40:60, 40:60] = 0.9                              # bright building on a dark background
    geom = TileGeometry(100, 100, 1.0, 1.0)
    f = building_features([[40, 40, 60, 40, 60, 60, 40, 60]], img, geom, tile_median=0.1)
    assert f["fp_mean"] == pytest.approx(0.9) and f["fp_bright"] == pytest.approx(1.0)
    assert 0.1 < f["ctx_mean"] < 0.9 and f["fp_rel_tile"] == pytest.approx(9.0)
    assert len(f) == 2 * (9 + len(GLCM_NAMES))


def _table(n=90, seed=0):
    rng = np.random.default_rng(seed)
    rows, tiles = [], {}
    for ev in ["a", "b", "c"]:
        img = np.full((200, 200), 0.1, np.float32)
        for i in range(n):
            r, c = divmod(i, 10)
            y0, x0 = 5 + r * 20, 5 + c * 19
            dmg = bool(rng.random() < 0.4)
            img[y0:y0 + 8, x0:x0 + 8] = rng.uniform(0.6, 1.0, (8, 8)) if dmg else 0.3
            rows.append({"event": ev, "tile_id": f"{ev}_t", "instance_id": i, "role": "development",
                         "damaged": dmg, "excluded_reason": None,
                         "segmentation_px": json.dumps([[x0, y0, x0 + 8, y0, x0 + 8, y0 + 8, x0, y0 + 8]])})
        tiles[f"{ev}_t"] = img
    b = pd.DataFrame(rows)
    b["damaged"] = b["damaged"].astype("boolean")
    return b, lambda t: (tiles[t], TileGeometry(200, 200, 1.0, 1.0))


def test_texture_table_and_leave_one_event_out():
    b, read = _table()
    t = texture_table(b, read)
    assert len(t) == len(b) and set(feature_columns("footprint_ctx32m")) <= set(t.columns)
    for pooling in ("footprint", "footprint_ctx32m"):
        fs = leave_one_event_out(t, ["a", "c"], pooling)
        assert set(fs["event"]) == {"a", "c"} and fs["encoder"].eq(ENCODER).all() and fs["pooling"].eq(pooling).all()
        assert (fs.loc[fs["event"] == "a", "train_events"] == "b+c").all()     # never trains on the held-out event
        g = fs[fs["event"] == "a"]
        y = g["damaged"].astype(bool).to_numpy()
        assert g.loc[y, "score"].min() > g.loc[~y, "score"].max()           # bright speckle vs flat grey


def test_unknown_setting_is_refused():
    with pytest.raises(ValueError):
        feature_columns("ring")
