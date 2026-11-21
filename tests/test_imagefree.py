import numpy as np
import pandas as pd
import pytest

from damagesar_bench.contracts import feature_columns
from damagesar_bench.controls.imagefree import ControlError, image_free_features
from damagesar_bench.probe.logistic import run_probes


def table(rows):
    df = pd.DataFrame(rows)
    df["role"] = df["event"].map({"beirut": "development", "hawaii": "development", "marshall": "secondary"})
    df["labels_locked"] = False
    df["damaged"] = df["damaged"].astype("boolean")
    return df


def bld(event, tile, iid, x, y, w=10, h=10, res=0.5, damaged=False, excluded=None):
    return {"event": event, "tile_id": tile, "instance_id": iid, "bbox_x": x, "bbox_y": y, "bbox_w": w,
            "bbox_h": h, "res_m": res, "area_m2": w * h * res * res, "damaged": damaged,
            "excluded_reason": excluded}


def test_features_are_geometry_only(events_cfg, before_freeze):
    # Tile t1: three buildings; b3 is excluded (no-data) but still counts as a neighbour.
    b = table([bld("beirut", "t1", 1, 0, 0), bld("beirut", "t1", 2, 40, 0),
               bld("beirut", "t1", 3, 0, 300, excluded="building_nodata"),
               bld("beirut", "t2", 4, 0, 0, w=20, h=20)])
    df = image_free_features(b, events_cfg, before_freeze, radius_m=50)
    assert sorted(df["instance_id"]) == [1, 2, 4]                 # only kept buildings get a row
    assert set(df["encoder"]) == {"imagefree"} and set(df["pooling"]) == {"geometry"}
    f = df.set_index("instance_id")[feature_columns(df)]
    assert f.loc[1, "f0000"] == pytest.approx(np.log(25.0))      # 10 x 10 px at 0.5 m
    assert f.loc[1, "f0001"] == pytest.approx(np.log1p(3))        # three buildings in t1 (one excluded)
    # Centres 20 m apart (40 px at 0.5 m): within 50 m of each other; b3 is 150 m away.
    assert f.loc[1, "f0002"] == pytest.approx(np.log1p(1))
    assert f.loc[1, "f0003"] == pytest.approx(np.log(20.0))
    # Alone in its tile: no neighbours, nearest distance capped at 4 x radius.
    assert f.loc[4, "f0002"] == pytest.approx(0.0)
    assert f.loc[4, "f0003"] == pytest.approx(np.log(200.0))


def test_runs_through_the_normal_probes(events_cfg, before_freeze):
    rng = np.random.default_rng(0)
    rows, iid = [], 0
    for ev in ("beirut", "hawaii", "marshall"):
        for t in range(20):
            for k in range(10):
                big = rng.random() < 0.3
                iid += 1
                rows.append(bld(ev, f"{ev}_{t}", iid, rng.uniform(0, 900), rng.uniform(0, 900),
                                w=40 if big else 10, h=40 if big else 10, damaged=big))
    df = image_free_features(table(rows), events_cfg, before_freeze)
    scores, fits = run_probes(df, events_cfg, before_freeze)
    assert len(fits) >= 1 and scores["score"].between(0, 1).all()


def test_rejects_missing_columns_and_bad_radius(events_cfg):
    b = table([bld("beirut", "t1", 1, 0, 0)])
    with pytest.raises(ControlError):
        image_free_features(b.drop(columns="bbox_w"), events_cfg)
    with pytest.raises(ControlError):
        image_free_features(b, events_cfg, radius_m=0)
