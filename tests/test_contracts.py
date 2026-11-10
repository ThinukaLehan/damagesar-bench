import numpy as np
import pandas as pd
import pytest

from damagesar_bench import contracts as C


def features(events_cfg, today_locked=True, dim=4, n=3):
    rows = []
    for ev in ("beirut", "haiti", "marshall"):
        role = events_cfg.events[ev].role
        locked = role == "confirmatory" and today_locked
        for i in range(n):
            r = {"event": ev, "tile_id": f"{ev}-x_00000001", "instance_id": i, "role": role,
                 "damaged": pd.NA if locked else bool(i % 2), "labels_locked": locked,
                 "area_m2": 50.0 + i, "encoder": "fake", "pooling": "outline"}
            r.update({C.feature_name(k): float(k + i) for k in range(dim)})
            rows.append(r)
    df = pd.DataFrame(rows)
    df["damaged"] = df["damaged"].astype("boolean")
    return df


def test_feature_names():
    assert C.feature_name(0) == "f0000" and C.feature_name(123) == "f0123"
    with pytest.raises(ValueError):
        C.feature_name(C.MAX_FEATURES)
    df = pd.DataFrame(columns=["f0001", "event", "f0000", "fx", "f00001"])
    assert C.feature_columns(df) == ["f0000", "f0001"]


def test_join_split_events():
    assert C.join_events(["libya", "beirut", "beirut"]) == "beirut+libya"
    assert C.split_events("beirut+libya") == {"beirut", "libya"}
    assert C.split_events("") == set()


def test_valid_building_features(events_cfg, before_freeze, after_freeze):
    assert C.validate_building_features(features(events_cfg), events_cfg, before_freeze) == \
        ["f0000", "f0001", "f0002", "f0003"]
    C.validate_building_features(features(events_cfg, today_locked=False), events_cfg, after_freeze)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.drop(columns=["area_m2"]), "missing columns"),
    (lambda d: d.iloc[0:0], "empty"),
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicate"),
    (lambda d: d.assign(role="development"), "role of"),
    (lambda d: d.assign(event="atlantis"), "unknown events"),
    (lambda d: d.drop(columns=["f0001"]), "contiguous"),
    (lambda d: d.drop(columns=[c for c in d if c.startswith("f0")]), "no feature columns"),
    (lambda d: d.assign(f0000=np.nan), "NaN"),
    (lambda d: d.assign(area_m2=0.0), "area_m2"),
    (lambda d: d.assign(encoder=""), "encoder"),
    (lambda d: d.assign(instance_id=None), "key columns"),
])
def test_building_features_rejects(events_cfg, before_freeze, mutate, msg):
    with pytest.raises(C.ContractError, match=msg):
        C.validate_building_features(mutate(features(events_cfg)), events_cfg, before_freeze)


def test_building_features_label_rules(events_cfg, before_freeze, after_freeze):
    d = features(events_cfg)
    leak = d.copy()
    leak.loc[leak["event"] == "haiti", "damaged"] = True
    with pytest.raises(C.ContractError, match="disagrees|confirmatory labels"):
        C.validate_building_features(leak, events_cfg, before_freeze)
    leak2 = leak.copy()
    leak2["labels_locked"] = False
    with pytest.raises(C.ContractError, match="disagrees"):
        C.validate_building_features(leak2, events_cfg, before_freeze)
    missing = d.copy()
    missing.loc[missing["event"] == "beirut", "damaged"] = pd.NA
    with pytest.raises(C.ContractError, match="missing labels"):
        C.validate_building_features(missing, events_cfg, before_freeze)
    # a table built before the freeze is stale after it (labels_locked no longer matches)
    with pytest.raises(C.ContractError, match="disagrees"):
        C.validate_building_features(d, events_cfg, after_freeze)


def test_two_poolings_may_share_buildings(events_cfg, before_freeze):
    d = features(events_cfg)
    C.validate_building_features(pd.concat([d, d.assign(pooling="dilated3m")]), events_cfg, before_freeze)


def scores(events_cfg):
    rows = []
    for ev, train in (("beirut", "hawaii+marshall"), ("hawaii", "beirut+marshall")):
        for i in range(4):
            rows.append({"event": ev, "tile_id": f"{ev}-t_1", "instance_id": i,
                         "role": events_cfg.events[ev].role, "damaged": bool(i % 2),
                         "score": 0.1 * i, "encoder": "fake", "pooling": "outline", "train_events": train})
    df = pd.DataFrame(rows)
    df["damaged"] = df["damaged"].astype("boolean")
    return df


def test_valid_fold_scores(events_cfg, before_freeze):
    C.validate_fold_scores(scores(events_cfg), events_cfg, before_freeze)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: d.assign(train_events="beirut+marshall"), "own training set"),
    (lambda d: d.assign(train_events="haiti+marshall"), "confirmatory events .* before the freeze"),
    (lambda d: d.assign(train_events="atlantis"), "unknown training events"),
    (lambda d: d.assign(score=np.inf), "finite"),
    (lambda d: d.assign(train_events=""), "train_events"),
    (lambda d: d.assign(train_events=["hawaii", "marshall"] * 2 + ["beirut"] * 4), "more than one training set"),
])
def test_fold_scores_rejects(events_cfg, before_freeze, mutate, msg):
    with pytest.raises(C.ContractError, match=msg):
        C.validate_fold_scores(mutate(scores(events_cfg)), events_cfg, before_freeze)


def test_fold_scores_confirmatory_training_allowed_after_freeze(events_cfg, after_freeze):
    d = scores(events_cfg).assign(train_events=lambda x: np.where(x["event"] == "beirut", "haiti+hawaii", "beirut+haiti"))
    C.validate_fold_scores(d, events_cfg, after_freeze)


def metrics_row(**kw):
    r = {"event": "beirut", "role": "development", "encoder": "fake", "pooling": "outline", "n": 100,
         "n_damaged": 10, "auroc": 0.7, "ap": 0.3, "auroc_se": 0.05, "boot_lower": 0.6,
         "boot_upper": 0.8, "boot_reliable": True}
    r.update(kw)
    return pd.DataFrame([r])


def test_event_metrics(events_cfg):
    C.validate_event_metrics(metrics_row(), events_cfg)
    for kw, msg in [({"n_damaged": 0}, "both classes"), ({"n_damaged": 100}, "both classes"),
                    ({"auroc": 1.2}, "auroc"), ({"boot_lower": 0.9}, "boot_lower"), ({"role": "secondary"}, "role of")]:
        with pytest.raises(C.ContractError, match=msg):
            C.validate_event_metrics(metrics_row(**kw), events_cfg)
    with pytest.raises(C.ContractError, match="duplicate"):
        C.validate_event_metrics(pd.concat([metrics_row(), metrics_row()]), events_cfg)
    with pytest.raises(C.ContractError, match="missing"):
        C.validate_event_metrics(metrics_row().drop(columns=["ap"]), events_cfg)
