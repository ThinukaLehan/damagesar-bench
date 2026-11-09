import datetime as dt

import pytest

from damagesar_bench import c2_metadata as c2
from damagesar_bench.config import load_events


def item(iid, bbox, date, **props):
    return {"id": iid, "bbox": bbox, "provider": "capella",
            "properties": {"datetime": f"{date}T10:00:00Z", **props}}


AREA = c2.BBox(35.50, 33.88, 35.54, 33.91)  # Beirut-ish


def test_event_dates_cover_all_configured_events():
    assert set(c2.EVENT_DATES) == set(load_events().events)
    for start, end, providers in c2.EVENT_DATES.values():
        assert dt.date.fromisoformat(start) <= dt.date.fromisoformat(end)
        assert set(providers) <= {"capella", "umbra"}


def test_bbox_rejects_inverted():
    with pytest.raises(ValueError):
        c2.BBox.of([1, 0, 0, 1])


def test_coverage_full_partial_none():
    assert c2.coverage(AREA, c2.BBox(35.0, 33.0, 36.0, 34.0)) == pytest.approx(1.0)
    half = c2.BBox(35.50, 33.88, 35.52, 33.91)
    assert c2.coverage(AREA, half) == pytest.approx(0.5)
    assert c2.coverage(AREA, c2.BBox(10, 10, 11, 11)) == 0.0


def test_union_bbox():
    u = c2.union_bbox([c2.BBox(0, 0, 1, 1), c2.BBox(2, -1, 3, 0.5)])
    assert u == c2.BBox(0, -1, 3, 1)
    with pytest.raises(ValueError):
        c2.union_bbox([])


def test_window_edges():
    assert c2.in_window("beirut", dt.date(2020, 7, 5))       # 30 days before
    assert not c2.in_window("beirut", dt.date(2020, 7, 4))
    assert c2.in_window("beirut", dt.date(2020, 11, 2))      # 90 days after
    assert not c2.in_window("beirut", dt.date(2020, 11, 3))
    # range events count from the end date
    assert c2.in_window("la_palma", dt.date(2022, 3, 13))


def test_extract_geometry_normalises():
    g = c2.extract_geometry({
        "view:incidence_angle": "38.2", "sar:observation_direction": "Left",
        "sat:orbit_state": "DESCENDING", "sar:instrument_mode": "spotlight",
        "capella:look_angle": 34.0,
    })
    assert g["incidence_deg"] == pytest.approx(38.2)
    assert g["look_direction"] == "left"
    assert g["orbit_state"] == "descending"
    assert g["look_angle_deg"] == 34.0
    assert g["mode"] == "spotlight"


def test_look_angle_is_not_used_as_incidence():
    g = c2.extract_geometry({"capella:look_angle": 30.0})
    assert g["incidence_deg"] is None and g["look_angle_deg"] == 30.0


def test_match_ranks_by_coverage_then_post_event_then_closeness():
    items = [
        item("before", [35.0, 33.0, 36.0, 34.0], "2020-08-01"),
        item("late", [35.0, 33.0, 36.0, 34.0], "2020-09-30"),
        item("early", [35.0, 33.0, 36.0, 34.0], "2020-08-05"),
        item("partial", [35.50, 33.88, 35.52, 33.91], "2020-08-04"),
        item("far", [10, 10, 11, 11], "2020-08-05"),
        item("too_late", [35.0, 33.0, 36.0, 34.0], "2021-08-05"),
    ]
    rows = c2.match_event("beirut", AREA, items)
    assert [r["item_id"] for r in rows] == ["early", "late", "before", "partial"]
    assert rows[0]["days_after"] == 1 and rows[2]["days_after"] == -3


def test_match_handles_3d_bbox():
    rows = c2.match_event("beirut", AREA, [item("z", [35.0, 33.0, 0, 36.0, 34.0, 100], "2020-08-05")])
    assert rows and rows[0]["coverage"] == pytest.approx(1.0)


def test_status_and_summary():
    full = {"incidence_deg": 40.0, "look_direction": "right", "orbit_state": "ascending"}
    assert c2.status_of(None) == "dilated-only"
    assert c2.status_of(full) == "geometry-ready"
    assert c2.status_of({**full, "orbit_state": None}) == "match-only"

    items = [
        item("a", [35.0, 33.0, 36.0, 34.0], "2020-08-05", **{
            "view:incidence_angle": 40.0, "sar:observation_direction": "right", "sat:orbit_state": "ascending"}),
        item("b", [35.0, 33.0, 36.0, 34.0], "2020-08-06", **{
            "view:incidence_angle": 30.0, "sar:observation_direction": "left", "sat:orbit_state": "descending"}),
    ]
    s = c2.summarise_event("beirut", c2.match_event("beirut", AREA, items))
    assert s["status"] == "geometry-ready"
    assert s["best_item_id"] == "a"
    assert s["geometry_consistent"] is False
    assert s["incidence_range_good"] == "30.0-40.0"

    empty = c2.summarise_event("noto", [])
    assert empty["status"] == "dilated-only" and empty["best_item_id"] is None
