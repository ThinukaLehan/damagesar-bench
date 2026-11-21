from pathlib import Path

import pandas as pd
import pytest

from damagesar_bench.reproduce import (
    ReproduceError, collect_actual, compare, environment, load_expected, verdict)

ROOT = Path(__file__).resolve().parents[1]


def test_expected_values_file_loads_and_covers_the_paper_tables():
    exp = load_expected(ROOT / "paper" / "expected_values.yaml")
    ids = set(exp["id"])
    assert {"buildings.total", "tiles.total", "auroc.native.outline.congo", "tokens.lt4.target.beirut",
            "cmp.0.3m vs native (outline).verdict", "all_converged"} <= ids
    assert len([i for i in ids if i.startswith("auroc.")]) == 21          # Table III (20) + mean
    assert len([i for i in ids if i.startswith("tokens.")]) == 22         # Table II, 11 events x 2
    assert len([i for i in ids if i.startswith("cmp.")]) == 16            # Table IV, 4 comparisons x 4


def test_collect_and_compare_on_a_tiny_run():
    tiles = pd.DataFrame({"n_tiles": [3, 1], "n_excluded_nodata": [1, 0], "n_problem_tiles": [0, 0]})
    b = pd.DataFrame({
        "event": ["beirut"] * 4 + ["haiti"],
        "role": ["development"] * 4 + ["confirmatory"],
        "excluded_reason": [None, None, None, "tile_nodata", None],
        "damaged": pd.array([True, False, False, True, pd.NA], dtype="boolean"),
        "reported_area_is_bbox": [True, True, False, True, True],
    })
    summ = pd.DataFrame({"event": ["beirut", "haiti"], "role": ["development", "confirmatory"],
                         "pct_lt_4_tokens_native": [41.9, 88.8], "pct_lt_4_tokens_target": [13.3, None],
                         "raster_agreement": [0.9995, None]})
    m = pd.DataFrame({"encoder": ["dinov3_vitb16@native", "dinov3_vitb16@0.3m"], "pooling": ["outline", "outline"],
                      "event": ["beirut", "beirut"], "auroc": [0.6462, 0.5312]})
    c = pd.DataFrame({"label": ["x"], "estimate": [-0.0296], "lower": [-0.1216], "upper": [0.0623],
                      "verdict": ["Inconclusive"]})
    fits = pd.DataFrame({"n_iter": [143, 150]})
    got = collect_actual(tiles, b, summ, m, c, fits)
    assert got["tiles.total"] == 4 and got["tiles.excluded_nodata"] == 1
    assert got["buildings.total"] == 5 and got["buildings.kept"] == 4
    assert got["buildings.excluded.tile_nodata"] == 1 and got["buildings.excluded.empty_footprint"] == 0
    assert got["positives.beirut"] == 1 and got["kept.beirut"] == 3 and "positives.haiti" not in got  # locked
    assert got["prevalence.min_unlocked_pct"] == pytest.approx(33.3)
    assert got["tokens.lt4.native.haiti"] == 88.8 and "tokens.lt4.target.haiti" not in got
    assert got["raster_agreement.min_unlocked"] == 0.9995
    assert got["auroc.native.outline.beirut"] == 0.646 and got["auroc.0.3m.outline.beirut"] == 0.531
    assert got["auroc.native.outline.mean5"] == 0.646
    assert got["cmp.x.lower"] == -0.122 and got["cmp.x.verdict"] == "Inconclusive"
    assert got["all_converged"] is True

    exp = pd.DataFrame([
        {"id": "tiles.total", "expected": 4, "kind": "exact", "tol": None, "paper": "t"},
        {"id": "auroc.native.outline.beirut", "expected": 0.649, "kind": "abs", "tol": 0.005, "paper": "t"},
        {"id": "auroc.native.outline.congo", "expected": 0.849, "kind": "abs", "tol": 0.005, "paper": "t"},
        {"id": "raster_agreement.min_unlocked", "expected": 0.999, "kind": "min", "tol": None, "paper": "t"},
        {"id": "cmp.x.verdict", "expected": "Not a worthwhile gain", "kind": "text", "tol": None, "paper": "t"},
        {"id": "area_field_is_bbox.share", "expected": None, "kind": "info", "tol": None, "paper": "t"},
    ])
    rep = compare(exp, got).set_index("id")["status"]
    assert rep["tiles.total"] == "PASS" and rep["auroc.native.outline.beirut"] == "PASS"
    assert rep["auroc.native.outline.congo"] == "MISSING" and rep["raster_agreement.min_unlocked"] == "PASS"
    assert rep["cmp.x.verdict"] == "FAIL" and rep["area_field_is_bbox.share"] == "INFO"
    v = verdict(compare(exp, got))
    assert v.startswith("NOT REPRODUCED") and "3 of 5" in v and "INFO 1" in v


def test_the_recorded_p02b_values_pass_their_own_check():
    # The 20 Nov numbers, as written into the paper, must pass the expected-values file (guards typos).
    exp = load_expected(ROOT / "paper" / "expected_values.yaml")
    rows = {"beirut": (0.6462258038, 0.6512450853, 0.5312248617, 0.5434026465),
            "congo": (0.849459652, 0.8530131825, 0.7619413614, 0.7739781721),
            "hawaii": (0.5035107037, 0.5008259447, 0.5759812554, 0.5655105781),
            "la_palma": (0.6038068559, 0.6045923386, 0.5984189416, 0.5983408254),
            "libya": (0.4984028253, 0.4783587103, 0.4857788677, 0.4973416578)}
    m = pd.DataFrame([{"encoder": f"dinov3_vitb16@{r}", "pooling": p, "event": ev, "auroc": v[i]}
                      for ev, v in rows.items()
                      for i, (r, p) in enumerate([("native", "outline"), ("native", "dilated3m"),
                                                  ("0.3m", "outline"), ("0.3m", "dilated3m")])])
    c = pd.DataFrame([
        {"label": "0.3m vs native (outline)", "estimate": -0.02961211058, "lower": -0.1215720993,
         "upper": 0.06234787812, "verdict": "Inconclusive"},
        {"label": "0.3m vs native (dilated3m)", "estimate": -0.02189227631, "lower": -0.1098308908,
         "upper": 0.0660463382, "verdict": "Inconclusive"},
        {"label": "dilated3m vs outline (native)", "estimate": -0.002674115876, "lower": -0.01527039396,
         "upper": 0.009922162212, "verdict": "Not a worthwhile gain"},
        {"label": "dilated3m vs outline (0.3m)", "estimate": 0.005045718398, "lower": -0.007513302966,
         "upper": 0.01760473976, "verdict": "Not a worthwhile gain"}])
    rep = compare(exp, collect_actual(event_metrics=m, comparisons=c))
    sub = rep[rep["id"].str.startswith(("auroc.", "cmp."))]
    assert (sub["status"] == "PASS").all(), sub[sub["status"] != "PASS"]


def test_errors_and_environment(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("data: {}\n")
    with pytest.raises(ReproduceError):
        load_expected(p)
    env = environment()
    assert env["python"] and "numpy" in env and "transformers" in env
