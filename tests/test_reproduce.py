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
    assert len([i for i in ids if i.startswith("auroc.")]) == 75          # Table III (20) + mean; p03 5 x (5 + mean);
    #                                                                       p04 3 x (5 + mean); g01 C2 5 + mean
    assert {"gate1.mean_gain", "gate1.n_better", "gate1.passed"} <= ids
    assert len([i for i in ids if i.startswith("tokens.")]) == 22         # Table II, 11 events x 2
    assert len([i for i in ids if i.startswith("cmp.")]) == 64            # Table IV 4 + p03 6 + p04 3 + g01 3, x 4


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
    sub = rep[rep["id"].str.startswith(("auroc.", "cmp.")) & ~rep["id"].isin(P03_IDS(exp) | LATER_IDS(exp))]
    assert len(sub) == 37 and (sub["status"] == "PASS").all(), sub[sub["status"] != "PASS"]


P03_LABELS = ("DINOv3 vs ImageNet (outline)", "building vs surroundings (DINOv3)",
              "building vs surroundings (ImageNet)", "DINOv3 vs image-free", "ImageNet vs image-free",
              "dilated3m vs outline (ImageNet)")


def P03_IDS(exp):
    ids = exp["id"]
    return set(ids[ids.str.contains("imagenet|imagefree|ring10m") | ids.str.startswith(
        tuple(f"cmp.{lab}." for lab in P03_LABELS))])


def test_the_recorded_p03_values_pass_their_own_check():
    exp = load_expected(ROOT / "paper" / "expected_values.yaml")
    ev = ["beirut", "congo", "hawaii", "la_palma", "libya"]
    rows = {("dinov3_vitb16@native", "ring10m"): [0.654, 0.853, 0.549, 0.533, 0.503],
            ("imagenet_vitb16@native", "outline"): [0.465, 0.781, 0.716, 0.755, 0.600],
            ("imagenet_vitb16@native", "dilated3m"): [0.451, 0.821, 0.710, 0.780, 0.603],
            ("imagenet_vitb16@native", "ring10m"): [0.385, 0.697, 0.727, 0.815, 0.542],
            ("imagefree", "geometry"): [0.661, 0.269, 0.405, 0.544, 0.272]}
    m = pd.DataFrame([{"encoder": e, "pooling": p, "event": x, "role": "development", "auroc": a}
                      for (e, p), v in rows.items() for x, a in zip(ev, v)])
    c = pd.DataFrame([
        {"label": P03_LABELS[0], "estimate": -0.04333542712, "lower": -0.2460759583, "upper": 0.1594051041},
        {"label": P03_LABELS[1], "estimate": 0.001901997863, "lower": -0.05053086865, "upper": 0.05433486437},
        {"label": P03_LABELS[2], "estimate": 0.030, "lower": -0.048, "upper": 0.109},
        {"label": P03_LABELS[3], "estimate": 0.190, "lower": -0.102, "upper": 0.482},
        {"label": P03_LABELS[4], "estimate": 0.234, "lower": -0.094, "upper": 0.561},
        {"label": P03_LABELS[5], "estimate": 0.009, "lower": -0.018, "upper": 0.037}]).assign(verdict="Inconclusive")
    rep = compare(exp, collect_actual(event_metrics=m, comparisons=c))
    sub = rep[rep["id"].isin(P03_IDS(exp))]
    assert len(sub) == 54 and (sub["status"] == "PASS").all(), sub[sub["status"] != "PASS"]


def test_errors_and_environment(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text("data: {}\n")
    with pytest.raises(ReproduceError):
        load_expected(p)
    env = environment()
    assert env["python"] and "numpy" in env and "transformers" in env


def test_offsets_are_collected_and_checked():
    exp = load_expected(ROOT / "paper" / "expected_values.yaml")
    off_ids = [i for i in exp["id"] if i.startswith("offsets.")]
    assert len(off_ids) == 36                                         # 6 events x 6 numbers
    osum = pd.DataFrame([{"event": "congo", "median_offset_m": 7.880029163, "median_offset_confident_m": 7.54209715,
                          "median_offset_tokens": 1.41421955, "share_confident": 0.8913043478,
                          "share_beats_null": 0.7282608696, "direction_R": 0.8437229919}])
    got = collect_actual(offsets=osum)
    assert got["offsets.share_beats_null.congo"] == 0.728
    rep = compare(exp[exp["id"].str.endswith(".congo") & exp["id"].str.startswith("offsets.")], got)
    assert (rep["status"] == "PASS").all()


def test_other_encoders_get_their_own_ids():
    from damagesar_bench.reproduce import _setting_key

    assert _setting_key("dinov3_vitb16@native") == "native"
    assert _setting_key("dinov3_vitb16@0.3m") == "0.3m"
    assert _setting_key("imagenet_vitb16@native") == "imagenet.native"
    assert _setting_key("imagefree") == "imagefree"
    assert _setting_key("dinov3_vitl16@native") == "dinov3_vitl16.native"
    assert _setting_key("dinov3_vitl16_sat@native") == "dinov3_vitl16_sat.native"
    assert _setting_key("dinov2_vitb14@native") == "dinov2_vitb14.native"
    m = pd.DataFrame({
        "encoder": ["dinov3_vitb16@native"] * 2 + ["imagenet_vitb16@native"] * 2 + ["imagefree"] * 2,
        "pooling": ["outline", "outline", "outline", "outline", "geometry", "geometry"],
        "event": ["beirut", "congo"] * 3, "role": ["development"] * 6,
        "auroc": [0.646, 0.849, 0.465, 0.781, 0.661, 0.269]})
    got = collect_actual(event_metrics=m)
    assert got["auroc.native.outline.beirut"] == 0.646  # not overwritten by ImageNet
    assert got["auroc.imagenet.native.outline.beirut"] == 0.465
    assert got["auroc.imagefree.geometry.congo"] == 0.269
    assert got["auroc.native.outline.mean5"] == round((0.646 + 0.849) / 2, 3)
    assert got["auroc.imagenet.native.outline.mean5"] == 0.623


P04_LABELS = ("DINOv3-L satellite vs web", "DINOv3 ViT-L vs ViT-B (web)", "DINOv3 vs DINOv2 (ViT-B)")
G01_LABELS = ("C2 matched vs outline", "C2 matched vs dilated 3 m", "C2 matched vs ring 10 m")


def LATER_IDS(exp):
    """Ids added by p04 and g01 (22 Nov)."""
    ids = exp["id"]
    return set(ids[ids.str.contains("dinov2_vitb14|dinov3_vitl16|matched\\.") | ids.str.startswith(
        tuple(f"cmp.{lab}." for lab in P04_LABELS + G01_LABELS))])


def test_the_recorded_p04_and_g01_values_pass_their_own_check():
    exp = load_expected(ROOT / "paper" / "expected_values.yaml")
    ev = ["beirut", "congo", "hawaii", "la_palma", "libya"]
    rows = {("dinov2_vitb14@native", "outline"): [0.437, 0.889, 0.591, 0.545, 0.386],
            ("dinov3_vitl16@native", "outline"): [0.709, 0.878, 0.586, 0.634, 0.437],
            ("dinov3_vitl16_sat@native", "outline"): [0.742, 0.827, 0.755, 0.856, 0.646],
            ("dinov3_vitb16@native", "matched"): [0.636, 0.855, 0.493, 0.643, 0.509]}
    m = pd.DataFrame([{"encoder": e, "pooling": p, "event": x, "role": "development", "auroc": a}
                      for (e, p), v in rows.items() for x, a in zip(ev, v)])
    c = pd.DataFrame([
        {"label": P04_LABELS[0], "estimate": 0.117, "lower": -0.033, "upper": 0.266},
        {"label": P04_LABELS[1], "estimate": 0.029, "lower": -0.040, "upper": 0.097},
        {"label": P04_LABELS[2], "estimate": 0.051, "lower": -0.097, "upper": 0.198},
        {"label": G01_LABELS[0], "estimate": 0.007, "lower": -0.018, "upper": 0.033},
        {"label": G01_LABELS[1], "estimate": 0.010, "lower": -0.021, "upper": 0.041},
        {"label": G01_LABELS[2], "estimate": 0.009, "lower": -0.067, "upper": 0.086}]).assign(verdict="Inconclusive")
    got = collect_actual(event_metrics=m, comparisons=c)
    got.update({"gate1.mean_gain": 0.007, "gate1.n_better": 3, "gate1.passed": False})
    rep = compare(exp, got)
    sub = rep[rep["id"].isin(LATER_IDS(exp)) | rep["id"].str.startswith("gate1.")]
    assert len(sub) == 24 + 24 + 3 and (sub["status"] == "PASS").all(), sub[sub["status"] != "PASS"]
