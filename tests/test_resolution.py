import pandas as pd
import pytest
from test_encoders import CountingEncoder, buildings_for  # synthetic tiles and buildings

from damagesar_bench.encoders.extract import read_bright_tile
from damagesar_bench.experiments.resolution import (
    extract_at_resolutions, resolution_comparisons, resolution_label, runs_table, setting_name)


def test_names():
    assert resolution_label(None) == "native" and resolution_label(0.3) == "0.3m"
    assert setting_name("dinov3_vitb16", None) == "dinov3_vitb16@native"
    assert setting_name("dinov3_vitb16", 0.3) == "dinov3_vitb16@0.3m"


def test_extract_at_two_resolutions(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    reader = lambda t: read_bright_tile(bright_root, t)  # noqa: E731
    feats, runs = extract_at_resolutions(b, reader, CountingEncoder(), ["outline", "dilated:3"], [None, 0.25],
                                         tmp_path / "f", events_cfg, before_freeze)
    assert sorted(feats["encoder"].unique()) == ["counting@0.25m", "counting@native"]
    assert sorted(feats["pooling"].unique()) == ["dilated3m", "outline"]
    assert len(feats) == 2 * 2 * 4                  # resolutions x poolings x Beirut buildings (Haiti skipped)
    assert (tmp_path / "f" / "native" / "manifest.json").exists()
    assert (tmp_path / "f" / "0.25m" / "manifest.json").exists()
    t = runs_table(runs)
    assert list(t["resolution"]) == ["native", "0.25m"] and list(t["tiles"]) == [2, 2]
    assert list(t["buildings"]) == [4, 4]
    # same buildings at both resolutions; the synthetic SAR tile is constant, so resampling must not
    # change its patch statistics (a check that resampling keeps values in place)
    a = feats[(feats.encoder == "counting@native") & (feats.pooling == "outline")].sort_values(["tile_id", "instance_id"])
    c = feats[(feats.encoder == "counting@0.25m") & (feats.pooling == "outline")].sort_values(["tile_id", "instance_id"])
    assert list(a.instance_id) == list(c.instance_id)
    pd.testing.assert_series_equal(a["f0000"].reset_index(drop=True), c["f0000"].reset_index(drop=True),
                                   check_exact=False, atol=1e-6)


def test_extract_at_resolutions_validation(tmp_path, bright_root, events_cfg, before_freeze):
    b = buildings_for(bright_root, events_cfg)
    reader = lambda t: read_bright_tile(bright_root, t)  # noqa: E731
    with pytest.raises(ValueError, match="at least one"):
        extract_at_resolutions(b, reader, CountingEncoder(), ["outline"], [], tmp_path, events_cfg, before_freeze)
    with pytest.raises(ValueError, match="distinct"):
        extract_at_resolutions(b, reader, CountingEncoder(), ["outline"], [0.3, 0.3], tmp_path, events_cfg, before_freeze)


def test_resolution_comparisons():
    comps = resolution_comparisons("dinov3_vitb16", [None, 0.3], ["outline", "dilated:3"])
    labels = [c.label for c in comps]
    assert labels == ["0.3m vs native (outline)", "0.3m vs native (dilated3m)",
                      "dilated3m vs outline (native)", "dilated3m vs outline (0.3m)"]
    first = comps[0]
    assert first.a == ("dinov3_vitb16@0.3m", "outline") and first.b == ("dinov3_vitb16@native", "outline")
    with pytest.raises(ValueError, match="baseline"):
        resolution_comparisons("x", [0.3], ["outline"], baseline=None)
    assert resolution_comparisons("x", [None], ["outline"]) == []


def test_features_feed_bench_run(tmp_path, events_cfg, before_freeze):
    # The renamed settings work in run_benchmark (synthetic features with a known signal).
    from test_probes import synth

    from damagesar_bench.run import run_benchmark
    a = synth(events_cfg, signal=2.5, encoder="enc@0.3m")
    n = synth(events_cfg, signal=0.8, encoder="enc@native")
    fa, fn = tmp_path / "a.parquet", tmp_path / "n.parquet"
    a.to_parquet(fa, index=False)
    n.to_parquet(fn, index=False)
    comps = resolution_comparisons("enc", [None, 0.3], ["outline"])
    out = run_benchmark([fa, fn], comps, events_cfg, tmp_path / "run", today=before_freeze, n_boot=100)
    c = pd.read_csv(out / "comparisons.csv").iloc[0]
    assert c["label"] == "0.3m vs native (outline)" and c["estimate"] > 0 and c["strength"] == "exploratory"
