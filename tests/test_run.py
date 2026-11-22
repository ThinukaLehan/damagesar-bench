import json

import pandas as pd
import pytest
from test_probes import synth  # synthetic building features with a known signal

from damagesar_bench.cli import main
from damagesar_bench.run import ComparisonSpec, RunError, comparison_events, run_benchmark, sha256, write_report


def write_features(tmp_path, events_cfg, locked=True):
    a = synth(events_cfg, today_locked=locked, signal=2.5, encoder="enc_a")
    b = synth(events_cfg, today_locked=locked, signal=0.6, encoder="enc_b")
    pa, pb = tmp_path / "a.parquet", tmp_path / "b.parquet"
    a.to_parquet(pa, index=False)
    b.to_parquet(pb, index=False)
    return [pa, pb]


def test_comparison_spec_parse():
    c = ComparisonSpec.parse("rq1=enc_a/outline:enc_b/outline")
    assert c.label == "rq1" and c.a == ("enc_a", "outline") and c.b == ("enc_b", "outline")
    assert ComparisonSpec.parse("x/y:z/w").label == "x/y vs z/w"
    for bad in ("enc_a/outline", "a:b", "a/b:c", "a/b/c:d/e", "/b:c/d"):
        with pytest.raises(RunError):
            ComparisonSpec.parse(bad)


def test_comparison_events(events_cfg, before_freeze, after_freeze):
    assert comparison_events(events_cfg, before_freeze, "auto") == ["beirut", "hawaii"]
    assert comparison_events(events_cfg, after_freeze, "auto") == ["haiti"]
    assert comparison_events(events_cfg, before_freeze, "all") is None
    with pytest.raises(RunError, match="refused before the freeze"):
        comparison_events(events_cfg, before_freeze, "confirmatory")
    with pytest.raises(RunError, match="unknown event set"):
        comparison_events(events_cfg, before_freeze, "x")


def test_run_development_phase(tmp_path, events_cfg, before_freeze):
    files = write_features(tmp_path, events_cfg)
    out = run_benchmark(files, [ComparisonSpec.parse("rq1=enc_a/outline:enc_b/outline")], events_cfg,
                        tmp_path / "run", today=before_freeze, n_boot=200)
    for f in ("fold_scores.parquet", "probe_fits.csv", "event_metrics.csv", "comparisons.csv",
              "run_manifest.json", "report.md"):
        assert (out / f).exists(), f
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["phase"] == "development" and man["comparison_events"] == ["beirut", "hawaii"]
    assert man["inputs"][str(files[0])] == sha256(files[0])
    m = pd.read_csv(out / "event_metrics.csv")
    assert set(m["event"]) == {"beirut", "hawaii"}            # no confirmatory event scored
    c = pd.read_csv(out / "comparisons.csv").iloc[0]
    assert c["strength"] == "exploratory" and c["n_events"] == 2 and c["estimate"] > 0
    gains = json.loads(c["gains"])
    ma = m.set_index(["encoder", "event"])["auroc"]
    assert gains["beirut"] == pytest.approx(ma["enc_a", "beirut"] - ma["enc_b", "beirut"], abs=1e-6)
    rep = (out / "report.md").read_text()
    assert "exploratory reading, not a claim" in rep and "haiti" not in rep


def test_run_after_freeze_uses_confirmatory_set(tmp_path, events_cfg, after_freeze):
    files = write_features(tmp_path, events_cfg, locked=False)
    # the fixture has only one confirmatory event (haiti): a t-interval needs at least 2 events
    with pytest.raises(ValueError, match="at least 2"):
        run_benchmark(files, [ComparisonSpec.parse("enc_a/outline:enc_b/outline")], events_cfg,
                      tmp_path / "run", today=after_freeze, n_boot=100)
    out = run_benchmark(files, [ComparisonSpec.parse("enc_a/outline:enc_b/outline")], events_cfg,
                        tmp_path / "run2", today=after_freeze, n_boot=100, event_set="all")
    c = pd.read_csv(out / "comparisons.csv").iloc[0]
    assert c["n_events"] == 4 and c["strength"] == "exploratory"   # 'all' is never the confirmatory claim


def test_run_refuses_unknown_setting_and_empty_input(tmp_path, events_cfg, before_freeze):
    files = write_features(tmp_path, events_cfg)
    with pytest.raises(RunError, match="not in the features"):
        run_benchmark(files, [ComparisonSpec.parse("enc_c/outline:enc_b/outline")], events_cfg,
                      tmp_path / "run", today=before_freeze, n_boot=50)
    with pytest.raises(RunError, match="at least one"):
        run_benchmark([], [], events_cfg, tmp_path / "run", today=before_freeze)


def test_run_is_reproducible(tmp_path, events_cfg, before_freeze):
    files = write_features(tmp_path, events_cfg)
    spec = [ComparisonSpec.parse("enc_a/outline:enc_b/outline")]
    a = run_benchmark(files, spec, events_cfg, tmp_path / "r1", today=before_freeze, n_boot=150)
    b = run_benchmark(files, spec, events_cfg, tmp_path / "r2", today=before_freeze, n_boot=150)
    for f in ("event_metrics.csv", "comparisons.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(a / f), pd.read_csv(b / f))


def test_report_rebuild_and_cli(tmp_path, events_cfg, capsys):
    files = write_features(tmp_path, events_cfg)
    args = ["run", "--today", "2026-11-09", "--n-boot", "100", "--out", str(tmp_path / "run"),
            "--compare", "rq1=enc_a/outline:enc_b/outline"]
    for f in files:
        args += ["--features", str(f)]
    # use the fixture roles: point the CLI at a temporary events.yaml
    ev = tmp_path / "events.yaml"
    ev.write_text("freeze_date: '2027-01-10'\nevents:\n" + "".join(
        f"  {k}: {{name: '{e.name}', hazard: {e.hazard}, role: {e.role}, units: {e.units}}}\n"
        for k, e in events_cfg.events.items()))
    assert main(args + ["--events", str(ev)]) == 0
    assert "DamageSAR-Bench run report" in capsys.readouterr().out
    (tmp_path / "run" / "report.md").unlink()
    assert main(["report", "--run", str(tmp_path / "run")]) == 0
    assert (tmp_path / "run" / "report.md").exists()
    assert write_report(tmp_path / "run").read_text().count("| enc_") >= 4


def test_only_compared_skips_unused_settings(tmp_path, events_cfg, before_freeze):
    files = write_features(tmp_path, events_cfg)
    extra = synth(events_cfg, today_locked=True, signal=1.0, encoder="enc_c")
    pc = tmp_path / "c.parquet"
    extra.to_parquet(pc, index=False)
    out = run_benchmark(files + [pc], [ComparisonSpec.parse("enc_a/outline:enc_b/outline")], events_cfg,
                        tmp_path / "run", today=before_freeze, n_boot=50, only_compared=True)
    m = pd.read_csv(out / "event_metrics.csv")
    assert set(m["encoder"]) == {"enc_a", "enc_b"}             # enc_c was in the files but not probed
    man = json.loads((out / "run_manifest.json").read_text())
    assert man["settings"] == ["enc_a/outline", "enc_b/outline"]


def test_comparison_spec_is_primary_unless_marked():
    assert ComparisonSpec.parse("a/outline:b/outline").primary
    assert not ComparisonSpec(("a", "outline"), ("b", "outline"), "RQ4 x", primary=False).primary
