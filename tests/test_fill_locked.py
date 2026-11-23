import importlib.util
import json
from datetime import date
from pathlib import Path

import pytest
from test_probes import synth

from damagesar_bench.config import EventInfo, EventsConfig
from damagesar_bench.locked import run_locked_analysis

TOOL = Path(__file__).resolve().parents[1] / "paper" / "tools" / "fill_locked.py"
spec = importlib.util.spec_from_file_location("fill_locked", TOOL)
fill = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fill)


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("k01")
    roles = {**{e: "confirmatory" for e in fill.CONFIRMATORY},
             "beirut": "development", "congo": "development", "marshall": "secondary"}
    cfg = EventsConfig(freeze_date=date(2020, 1, 1),
                       events={k: EventInfo(k, k, "x", r, "instance") for k, r in roles.items()})
    files = []
    for enc, pool, sig in [("dinov3_vitl16_sat@native", "outline", 2.5), ("dinov3_vitl16_sat@native", "ring10m", 1.0),
                           ("dinov3_vitb16@native", "outline", 1.5), ("imagenet_vitb16@native", "outline", 0.8),
                           ("imagefree", "geometry", 0.3)]:
        df = synth(cfg, today_locked=False, signal=sig, encoder=enc, pooling=pool, events=tuple(roles), n=150, seed=3)
        p = tmp / f"{enc}_{pool}.parquet".replace("@", "_")
        df.to_parquet(p, index=False)
        files.append(p)
    run_locked_analysis(files, cfg, tmp / "run", rq1_encoder="dinov3_vitl16_sat", run_rq2=False, level=0.975,
                        n_boot=100)
    return tmp / "run"


def test_fill_writes_checked_outputs(run_dir, tmp_path):
    assert fill.main([str(run_dir), "--out", str(tmp_path / "out")]) == 0
    vals = (tmp_path / "out" / "values.tex").read_text()
    for k in ("RQoneEst", "RQoneLow", "RQoneHigh", "RQoneVerdict", "RQoneSatBetter", "RQoneSignFlipOne"):
        assert f"\\newcommand{{\\{k}}}" in vals
    assert "[date]" not in vals and "\\newcommand{\\LockedDate}{" + date.today().isoformat() + "}" in vals
    table = (tmp_path / "out" / "table_locked.tex").read_text()
    assert "Haiti" in table and "DINOv3 ViT-L (satellite) &" in table and "Image-free" in table
    md = (tmp_path / "out" / "locked_results.md").read_text()
    assert "re-derived" in md and "not blind" in md


def test_fill_stops_on_tampered_verdict(run_dir, tmp_path):
    bad = tmp_path / "bad"
    import shutil
    shutil.copytree(run_dir, bad)
    p = bad / "bench_run" / "comparisons.csv"
    import pandas as pd
    c = pd.read_csv(p)
    c.loc[c["label"] == fill.RQ1, "verdict"] = "Supported: a worthwhile gain" if (
        c.loc[c["label"] == fill.RQ1, "verdict"].iloc[0] != "Supported: a worthwhile gain") else "Inconclusive"
    c.to_csv(p, index=False)
    assert fill.main([str(bad), "--out", str(tmp_path / "o")]) == 1


def test_fill_stops_on_wrong_level(run_dir, tmp_path):
    import shutil
    import pandas as pd
    bad = tmp_path / "lvl"
    shutil.copytree(run_dir, bad)
    p = bad / "bench_run" / "comparisons.csv"
    c = pd.read_csv(p)
    c["level"] = 0.95
    c.to_csv(p, index=False)
    assert fill.main([str(bad), "--out", str(tmp_path / "o2")]) == 1
    assert json  # keep import used


def test_fill_stops_on_development_run(run_dir, tmp_path):
    import shutil
    bad = tmp_path / "bad"
    shutil.copytree(run_dir, bad)
    p = bad / "bench_run" / "run_manifest.json"
    m = json.loads(p.read_text())
    m["phase"] = "development"
    p.write_text(json.dumps(m))
    assert fill.main([str(bad), "--out", str(tmp_path / "out")]) == 1
