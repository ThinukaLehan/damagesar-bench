import json
from dataclasses import replace

import pandas as pd
from test_probes import synth

from damagesar_bench.config import EventInfo, EventsConfig
from damagesar_bench.locked import locked_comparisons, run_locked_analysis


def test_locked_comparisons_mark_only_rq1_rq2_primary():
    c = locked_comparisons("dinov3_vitl16_sat", run_rq2=False)
    assert [x.label for x in c if x.primary] == ["RQ1 DINOv3 vs ImageNet (outline)"]
    assert c[0].a == ("dinov3_vitl16_sat@native", "outline") and c[0].b == ("imagenet_vitb16@native", "outline")
    assert any(x.label.startswith("exploratory DINOv3 ViT-B") and not x.primary for x in c)
    c2 = locked_comparisons("dinov3_vitb16", run_rq2=True)
    assert [x.label[:3] for x in c2 if x.primary] == ["RQ1", "RQ2"]
    assert not any(x.label.startswith("exploratory") for x in c2)


def _cfg():
    roles = {"e1": "confirmatory", "e2": "confirmatory", "e3": "confirmatory", "d1": "development",
             "d2": "development", "s1": "secondary"}
    ev = {k: EventInfo(k, k, "x", r, "instance") for k, r in roles.items()}
    from datetime import date
    return EventsConfig(freeze_date=date(2020, 1, 1), events=ev)


def test_run_locked_analysis_end_to_end(tmp_path):
    cfg = _cfg()
    evs = tuple(cfg.events)
    files = []
    for enc, pool, sig in [("dinov3_vitl16_sat@native", "outline", 2.5), ("dinov3_vitl16_sat@native", "ring10m", 1.0),
                           ("dinov3_vitb16@native", "outline", 1.5), ("imagenet_vitb16@native", "outline", 0.8),
                           ("imagefree", "geometry", 0.3)]:
        df = synth(cfg, today_locked=False, signal=sig, encoder=enc, pooling=pool, events=evs, n=200,
                   seed=hash(pool) % 7)
        p = tmp_path / f"{enc}_{pool}.parquet".replace("@", "_")
        df.to_parquet(p, index=False)
        files.append(p)
    r = run_locked_analysis(files, cfg, tmp_path / "run", rq1_encoder="dinov3_vitl16_sat", run_rq2=False,
                            level=0.975, n_boot=200)
    c = r["comparisons"].set_index("label")
    assert c.loc["RQ1 DINOv3 vs ImageNet (outline)", "strength"] == "confirmatory"
    assert (c.drop(index="RQ1 DINOv3 vs ImageNet (outline)")["strength"] == "exploratory").all()
    assert (c["level"] == 0.975).all() and (c["n_events"] == 3).all()
    assert set(r["per_event"]["event"]) == {"e1", "e2", "e3"}
    assert r["supporting"].n_events == 3 and r["supporting"].n_a_better >= 2
    assert set(r["c3"]["event"]) <= {"e1", "e2", "e3"} and len(r["c3"]) == 3
    assert json.loads((tmp_path / "run" / "rq1_supporting.json").read_text())["n_events"] == 3


def test_c3_source_roles_rehearsal_and_clear_error(tmp_path):
    # Rehearsal shape: no development-role events, as in k00 (development events play the confirmatory role)
    import pytest

    from damagesar_bench.recal.prevalence import RecalError
    roles = {"e1": "confirmatory", "e2": "confirmatory", "e3": "confirmatory"}
    ev = {k: EventInfo(k, k, "x", r, "instance") for k, r in roles.items()}
    from datetime import date
    cfg = EventsConfig(freeze_date=date(2020, 1, 1), events=ev)
    files = []
    for enc, pool, sig in [("dinov3_vitl16_sat@native", "outline", 2.5), ("dinov3_vitl16_sat@native", "ring10m", 1.0),
                           ("dinov3_vitb16@native", "outline", 1.5), ("imagenet_vitb16@native", "outline", 0.8),
                           ("imagefree", "geometry", 0.3)]:
        df = synth(cfg, today_locked=False, signal=sig, encoder=enc, pooling=pool, events=tuple(ev), n=200, seed=1)
        p = tmp_path / f"{enc}_{pool}.parquet".replace("@", "_")
        df.to_parquet(p, index=False)
        files.append(p)
    kw = dict(rq1_encoder="dinov3_vitl16_sat", run_rq2=False, level=0.975, n_boot=100)
    with pytest.raises(RecalError, match="no labelled source scores"):
        run_locked_analysis(files, cfg, tmp_path / "a", **kw)
    r = run_locked_analysis(files, cfg, tmp_path / "b", c3_source_roles=("confirmatory", "secondary"), **kw)
    assert len(r["c3"]) == 3


def test_c3_source_excludes_confirmatory_events_by_default(tmp_path):
    # k01 shape: each confirmatory target is calibrated on development + secondary scores only
    from damagesar_bench.recal import prevalence
    seen = []
    orig = prevalence.estimate_event

    def spy(t, s, y, ev, *a, **k):
        seen.append((ev, len(s)))
        return orig(t, s, y, ev, *a, **k)
    import damagesar_bench.recal.prevalence as m
    m.estimate_event = spy
    try:
        test_run_locked_analysis_end_to_end(tmp_path)
    finally:
        m.estimate_event = orig
    conf_rows = {e: n for e, n in seen if e.startswith("e")}
    assert set(conf_rows) == {"e1", "e2", "e3"}
    assert len(set(conf_rows.values())) == 1   # same source (d1 + d2 + s1) for every confirmatory target
