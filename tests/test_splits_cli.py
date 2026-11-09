import json

from damagesar_bench.cli import main
from damagesar_bench.config import load_bench, load_events
from damagesar_bench.data.splits import make_folds, write_splits


def test_pre_freeze_folds_never_touch_confirmatory(events_cfg, before_freeze):
    folds = make_folds(events_cfg, today=before_freeze)
    assert {f.held_out for f in folds} == {"beirut", "hawaii"}
    for f in folds:
        assert f.phase == "development"
        assert "haiti" not in f.train          # confirmatory never used for training
        assert f.held_out not in f.train
        assert "mexico" not in f.train         # no building units
        assert "marshall" in f.train           # secondary events are allowed


def test_post_freeze_folds_cover_every_event(events_cfg, after_freeze):
    folds = make_folds(events_cfg, today=after_freeze)
    assert {f.held_out for f in folds} == set(events_cfg.events)
    haiti = next(f for f in folds if f.held_out == "haiti")
    assert haiti.phase == "locked"
    assert set(haiti.train) == {"beirut", "hawaii", "marshall"}


def test_write_splits(tmp_path, events_cfg, before_freeze):
    p = write_splits(events_cfg, tmp_path / "splits.json", today=before_freeze)
    data = json.loads(p.read_text())
    assert data["labels_unlocked"] is False
    assert data["roles"]["haiti"] == "confirmatory"
    assert len(data["folds"]) == 2


def test_repo_configs_load():
    events = load_events()
    assert len(events.events) == 14
    assert len(events.keys_with_role("confirmatory")) == 6
    assert len(events.keys_with_role("development")) == 5
    assert len(events.keys_with_role("secondary")) == 3
    assert load_bench().nodata.tile_exclude_frac == 0.95


def test_cli_validate_and_audit(tmp_path, bright_root, capsys):
    out = tmp_path / "out"
    # a blank (excluded) tile is expected data, not a problem -> exit 0
    assert main(["validate", "--root", str(bright_root), "--out", str(out), "--today", "2026-11-09"]) == 0
    assert (out / "tile_report.csv").exists()
    assert main(["audit", "--out", str(out), "--today", "2026-11-09"]) == 0
    folds = json.loads((out / "splits.json").read_text())["folds"]
    assert len(folds) == 5  # five development events in the real config
    assert main(["status", "--today", "2026-11-09"]) == 0
    assert "LOCKED" in capsys.readouterr().out


def test_cli_validate_fails_on_problems(tmp_path, bright_root):
    (bright_root / "target" / "beirut-explosion_00000001_building_damage.tif").unlink()
    assert main(["validate", "--root", str(bright_root), "--out", str(tmp_path), "--today", "2026-11-09"]) == 1


def test_cli_planned_command(capsys):
    assert main(["new-event"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
