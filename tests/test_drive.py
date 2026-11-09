from datetime import date

import pytest

from damagesar_bench import drive


def test_layout_is_created_and_idempotent(tmp_path):
    made = drive.ensure_layout(tmp_path)
    assert all(p.is_dir() for p in made)
    assert (tmp_path / "DamageSAR-Bench" / "03_notebooks" / "checks").is_dir()
    drive.ensure_layout(tmp_path)  # second call must not fail


def test_every_alias_points_inside_the_layout():
    for rel in drive.ALIASES.values():
        assert rel in drive.FOLDERS


def test_path_alias(tmp_path):
    assert drive.path("derived", tmp_path) == tmp_path / "DamageSAR-Bench" / "01_data" / "derived"
    with pytest.raises(KeyError):
        drive.path("downloads", tmp_path)


def test_run_dir_name(tmp_path):
    p = drive.run_dir("02 Token size / Beirut", commit="508730c9ab", today=date(2026, 11, 9), drive_root=tmp_path)
    assert p.name == "2026-11-09_02-token-size-beirut_508730c"
    assert p.is_dir()
    assert p.parent == drive.path("runs", tmp_path)


def test_run_dir_without_commit(tmp_path):
    p = drive.run_dir("check", today=date(2026, 11, 9), drive_root=tmp_path, create=False)
    assert p.name == "2026-11-09_check"
    assert not p.exists()
