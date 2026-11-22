import dataclasses
import subprocess
from datetime import date

import pytest

from damagesar_bench.data.events import UNLOCK_ENV
from damagesar_bench.freeze import FreezeError, check_freeze, read_record

GOOD = "date: 2026-11-25\ntag: prereg-freeze\ndoi: 10.5281/zenodo.1234567\nlevel: 0.975\n"


def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                        "GIT_COMMITTER_EMAIL": "t@t", "PATH": "/usr/bin:/bin"})


def repo_with(tmp_path, record=GOOD, tag=True):
    git(tmp_path, "init", "-q")
    (tmp_path / "docs").mkdir()
    if record is not None:
        (tmp_path / "docs" / "FREEZE.md").write_text(record)
    (tmp_path / "x").write_text("x")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "freeze")
    if tag:
        git(tmp_path, "tag", "prereg-freeze")
    return tmp_path


def test_record_parsing_and_validation(tmp_path):
    p = tmp_path / "F.md"
    p.write_text(GOOD)
    r = read_record(p)
    assert (r.date, r.tag, r.doi, r.level) == (date(2026, 11, 25), "prereg-freeze", "10.5281/zenodo.1234567", 0.975)
    for bad, msg in [(GOOD.replace("doi: 10.5281/zenodo.1234567", "doi: TBD"), "not a DOI"),
                     (GOOD.replace("level: 0.975", "level: 0.9"), "level"),
                     (GOOD.replace("tag: prereg-freeze\n", ""), "misses")]:
        p.write_text(bad)
        with pytest.raises(FreezeError, match=msg):
            read_record(p)
    with pytest.raises(FreezeError, match="not frozen"):
        read_record(tmp_path / "missing.md")


def test_check_freeze_passes_only_when_everything_holds(tmp_path, events_cfg, monkeypatch):
    monkeypatch.delenv(UNLOCK_ENV, raising=False)
    repo = repo_with(tmp_path)
    ev = dataclasses.replace(events_cfg, freeze_date=date(2026, 11, 25))
    assert check_freeze(repo, ev, today=date(2026, 11, 26)).level == 0.975
    with pytest.raises(FreezeError, match="before the freeze"):
        check_freeze(repo, ev, today=date(2026, 11, 24))
    with pytest.raises(FreezeError, match="freeze_date"):
        check_freeze(repo, dataclasses.replace(events_cfg, freeze_date=date(2027, 1, 10)), today=date(2027, 1, 11))
    monkeypatch.setenv(UNLOCK_ENV, "1")
    with pytest.raises(FreezeError, match="override"):
        check_freeze(repo, ev, today=date(2026, 11, 26))


def test_check_freeze_needs_the_tag(tmp_path, events_cfg, monkeypatch):
    monkeypatch.delenv(UNLOCK_ENV, raising=False)
    repo = repo_with(tmp_path, tag=False)
    ev = dataclasses.replace(events_cfg, freeze_date=date(2026, 11, 25))
    with pytest.raises(FreezeError, match="tag"):
        check_freeze(repo, ev, today=date(2026, 11, 26))


def test_check_freeze_without_a_record(tmp_path, events_cfg, monkeypatch):
    monkeypatch.delenv(UNLOCK_ENV, raising=False)
    repo = repo_with(tmp_path, record=None)
    with pytest.raises(FreezeError, match="not frozen"):
        check_freeze(repo, events_cfg, today=date(2027, 2, 1))
