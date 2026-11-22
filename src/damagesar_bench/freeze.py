"""The freeze record: the confirmatory labels may be opened only after the analysis plan is frozen.

``docs/FREEZE.md`` is written on the freeze day and holds four lines (see ``docs/FREEZE_TEMPLATE.md``)::

    date: 2026-11-25
    tag: prereg-freeze
    doi: 10.5281/zenodo.1234567
    level: 0.975

``check_freeze`` refuses the locked run unless all of these hold:

1. the record exists and every field is filled (a DOI of the form ``10.<digits>/...``, level 0.95 or 0.975);
2. the git tag exists and is an ancestor of the commit being run (the plan was fixed before this code ran);
3. ``configs/events.yaml``'s ``freeze_date`` equals the record's date, and today is on or after it;
4. the override environment variable is NOT set (the locked run never uses it).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from .config import EventsConfig
from .data.events import UNLOCK_ENV

FIELDS = ("date", "tag", "doi", "level")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")


class FreezeError(RuntimeError):
    pass


@dataclass(frozen=True)
class FreezeRecord:
    date: date
    tag: str
    doi: str
    level: float


def read_record(path: str | Path) -> FreezeRecord:
    p = Path(path)
    if not p.exists():
        raise FreezeError(f"no freeze record at {p}: the plan is not frozen")
    vals: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*(date|tag|doi|level)\s*:\s*(\S+)\s*$", line)
        if m:
            vals[m.group(1)] = m.group(2)
    missing = [f for f in FIELDS if not vals.get(f)]
    if missing:
        raise FreezeError(f"freeze record misses {missing}")
    if not DOI_RE.match(vals["doi"]):
        raise FreezeError(f"freeze record DOI {vals['doi']!r} is not a DOI (10.xxxx/...)")
    level = float(vals["level"])
    if level not in (0.95, 0.975):
        raise FreezeError("level must be 0.95 or 0.975 (Bonferroni over the two primary claims)")
    try:
        d = date.fromisoformat(vals["date"])
    except ValueError:
        raise FreezeError(f"freeze date {vals['date']!r} is not YYYY-MM-DD") from None
    return FreezeRecord(d, vals["tag"], vals["doi"], level)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)


def check_freeze(repo_root: str | Path, events: EventsConfig, today: date | None = None) -> FreezeRecord:
    """Raise ``FreezeError`` unless the locked run is allowed (module docstring); return the record."""
    repo = Path(repo_root)
    if os.environ.get(UNLOCK_ENV):
        raise FreezeError(f"{UNLOCK_ENV} is set: the locked run must not use the override")
    rec = read_record(repo / "docs" / "FREEZE.md")
    if _git(repo, "rev-parse", "-q", "--verify", f"refs/tags/{rec.tag}").returncode != 0:
        raise FreezeError(f"git tag {rec.tag!r} not found (fetch tags: git fetch --tags)")
    if _git(repo, "merge-base", "--is-ancestor", rec.tag, "HEAD").returncode != 0:
        raise FreezeError(f"tag {rec.tag!r} is not an ancestor of this commit")
    if events.freeze_date != rec.date:
        raise FreezeError(f"configs/events.yaml freeze_date {events.freeze_date} != freeze record {rec.date}")
    today = today or date.today()
    if today < rec.date:
        raise FreezeError(f"today {today} is before the freeze date {rec.date}")
    return rec
