"""Leave-one-event-out folds (FR3).

Training-set rule (pre-registration, decision 2 in the supervisor briefing):
- Before the freeze, only development events are held out, and probes train on development +
  secondary events. Confirmatory labels are never used, not even as training data.
- After the freeze, every event can be held out, and probes train on all other events with valid
  building units.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from damagesar_bench import __version__
from damagesar_bench.config import EventsConfig
from damagesar_bench.data.events import labels_unlocked


@dataclass(frozen=True)
class Fold:
    held_out: str
    held_out_role: str
    train: tuple[str, ...]
    phase: str  # "development" (pre-freeze) or "locked" (post-freeze)


def usable_for_training(key: str, events: EventsConfig) -> bool:
    """Events without building units cannot provide training labels."""
    return events.events[key].units != "none"


def make_folds(events: EventsConfig, today: date | None = None) -> list[Fold]:
    unlocked = labels_unlocked(events, today)
    keys = sorted(events.events)
    folds: list[Fold] = []
    if not unlocked:
        allowed = [k for k in keys if events.events[k].role in ("development", "secondary")]
        for k in events.keys_with_role("development"):
            train = tuple(t for t in allowed if t != k and usable_for_training(t, events))
            folds.append(Fold(held_out=k, held_out_role="development", train=train, phase="development"))
    else:
        for k in keys:
            train = tuple(t for t in keys if t != k and usable_for_training(t, events))
            folds.append(Fold(held_out=k, held_out_role=events.events[k].role, train=train, phase="locked"))
    return folds


def write_splits(events: EventsConfig, path: str | Path, today: date | None = None) -> Path:
    today = today or date.today()
    folds = make_folds(events, today)
    payload = {
        "generated": today.isoformat(),
        "tool_version": __version__,
        "freeze_date": events.freeze_date.isoformat(),
        "labels_unlocked": labels_unlocked(events, today),
        "roles": {k: e.role for k, e in sorted(events.events.items())},
        "units": {k: e.units for k, e in sorted(events.events.items())},
        "folds": [asdict(f) for f in folds],
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p
