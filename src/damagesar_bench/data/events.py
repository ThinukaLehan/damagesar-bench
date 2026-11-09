"""Event roles and the confirmatory-label guard (FR3, DR3).

The guard enforces the pre-registration in code: before the freeze date, any attempt to read
labels of a confirmatory event raises ``ConfirmatoryLockError``. Reading images (no labels) is
always allowed, so features can be extracted and cached before the freeze.
"""

from __future__ import annotations

import os
from datetime import date

from damagesar_bench.config import EventsConfig

# Escape hatch for the locked run after the freeze, or for tests. Never set it before 10 Jan.
UNLOCK_ENV = "DAMAGESAR_UNLOCK_CONFIRMATORY"


class ConfirmatoryLockError(PermissionError):
    """Raised when confirmatory labels are requested before the plan freeze."""


class UnknownEventError(KeyError):
    """Raised for an event key that is not in the events config."""


def event_key_from_filename(filename: str) -> str:
    """BRIGHT names files ``<event>-<hazard>_<8-digit id>_<suffix>.tif``.

    The event key is everything before the first ``-``, e.g.
    ``la_palma-volcano_00000001_post_disaster.tif`` -> ``la_palma``.
    """
    stem = os.path.basename(filename)
    if "-" not in stem:
        raise ValueError(f"cannot find an event name in {filename!r}")
    return stem.split("-", 1)[0].lower()


def tile_id_from_filename(filename: str, suffix: str) -> str:
    """``bata-explosion_00000003_post_disaster.tif`` -> ``bata-explosion_00000003``."""
    name = os.path.basename(filename)
    if not name.endswith(suffix):
        raise ValueError(f"{filename!r} does not end with {suffix!r}")
    return name[: -len(suffix)]


def labels_unlocked(events: EventsConfig, today: date | None = None) -> bool:
    if os.environ.get(UNLOCK_ENV) == "1":
        return True
    today = today or date.today()
    return today >= events.freeze_date


def assert_label_access(event_key: str, events: EventsConfig, today: date | None = None) -> None:
    """Refuse confirmatory labels before the freeze. Development and secondary are always allowed."""
    if event_key not in events.events:
        raise UnknownEventError(event_key)
    info = events.events[event_key]
    if info.role == "confirmatory" and not labels_unlocked(events, today):
        raise ConfirmatoryLockError(
            f"Labels of confirmatory event {event_key!r} are locked until "
            f"{events.freeze_date.isoformat()} (pre-registration). Use development events instead."
        )
