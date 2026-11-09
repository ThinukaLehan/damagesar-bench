"""Data contracts between pipeline stages (ADR 0002).

Three tables connect the stages. Every stage validates what it writes and what it reads,
so a mistake stops the run instead of silently changing a result.

1. **Building features** (pooling -> probes): one row per kept building and one
   (encoder, pooling) pair.
2. **Fold scores** (probes -> statistics): one row per held-out building, with the probe
   score and the events the probe was trained on.
3. **Event metrics** (statistics -> report): one row per (encoder, pooling, held-out event).

Feature columns are named ``f0000, f0001, ...`` (contiguous, float). Keys are
``(event, tile_id, instance_id)``; ``instance_id`` is the instance number inside the tile's
BRIGHT instance JSON.
"""

from __future__ import annotations

import re
from datetime import date

import numpy as np
import pandas as pd

from .config import EventsConfig
from .data.events import labels_unlocked

KEYS = ("event", "tile_id", "instance_id")
FEATURE_RE = re.compile(r"^f\d{4}$")
MAX_FEATURES = 10_000

BUILDING_FEATURE_COLS = (*KEYS, "role", "damaged", "labels_locked", "area_m2", "encoder", "pooling")
FOLD_SCORE_COLS = (*KEYS, "role", "damaged", "score", "encoder", "pooling", "train_events")
EVENT_METRIC_COLS = ("event", "role", "encoder", "pooling", "n", "n_damaged", "auroc", "ap",
                     "auroc_se", "boot_lower", "boot_upper", "boot_reliable")


class ContractError(ValueError):
    """A table does not meet its contract."""


def feature_name(i: int) -> str:
    if not 0 <= i < MAX_FEATURES:
        raise ValueError(f"feature index {i} out of range")
    return f"f{i:04d}"


def feature_columns(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if FEATURE_RE.match(str(c)))


def join_events(events) -> str:
    """Canonical text form of a set of events: sorted, '+'-joined."""
    return "+".join(sorted(set(events)))


def split_events(text: str) -> set[str]:
    return set(filter(None, str(text).split("+")))


# ---------------------------------------------------------------- helpers

def _require(df: pd.DataFrame, cols, table: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ContractError(f"{table}: missing columns {missing}")


def _check_keys(df: pd.DataFrame, extra: tuple[str, ...], table: str) -> None:
    keys = [*extra, *KEYS]
    if df[keys].isna().any().any():
        raise ContractError(f"{table}: key columns {keys} must not be empty")
    dup = df.duplicated(keys)
    if dup.any():
        raise ContractError(f"{table}: {int(dup.sum())} duplicate rows for key {keys}, e.g. "
                            f"{df.loc[dup, keys].iloc[0].to_dict()}")


def _check_roles(df: pd.DataFrame, events: EventsConfig, table: str) -> None:
    unknown = sorted(set(df["event"]) - set(events.events))
    if unknown:
        raise ContractError(f"{table}: unknown events {unknown}")
    expected = df["event"].map(lambda e: events.events[e].role)
    bad = df["role"] != expected
    if bad.any():
        r = df.loc[bad].iloc[0]
        raise ContractError(f"{table}: role of {r['event']} is {r['role']!r}, config says {expected[bad].iloc[0]!r}")


def _check_labels(df: pd.DataFrame, events: EventsConfig, today: date | None, table: str,
                  locked_col: str | None = "labels_locked") -> None:
    """Confirmatory labels must be absent before the freeze; other labels must be present."""
    locked_now = (df["role"] == "confirmatory") & (not labels_unlocked(events, today))
    if locked_col is not None:
        if (df[locked_col].astype(bool) != locked_now).any():
            raise ContractError(f"{table}: '{locked_col}' disagrees with the freeze rule")
    has_label = df["damaged"].notna()
    if (has_label & locked_now).any():
        ev = sorted(df.loc[has_label & locked_now, "event"].unique())
        raise ContractError(f"{table}: confirmatory labels present before the freeze for {ev}")
    if (~has_label & ~locked_now).any():
        ev = sorted(df.loc[~has_label & ~locked_now, "event"].unique())
        raise ContractError(f"{table}: missing labels for unlocked events {ev}")


def _check_single(df: pd.DataFrame, col: str, table: str) -> None:
    if df[col].isna().any() or (df[col].astype(str).str.len() == 0).any():
        raise ContractError(f"{table}: '{col}' must be filled in on every row")


# ---------------------------------------------------------------- 1. building features

def validate_building_features(df: pd.DataFrame, events: EventsConfig, today: date | None = None) -> list[str]:
    """Check a building-features table; returns its feature columns."""
    t = "building features"
    _require(df, BUILDING_FEATURE_COLS, t)
    if len(df) == 0:
        raise ContractError(f"{t}: table is empty")
    for c in ("encoder", "pooling"):
        _check_single(df, c, t)
    _check_keys(df, ("encoder", "pooling"), t)
    _check_roles(df, events, t)
    _check_labels(df, events, today, t)
    feats = feature_columns(df)
    if not feats:
        raise ContractError(f"{t}: no feature columns (f0000, f0001, ...)")
    if feats != [feature_name(i) for i in range(len(feats))]:
        raise ContractError(f"{t}: feature columns must be contiguous from f0000")
    x = df[feats].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ContractError(f"{t}: features contain NaN or inf")
    area = pd.to_numeric(df["area_m2"], errors="coerce")
    if area.isna().any() or (area <= 0).any():
        raise ContractError(f"{t}: area_m2 must be a positive number")
    return feats


# ---------------------------------------------------------------- 2. fold scores

def validate_fold_scores(df: pd.DataFrame, events: EventsConfig, today: date | None = None) -> None:
    """Check probe scores. Also enforces the probe-training rule (FYP Master §5.3)."""
    t = "fold scores"
    _require(df, FOLD_SCORE_COLS, t)
    if len(df) == 0:
        raise ContractError(f"{t}: table is empty")
    for c in ("encoder", "pooling", "train_events"):
        _check_single(df, c, t)
    _check_keys(df, ("encoder", "pooling"), t)
    _check_roles(df, events, t)
    _check_labels(df, events, today, t, locked_col=None)
    s = pd.to_numeric(df["score"], errors="coerce")
    if s.isna().any() or not np.isfinite(s).all():
        raise ContractError(f"{t}: score must be a finite number")
    unlocked = labels_unlocked(events, today)
    confirmatory = set(events.keys_with_role("confirmatory"))
    for (enc, pool, ev), g in df.groupby(["encoder", "pooling", "event"], sort=False):
        trains = g["train_events"].unique()
        if len(trains) != 1:
            raise ContractError(f"{t}: {enc}/{pool}/{ev} has more than one training set")
        train = split_events(trains[0])
        if not train:
            raise ContractError(f"{t}: {enc}/{pool}/{ev} has an empty training set")
        if ev in train:
            raise ContractError(f"{t}: held-out event {ev} is in its own training set")
        unknown = train - set(events.events)
        if unknown:
            raise ContractError(f"{t}: unknown training events {sorted(unknown)}")
        if not unlocked and train & confirmatory:
            raise ContractError(f"{t}: confirmatory events {sorted(train & confirmatory)} used for training "
                                "before the freeze")


# ---------------------------------------------------------------- 3. event metrics

def validate_event_metrics(df: pd.DataFrame, events: EventsConfig) -> None:
    t = "event metrics"
    _require(df, EVENT_METRIC_COLS, t)
    if df[["encoder", "pooling", "event"]].isna().any().any():
        raise ContractError(f"{t}: encoder, pooling and event must be filled in")
    dup = df.duplicated(["encoder", "pooling", "event"])
    if dup.any():
        raise ContractError(f"{t}: duplicate (encoder, pooling, event) rows")
    _check_roles(df, events, t)
    n, npos = df["n"], df["n_damaged"]
    if ((npos < 1) | (npos >= n)).any():
        raise ContractError(f"{t}: every event needs both classes (1 <= n_damaged < n)")
    for c in ("auroc", "ap"):
        v = pd.to_numeric(df[c], errors="coerce")
        if v.isna().any() or ((v < 0) | (v > 1)).any():
            raise ContractError(f"{t}: {c} must be in [0, 1]")
    lo, hi = df["boot_lower"], df["boot_upper"]
    if (lo > hi).any():
        raise ContractError(f"{t}: boot_lower > boot_upper")
