"""ADR 0001 round 3: does the encoder's input resolution change building-level AUROC?

The same encoder is run on the same tiles at several ground resolutions (``None`` = native).
Each run becomes its own *setting* by renaming the encoder column, e.g. ``dinov3_vitb16@native``
and ``dinov3_vitb16@0.3m``, so ``bench run`` can compare them like two encoders.

Before the freeze this only touches development and secondary events (``extract`` skips
confirmatory tiles), and every comparison is exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import EventsConfig
from ..contracts import validate_building_features
from ..encoders.base import Encoder
from ..encoders.extract import extract, tiles_per_second
from ..pooling.footprint import Method, TileGeometry
from ..run import ComparisonSpec


def resolution_label(res_m: float | None) -> str:
    return "native" if res_m is None else f"{res_m:g}m"


def setting_name(encoder: str, res_m: float | None) -> str:
    return f"{encoder}@{resolution_label(res_m)}"


@dataclass(frozen=True)
class ResolutionRun:
    resolution: str
    setting: str
    tiles: int
    seconds: float
    tiles_per_s: float
    buildings: int


def extract_at_resolutions(
    buildings: pd.DataFrame,
    read_tile: Callable[[str], tuple[np.ndarray, TileGeometry]],
    encoder: Encoder,
    methods: list[str],
    resolutions: list[float | None],
    out_root: str | Path,
    events: EventsConfig,
    today: date | None = None,
    log: Callable[[str], None] = lambda s: None,
    shifts: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, list[ResolutionRun]]:
    """Extract features once per resolution; one folder (resumable) per resolution."""
    if not resolutions:
        raise ValueError("give at least one resolution")
    if len({resolution_label(r) for r in resolutions}) != len(resolutions):
        raise ValueError("resolutions must be distinct")
    parts, runs = [], []
    for res in resolutions:
        label = resolution_label(res)
        log(f"--- {encoder.name} at {label}")
        df, rep = extract(buildings, read_tile, encoder, methods, Path(out_root) / label, events,
                          today=today, target_res_m=res, log=log, shifts=shifts)
        df = df.assign(encoder=setting_name(encoder.name, res))
        parts.append(df)
        runs.append(ResolutionRun(label, setting_name(encoder.name, res), rep.tiles_total, rep.seconds,
                                  tiles_per_second(rep), int(df["pooling"].eq(Method.parse(methods[0]).name).sum())))
    feats = pd.concat(parts, ignore_index=True)
    feats["damaged"] = feats["damaged"].astype("boolean")
    validate_building_features(feats, events, today)
    return feats, runs


def resolution_comparisons(encoder: str, resolutions: list[float | None], methods: list[str],
                           baseline: float | None = None) -> list[ComparisonSpec]:
    """Each resolution vs the baseline (same pooling), and each pooling vs the first one (same resolution)."""
    if baseline not in resolutions:
        raise ValueError("the baseline resolution must be one of the resolutions")
    pools = [Method.parse(m).name for m in methods]
    base = setting_name(encoder, baseline)
    out = []
    for res in resolutions:
        if res == baseline:
            continue
        s = setting_name(encoder, res)
        for p in pools:
            out.append(ComparisonSpec((s, p), (base, p), f"{resolution_label(res)} vs {resolution_label(baseline)} ({p})"))
    for res in resolutions:
        s = setting_name(encoder, res)
        for p in pools[1:]:
            out.append(ComparisonSpec((s, p), (s, pools[0]), f"{p} vs {pools[0]} ({resolution_label(res)})"))
    return out


def runs_table(runs: list[ResolutionRun]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in runs])
