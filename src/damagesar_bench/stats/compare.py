"""From fold scores to event metrics, and from event metrics to a claim.

``event_metrics``: per (encoder, pooling, held-out event): AUROC, AP, DeLong SE and a
tile-cluster bootstrap interval for AUROC (within-event description only).

``compare``: per-event AUROC gains of setting A over setting B on the events both have,
then the event-level t-interval, the ordered v11.1 verdict and the sign-flip test.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import EventsConfig
from ..contracts import validate_event_metrics
from .bootstrap import cluster_bootstrap_auc
from .event_level import Interval, SignFlip, Verdict, decide_interval, sign_flip_test, t_interval
from .metrics import average_precision, delong


def event_metrics(scores: pd.DataFrame, events: EventsConfig, *, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for (enc, pool, ev), g in scores.groupby(["encoder", "pooling", "event"], sort=True):
        g = g[g["damaged"].notna()]
        if g.empty:
            continue  # locked confirmatory event: nothing to score yet
        y = g["damaged"].astype(int).to_numpy()
        s = g["score"].to_numpy(float)
        d = delong(y, s)
        boot = cluster_bootstrap_auc(y, s, g["tile_id"].to_numpy(), n_boot=n_boot, seed=seed)
        rows.append({
            "event": ev, "role": g["role"].iloc[0], "encoder": enc, "pooling": pool,
            "n": len(g), "n_damaged": int(y.sum()), "auroc": float(d.auc[0]),
            "ap": average_precision(y, s), "auroc_se": float(d.se[0]),
            "boot_lower": boot.lower, "boot_upper": boot.upper, "boot_reliable": boot.reliable,
        })
    df = pd.DataFrame(rows)
    if len(df):
        validate_event_metrics(df, events)
    return df


@dataclass(frozen=True)
class Comparison:
    a: tuple[str, str]
    b: tuple[str, str]
    gains: pd.Series            # per event, AUROC(a) - AUROC(b)
    interval: Interval
    verdict: Verdict
    sign_flip: SignFlip


def compare(metrics: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], *,
            events: list[str] | None = None, level: float = 0.95) -> Comparison:
    """Compare setting ``a`` = (encoder, pooling) against ``b`` over the events both have.

    ``events`` fixes the event set (e.g. the 6 confirmatory events); then every listed event
    must be present for both settings, so an event cannot silently drop out.
    """
    def per_event(key):
        m = metrics[(metrics["encoder"] == key[0]) & (metrics["pooling"] == key[1])]
        return m.set_index("event")["auroc"]

    ga, gb = per_event(a), per_event(b)
    if events is not None:
        missing = [e for e in events if e not in ga.index or e not in gb.index]
        if missing:
            raise ValueError(f"events missing for one of the settings: {missing}")
        common = list(events)
    else:
        common = sorted(set(ga.index) & set(gb.index))
    if len(common) < 2:
        raise ValueError("need at least 2 events with both settings")
    gains = (ga.loc[common] - gb.loc[common]).rename("gain")
    iv = t_interval(gains.to_numpy(), level)
    return Comparison(a, b, gains, iv, decide_interval(iv), sign_flip_test(gains.to_numpy()))
