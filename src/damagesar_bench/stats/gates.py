"""Gate 1 (C2 go/no-go, 18 Dec), on the 5 development events only.

FYP Master §7: mean ΔAUROC (projected - outline) >= +0.02 over the development events,
projected better in >= 4 of 5 events, and projected beats the random-shift null.

"Beats the random-shift null" is made concrete here (to be written into the
pre-registration): the random-shift control is run ``R`` times with different random
shifts of the same size distribution; each run gives a mean ΔAUROC (random shift - outline)
over the same events. Projected beats the null if its observed mean gain is larger than the
``null_quantile`` (default 95th percentile) of those null means. With a single null run
this reduces to "strictly larger than the random-shift gain".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .event_level import ROUND_DECIMALS, WORTHWHILE


@dataclass(frozen=True)
class Gate1:
    mean_gain: float
    n_better: int
    n_events: int
    null_threshold: float
    mean_ok: bool
    count_ok: bool
    null_ok: bool

    @property
    def passed(self) -> bool:
        return self.mean_ok and self.count_ok and self.null_ok


def gate1(gains_by_event: dict[str, float], null_mean_gains, *, min_better: int = 4,
          worthwhile: float = WORTHWHILE, null_quantile: float = 0.95) -> Gate1:
    if not gains_by_event:
        raise ValueError("no events")
    g = np.array(list(gains_by_event.values()), dtype=float)
    null = np.atleast_1d(np.asarray(null_mean_gains, dtype=float))
    if not np.isfinite(g).all() or null.size == 0 or not np.isfinite(null).all():
        raise ValueError("gains and null values must be finite and non-empty")
    r = lambda v: round(float(v), ROUND_DECIMALS)  # noqa: E731
    mean = float(g.mean())
    thr = float(null[0]) if null.size == 1 else float(np.quantile(null, null_quantile))
    n_better = int(np.sum(np.round(g, ROUND_DECIMALS) > 0))
    return Gate1(
        mean_gain=mean, n_better=n_better, n_events=len(g), null_threshold=thr,
        mean_ok=r(mean) >= r(worthwhile), count_ok=n_better >= min_better, null_ok=r(mean) > r(thr),
    )
