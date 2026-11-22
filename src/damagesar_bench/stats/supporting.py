"""Pre-registered supporting evidence for a primary comparison (fixed 22 Nov 2026, before the freeze).

With five confirmatory events the event-level interval can only confirm a large gain. Two pieces of supporting
evidence make the locked result informative without changing the primary claim:

1. **Within each event:** the paired AUROC difference (setting a - setting b, the same buildings) with a 95%
   tile-cluster bootstrap interval (tiles resampled with replacement, 2,000 replicates, seed 0). An event counts
   for a (or b) when the whole interval is above (below) zero. The paired DeLong difference and its p-value are
   reported as a side check; DeLong treats buildings as independent, so it is not used for the count.
2. **Across events:** the exact one-sided sign-flip p-value in the pre-registered direction (a better than b):
   the share of the 2^n sign patterns whose sum of gains is at least the observed sum.

Neither replaces the primary event-level verdict; both are labelled supporting in the report.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from .bootstrap import cluster_bootstrap_paired_auc
from .event_level import MAX_EXACT_SIGNFLIP, _gains
from .metrics import delong_paired_diff

KEYS = ["event", "tile_id", "instance_id"]


def one_sided_sign_flip(gains) -> float:
    """Exact one-sided sign-flip p-value for 'a better than b' (positive gains)."""
    g = _gains(gains)
    n = len(g)
    if n > MAX_EXACT_SIGNFLIP:
        raise ValueError(f"exact test limited to {MAX_EXACT_SIGNFLIP} events")
    obs = g.sum()
    signs = np.array(list(itertools.product((1, -1), repeat=n)))
    sums = signs @ g
    tol = 1e-12 * max(1.0, abs(obs))
    return float(np.sum(sums >= obs - tol)) / 2**n


def per_event_paired(scores: pd.DataFrame, a: tuple[str, str], b: tuple[str, str], events: list[str], *,
                     n_boot: int = 2000, seed: int = 0, level: float = 0.95) -> pd.DataFrame:
    """One row per event: paired AUROC difference a - b on the same buildings, tile-cluster bootstrap interval,
    DeLong side check, and which setting the event counts for ("a", "b" or "neither")."""
    def pick(enc, pool):
        s = scores[(scores["encoder"] == enc) & (scores["pooling"] == pool) & scores["event"].isin(events)]
        if s.empty:
            raise ValueError(f"no scores for {enc}/{pool}")
        return s[KEYS + ["damaged", "score"]]

    m = pick(*a).merge(pick(*b), on=KEYS, suffixes=("_a", "_b"), validate="one_to_one")
    rows = []
    for ev in events:
        g = m[(m["event"] == ev) & m["damaged_a"].notna()]
        if g.empty:
            raise ValueError(f"event {ev}: no labelled buildings in both settings")
        y = g["damaged_a"].astype(bool).to_numpy().astype(int)
        boot = cluster_bootstrap_paired_auc(y, g["score_a"].to_numpy(), g["score_b"].to_numpy(),
                                            g["tile_id"].to_numpy(), n_boot=n_boot, level=level, seed=seed)
        diff, se = delong_paired_diff(y, g["score_a"].to_numpy(), g["score_b"].to_numpy())
        p = float(2 * norm.sf(abs(diff) / se)) if se > 0 else float("nan")
        counts = "a" if boot.lower > 0 else ("b" if boot.upper < 0 else "neither")
        rows.append({"event": ev, "n": len(g), "diff": boot.estimate, "boot_lower": boot.lower,
                     "boot_upper": boot.upper, "boot_reliable": boot.reliable, "delong_se": se,
                     "delong_p": p, "counts_for": counts})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class Supporting:
    n_events: int
    n_a_better: int          # events whose 95% tile-bootstrap interval is above 0
    n_b_better: int          # ... below 0
    sign_flip_one_sided_p: float


def summarise(per_event: pd.DataFrame) -> Supporting:
    return Supporting(len(per_event), int((per_event["counts_for"] == "a").sum()),
                      int((per_event["counts_for"] == "b").sum()),
                      one_sided_sign_flip(per_event.set_index("event")["diff"]))
