"""Within-event inference: tile-cluster bootstrap of a paired AUROC difference.

Buildings in one tile share imaging conditions, so they are not independent. The bootstrap
resamples whole tiles (clusters) with replacement and keeps every building of a chosen tile.
This describes one event only; it cannot show generalisation to new events (that is the
event-level layer in ``event_level.py``).

Replicates that contain only one class are skipped and counted; if more than
``max_skip_frac`` of the replicates are skipped the result is flagged unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import MetricError, auroc


@dataclass(frozen=True)
class BootstrapResult:
    estimate: float
    lower: float
    upper: float
    level: float
    n_boot: int
    n_valid: int
    n_skipped: int
    reliable: bool


def cluster_bootstrap_paired_auc(
    y, score_a, score_b, clusters, *, n_boot: int = 2000, level: float = 0.95,
    seed: int = 0, max_skip_frac: float = 0.05,
) -> BootstrapResult:
    """Percentile interval for AUROC(a) - AUROC(b), resampling clusters (tiles)."""
    y = np.asarray(y)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    clusters = np.asarray(clusters)
    if not (len(y) == len(a) == len(b) == len(clusters)):
        raise ValueError("y, scores and clusters must have the same length")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    estimate = auroc(y, a) - auroc(y, b)

    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    diffs, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), size=len(ids))
        idx = np.concatenate([members[k] for k in pick])
        try:
            diffs.append(auroc(y[idx], a[idx]) - auroc(y[idx], b[idx]))
        except MetricError:
            skipped += 1
    if not diffs:
        raise MetricError("every bootstrap replicate had a single class")
    alpha = 1 - level
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=float(estimate), lower=float(lo), upper=float(hi), level=level,
        n_boot=n_boot, n_valid=len(diffs), n_skipped=skipped,
        reliable=skipped <= max_skip_frac * n_boot,
    )
