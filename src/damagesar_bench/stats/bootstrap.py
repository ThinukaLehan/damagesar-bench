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


def _cluster_bootstrap(stat, n, clusters, *, n_boot, level, seed, max_skip_frac, estimate):
    clusters = np.asarray(clusters)
    if len(clusters) != n:
        raise ValueError("y, scores and clusters must have the same length")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    ids, inverse = np.unique(clusters, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(ids))]
    rng = np.random.default_rng(seed)
    values, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), size=len(ids))
        idx = np.concatenate([members[k] for k in pick])
        try:
            values.append(stat(idx))
        except MetricError:
            skipped += 1
    if not values:
        raise MetricError("every bootstrap replicate had a single class")
    alpha = 1 - level
    lo, hi = np.quantile(values, [alpha / 2, 1 - alpha / 2])
    return BootstrapResult(
        estimate=float(estimate), lower=float(lo), upper=float(hi), level=level,
        n_boot=n_boot, n_valid=len(values), n_skipped=skipped,
        reliable=skipped <= max_skip_frac * n_boot,
    )


def cluster_bootstrap_paired_auc(
    y, score_a, score_b, clusters, *, n_boot: int = 2000, level: float = 0.95,
    seed: int = 0, max_skip_frac: float = 0.05,
) -> BootstrapResult:
    """Percentile interval for AUROC(a) - AUROC(b), resampling clusters (tiles)."""
    y = np.asarray(y)
    a, b = np.asarray(score_a, dtype=float), np.asarray(score_b, dtype=float)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("y, scores and clusters must have the same length")
    est = auroc(y, a) - auroc(y, b)
    return _cluster_bootstrap(lambda i: auroc(y[i], a[i]) - auroc(y[i], b[i]), len(y), clusters,
                              n_boot=n_boot, level=level, seed=seed, max_skip_frac=max_skip_frac, estimate=est)


def cluster_bootstrap_auc(
    y, score, clusters, *, n_boot: int = 2000, level: float = 0.95,
    seed: int = 0, max_skip_frac: float = 0.05,
) -> BootstrapResult:
    """Percentile interval for one AUROC, resampling clusters (tiles)."""
    y = np.asarray(y)
    s = np.asarray(score, dtype=float)
    if len(y) != len(s):
        raise ValueError("y, scores and clusters must have the same length")
    est = auroc(y, s)
    return _cluster_bootstrap(lambda i: auroc(y[i], s[i]), len(y), clusters,
                              n_boot=n_boot, level=level, seed=seed, max_skip_frac=max_skip_frac, estimate=est)
