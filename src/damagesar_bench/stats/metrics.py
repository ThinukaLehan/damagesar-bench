"""Building-level metrics: AUROC, average precision, DeLong variance (single and paired).

AUROC uses mid-ranks, so ties count one half (same as the Mann-Whitney U statistic and
scikit-learn's ``roc_auc_score``). Average precision follows scikit-learn's definition
(step-wise, no interpolation). DeLong follows Sun and Xu (2014), "Fast implementation of
DeLong's algorithm", IEEE Signal Processing Letters 21(11):1389-1393.

Labels are binary: 1 = damaged (damaged or destroyed), 0 = intact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata


class MetricError(ValueError):
    """Raised when a metric is undefined for the given labels."""


def _check(y, *scores):
    y = np.asarray(y)
    if y.ndim != 1:
        raise MetricError("labels must be 1-D")
    if not np.isin(y, (0, 1)).all():
        raise MetricError("labels must be 0/1")
    out = []
    for s in scores:
        s = np.asarray(s, dtype=float)
        if s.shape != y.shape:
            raise MetricError(f"scores shape {s.shape} != labels shape {y.shape}")
        if not np.isfinite(s).all():
            raise MetricError("scores must be finite")
        out.append(s)
    n_pos = int(y.sum())
    if n_pos == 0 or n_pos == len(y):
        raise MetricError("need both classes (at least one damaged and one intact building)")
    return y.astype(int), out


def auroc(y, score) -> float:
    y, (s,) = _check(y, score)
    r = rankdata(s)  # mid-ranks for ties
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, score) -> float:
    y, (s,) = _check(y, score)
    order = np.argsort(-s, kind="mergesort")
    s_sorted, y_sorted = s[order], y[order]
    # one threshold per distinct score: take the last index of each run of equal scores
    last = np.r_[np.nonzero(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp = np.cumsum(y_sorted)[last]
    fp = (last + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


@dataclass(frozen=True)
class DeLong:
    auc: np.ndarray        # AUROC per score column
    cov: np.ndarray        # DeLong covariance matrix of the AUROCs

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))


def delong(y, *scores) -> DeLong:
    """AUROCs and their DeLong covariance for one or more scores on the same buildings."""
    y, ss = _check(y, *scores)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    aucs, v10, v01 = [], [], []
    for s in ss:
        x, z = s[pos], s[neg]
        tx = rankdata(x)               # ranks within positives
        tz = rankdata(z)               # ranks within negatives
        tall = rankdata(np.r_[x, z])   # ranks in the pooled sample
        aucs.append((tall[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tall[:m] - tx) / n)          # per positive: P(score > a negative)
        v01.append(1.0 - (tall[m:] - tz) / m)    # per negative: P(a positive > score)
    v10, v01 = np.array(v10), np.array(v01)
    s10 = np.atleast_2d(np.cov(v10)) if m > 1 else np.zeros((len(ss), len(ss)))
    s01 = np.atleast_2d(np.cov(v01)) if n > 1 else np.zeros((len(ss), len(ss)))
    return DeLong(auc=np.array(aucs), cov=s10 / m + s01 / n)


def delong_paired_diff(y, score_a, score_b) -> tuple[float, float]:
    """AUROC(a) - AUROC(b) and its DeLong standard error (same buildings, paired)."""
    d = delong(y, score_a, score_b)
    var = d.cov[0, 0] + d.cov[1, 1] - 2 * d.cov[0, 1]
    return float(d.auc[0] - d.auc[1]), float(np.sqrt(max(var, 0.0)))
