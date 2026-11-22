"""C3: label-free estimate of a new event's damage prevalence, with an abstain rule.

Inputs are the probes' held-out scores (``fold_scores``): for a target event ``e``, its buildings' scores come
from the probe trained without ``e``; the *source* is every other event's held-out scores (each also from a probe
that never saw that event), so source scores are out-of-sample, as the methods need. Target labels are never
read; they are used afterwards, only to score the estimate.

Steps for one target event (rule fixed 22 Nov 2026, before any C3 result was computed):

1. **Calibrate on the source** (Platt scaling of the logit score, as bias-corrected calibration recommends
   [Alexandari et al., 2020] in its simplest form): ``p = sigmoid(a * logit(s) + b)`` fitted on source labels.
   The calibrated posteriors carry the source prior ``pi_s`` (the source share of damaged buildings).
2. **EM** [Saerens et al., 2002]: re-weight the calibrated target posteriors by ``pi_t / pi_s`` and iterate
   ``pi_t = mean(posteriors)`` to convergence. This is the reported estimate.
3. **BBSE** [Lipton et al., 2018], binary form (adjusted classify-and-count): at the probe's 0.5 threshold,
   ``pi_t = (q - fpr) / (tpr - fpr)`` with ``q`` the target's predicted-positive share and ``tpr``, ``fpr``
   measured on the source; clipped to [0, 1]. Used as a cross-check.
4. **Abstain** when the estimate cannot be trusted:
   * the source separates the classes too weakly at the threshold (``tpr - fpr < MIN_YOUDEN`` = 0.2), or
   * EM and BBSE disagree by more than ``MAX_DISAGREEMENT`` = 50% of the larger estimate.
   An abstention reports no count.

Baseline: the training-average prevalence (the source share ``pi_s``), i.e. what a model that ignores the new
event would assume.

C3 gate (development events): relative error ``|pi_hat - pi| / pi`` <= 0.5 on >= 4 of 5 events, and lower than
the baseline's on >= 4 of 5. An abstention counts as a miss for both.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

MIN_YOUDEN = 0.2
MAX_DISAGREEMENT = 0.5
MAX_REL_ERROR = 0.5
MIN_EVENTS = 4
EPS = 1e-6


class RecalError(ValueError):
    pass


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def platt_fit(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(a, b) of ``sigmoid(a * logit(s) + b)`` fitted by maximum likelihood on source labels."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        raise RecalError("the source needs both classes to calibrate")
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(scores)[:, None], y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def platt_apply(scores: np.ndarray, a: float, b: float) -> np.ndarray:
    return _sigmoid(a * _logit(scores) + b)


def em_prior(posteriors: np.ndarray, pi_s: float, tol: float = 1e-8, max_iter: int = 1000) -> tuple[float, int]:
    """Saerens et al. (2002) EM for the target prior, given posteriors calibrated to source prior ``pi_s``."""
    p = np.clip(np.asarray(posteriors, dtype=float), EPS, 1 - EPS)
    if not 0 < pi_s < 1:
        raise RecalError("source prior must be in (0, 1)")
    pi = float(pi_s)
    for it in range(1, max_iter + 1):
        w1, w0 = pi / pi_s, (1 - pi) / (1 - pi_s)
        post = w1 * p / (w1 * p + w0 * (1 - p))
        new = float(post.mean())
        if abs(new - pi) < tol:
            return new, it
        pi = new
    return pi, max_iter


def bbse_prior(q: float, tpr: float, fpr: float) -> float:
    """Binary BBSE / adjusted classify-and-count, clipped to [0, 1]."""
    d = tpr - fpr
    if d <= 0:
        return float("nan")
    return float(np.clip((q - fpr) / d, 0.0, 1.0))


@dataclass(frozen=True)
class PrevalenceEstimate:
    event: str
    n: int
    true_prevalence: float
    baseline: float               # training-average prevalence (source share)
    em: float
    bbse: float
    youden: float                 # source tpr - fpr at the 0.5 threshold
    abstain: bool
    reason: str
    estimate: float               # EM, or NaN when abstaining
    count_estimate: float         # estimate * n
    rel_error: float              # |estimate - true| / true (NaN when abstaining)
    baseline_rel_error: float


def estimate_event(target_scores: np.ndarray, source_scores: np.ndarray, source_y: np.ndarray, event: str = "",
                   target_y: np.ndarray | None = None, threshold: float = 0.5) -> PrevalenceEstimate:
    """C3 for one target event. ``target_y`` is used only to score the estimate afterwards."""
    s_src = np.asarray(source_scores, dtype=float)
    y_src = np.asarray(source_y, dtype=bool)
    s_tgt = np.asarray(target_scores, dtype=float)
    if s_tgt.size == 0:
        raise RecalError(f"{event}: no target scores")
    pi_s = float(y_src.mean())
    a, b = platt_fit(s_src, y_src)
    em, _ = em_prior(platt_apply(s_tgt, a, b), pi_s)
    hat_src = s_src >= threshold
    tpr, fpr = float(hat_src[y_src].mean()), float(hat_src[~y_src].mean())
    youden = tpr - fpr
    bbse = bbse_prior(float((s_tgt >= threshold).mean()), tpr, fpr)
    reasons = []
    if not youden >= MIN_YOUDEN:
        reasons.append(f"source tpr - fpr {youden:.2f} < {MIN_YOUDEN}")
    if not np.isfinite(bbse) or abs(em - bbse) > MAX_DISAGREEMENT * max(em, bbse, EPS):
        reasons.append("EM and BBSE disagree by more than 50%")
    abstain = bool(reasons)
    est = float("nan") if abstain else em
    true = float(np.asarray(target_y, dtype=bool).mean()) if target_y is not None else float("nan")
    rel = abs(est - true) / true if (true > 0 and not abstain) else float("nan")
    base_rel = abs(pi_s - true) / true if true > 0 else float("nan")
    return PrevalenceEstimate(event, int(s_tgt.size), true, pi_s, em, bbse, youden, abstain, "; ".join(reasons),
                              est, est * s_tgt.size if not abstain else float("nan"), rel, base_rel)


def estimate_all(fold_scores: pd.DataFrame, encoder: str, pooling: str,
                 source_roles: tuple[str, ...] = ("development", "secondary")) -> pd.DataFrame:
    """C3 for every held-out event of one (encoder, pooling) setting. Each target's source = the other events."""
    s = fold_scores[(fold_scores["encoder"] == encoder) & (fold_scores["pooling"] == pooling)]
    if s.empty:
        raise RecalError(f"no fold scores for {encoder} / {pooling}")
    rows = []
    for ev in sorted(s["event"].unique()):
        tgt = s[s["event"] == ev]
        src = s[(s["event"] != ev) & s["role"].isin(source_roles) & s["damaged"].notna()]
        y_t = tgt["damaged"].astype(bool).to_numpy() if tgt["damaged"].notna().all() else None
        r = estimate_event(tgt["score"].to_numpy(float), src["score"].to_numpy(float),
                           src["damaged"].astype(bool).to_numpy(), ev, y_t)
        rows.append({"encoder": encoder, "pooling": pooling, **asdict(r)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class C3Gate:
    n_events: int
    n_within: int          # rel_error <= 0.5 (abstentions count as misses)
    n_beat_baseline: int   # rel_error < baseline_rel_error (abstentions count as misses)
    n_abstain: int

    @property
    def passed(self) -> bool:
        return self.n_within >= MIN_EVENTS and self.n_beat_baseline >= MIN_EVENTS


def c3_gate(estimates: pd.DataFrame) -> C3Gate:
    e = estimates
    ok = ~e["abstain"].astype(bool)
    within = ok & (e["rel_error"] <= MAX_REL_ERROR)
    beat = ok & (e["rel_error"] < e["baseline_rel_error"])
    return C3Gate(len(e), int(within.sum()), int(beat.sum()), int((~ok).sum()))
