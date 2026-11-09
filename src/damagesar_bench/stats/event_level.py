"""Across-event inference: the claim layer (FYP Master §5.0-5.2, v11.1).

The event is the unit. The per-event gains (e.g. ΔAUROC of DINOv3-LVD over ImageNet ViT-B
on each confirmatory event) are summarised by a t-interval with n-1 degrees of freedom,
and the interval is read with the ordered v11.1 decision table:

    1  lower > 0  and estimate >= +0.02   -> supported (worthwhile gain)
    2  lower > 0  and estimate <  +0.02   -> small positive effect, below worthwhile
    3  lower <= 0 and upper    <  +0.02   -> not a worthwhile gain
    4  anything else                      -> inconclusive

Rows are checked in order; the first match wins. Interval level: 95%, or 97.5% per claim
under the Bonferroni split for the two primary claims (RQ1, RQ2).

Supporting evidence only (never the decision): the exact two-sided sign-flip test and the
Hartung-Knapp random-effects interval.

All comparisons are made after rounding to ``ROUND_DECIMALS`` decimals, so floating-point
noise (0.1 - 0.08 = 0.020000000000000004) cannot change a verdict. Rounding to 10 decimals
is far below any reported precision.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy import stats as st
from scipy.optimize import brentq

WORTHWHILE = 0.02          # smallest worthwhile gain in AUROC (pre-registered)
ROUND_DECIMALS = 10
MAX_EXACT_SIGNFLIP = 20    # 2**20 sign patterns; more events would need Monte Carlo


def _gains(gains) -> np.ndarray:
    g = np.asarray(gains, dtype=float)
    if g.ndim != 1:
        raise ValueError("gains must be 1-D (one value per event)")
    if len(g) < 2:
        raise ValueError("need at least 2 events")
    if not np.isfinite(g).all():
        raise ValueError("gains must be finite (an event with an undefined gain must be dropped explicitly)")
    return g


def bonferroni_level(n_claims: int = 2, family_level: float = 0.95) -> float:
    """Per-claim two-sided confidence level, e.g. 2 claims at family 95% -> 97.5%."""
    if n_claims < 1:
        raise ValueError("n_claims must be >= 1")
    return 1 - (1 - family_level) / n_claims


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float
    level: float
    n: int
    df: int
    se: float
    method: str


def t_interval(gains, level: float = 0.95) -> Interval:
    """Two-sided t-interval for the mean per-event gain (df = n - 1)."""
    g = _gains(gains)
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    n = len(g)
    mean = float(g.mean())
    se = float(g.std(ddof=1) / math.sqrt(n))
    q = float(st.t.ppf(1 - (1 - level) / 2, df=n - 1))
    return Interval(mean, mean - q * se, mean + q * se, level, n, n - 1, se, "t")


class Verdict(Enum):
    SUPPORTED = (1, "Supported: a worthwhile gain")
    SMALL_POSITIVE = (2, "Small positive effect, below worthwhile")
    NOT_WORTHWHILE = (3, "Not a worthwhile gain")
    INCONCLUSIVE = (4, "Inconclusive")

    @property
    def row(self) -> int:
        return self.value[0]

    @property
    def meaning(self) -> str:
        return self.value[1]


def decide(lower: float, estimate: float, upper: float, worthwhile: float = WORTHWHILE) -> Verdict:
    """Apply the ordered v11.1 decision table to an event-level interval."""
    for v in (lower, estimate, upper):
        if not math.isfinite(v):
            raise ValueError("interval values must be finite")
    lo, est, hi, w = (round(v, ROUND_DECIMALS) for v in (lower, estimate, upper, worthwhile))
    if not lo <= est <= hi:
        raise ValueError(f"need lower <= estimate <= upper, got {lower}, {estimate}, {upper}")
    if lo > 0 and est >= w:
        return Verdict.SUPPORTED
    if lo > 0:
        return Verdict.SMALL_POSITIVE
    if hi < w:
        return Verdict.NOT_WORTHWHILE
    return Verdict.INCONCLUSIVE


def decide_interval(iv: Interval, worthwhile: float = WORTHWHILE) -> Verdict:
    return decide(iv.lower, iv.estimate, iv.upper, worthwhile)


@dataclass(frozen=True)
class SignFlip:
    statistic: float      # sum of gains
    p_value: float        # exact two-sided
    n_patterns: int
    min_p: float          # smallest p-value possible with these events (no zero gains)


def sign_flip_test(gains) -> SignFlip:
    """Exact two-sided sign-flip test of 'gains symmetric around 0' (statistic: |sum|).

    Assumes independent events with gains symmetric under no effect. With 6 events the
    smallest possible p is 2/64; with 5 events 2/32 (> 0.05), which is why this test is
    supporting evidence only.
    """
    g = _gains(gains)
    n = len(g)
    if n > MAX_EXACT_SIGNFLIP:
        raise ValueError(f"exact test limited to {MAX_EXACT_SIGNFLIP} events")
    obs = abs(g.sum())
    signs = np.array(list(itertools.product((1, -1), repeat=n)))
    sums = np.abs(signs @ g)
    tol = 1e-12 * max(1.0, obs)
    count = int(np.sum(sums >= obs - tol))
    return SignFlip(float(g.sum()), count / 2**n, 2**n, 2 / 2**n)


def mde(n_events: int, sd: float = 1.0, alpha: float = 0.05, power: float = 0.80,
        dist: str = "exact") -> float:
    """Minimum detectable mean gain, two-sided one-sample t-test at ``alpha``, given power.

    exact:  solves the power equation with the noncentral t distribution (pre-registered)
    t:      (t_{1-alpha/2, n-1} + t_{power, n-1}) * sd / sqrt(n)   (quick approximation)
    normal: (z_{1-alpha/2} + z_{power}) * sd / sqrt(n)               (large-sample lower bound)
    n = 6, alpha 0.05, power 0.8: exact 1.43 SD, t-approx 1.42 SD, normal 1.14 SD.
    n = 5 (Ukraine fallback): exact 1.68 SD.
    """
    if n_events < 2:
        raise ValueError("need at least 2 events")
    if not (0 < alpha < 1 and 0 < power < 1):
        raise ValueError("alpha and power must be in (0, 1)")
    df = n_events - 1
    if dist == "normal":
        q = st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)
    elif dist == "t":
        q = st.t.ppf(1 - alpha / 2, df) + st.t.ppf(power, df)
    elif dist == "exact":
        if n_events < 3:
            raise ValueError("exact MDE needs at least 3 events")
        tc = st.t.ppf(1 - alpha / 2, df)
        root_n = math.sqrt(n_events)

        def power_at(d):
            nc = d * root_n
            return st.nct.sf(tc, df, nc) + st.nct.cdf(-tc, df, nc) - power

        # bracket: the normal value is a lower bound; widen upwards from the t
        # approximation until power is reached (nct gives NaN for absurdly large nc)
        lo = (st.norm.ppf(1 - alpha / 2) + st.norm.ppf(power)) / root_n * 0.5
        hi = (st.t.ppf(1 - alpha / 2, df) + st.t.ppf(power, df)) / root_n
        while power_at(hi) < 0:
            hi *= 1.5
        if not math.isfinite(power_at(hi)):
            raise ValueError("power equation could not be solved for these inputs")
        q = brentq(power_at, lo, hi) * root_n
    else:
        raise ValueError("dist must be 'exact', 't' or 'normal'")
    return float(q * sd / math.sqrt(n_events))


class PilotAction(Enum):
    PROCEED = "SD <= 0.020: proceed as planned"
    LEAD_WITH_ESTIMATE = "0.020 < SD <= 0.035: lead with the effect estimate and interval"
    LEAD_WITH_SECONDARY = "SD > 0.035: lead with the secondary estimate"


def pilot_sd_rule(sd: float) -> PilotAction:
    """Pilot SD rule (FYP Master §5.2) for the between-event SD of the gain."""
    if not math.isfinite(sd) or sd < 0:
        raise ValueError("sd must be a finite, non-negative number")
    s = round(sd, ROUND_DECIMALS)
    if s <= 0.020:
        return PilotAction.PROCEED
    if s <= 0.035:
        return PilotAction.LEAD_WITH_ESTIMATE
    return PilotAction.LEAD_WITH_SECONDARY


def hartung_knapp(gains, variances, level: float = 0.95) -> Interval:
    """Random-effects mean with the Hartung-Knapp interval (DerSimonian-Laird tau^2).

    Secondary analysis only. ``variances`` are the within-event variances of each gain
    (e.g. squared DeLong or bootstrap standard errors). No ad-hoc truncation of the HK
    scale factor is applied.
    """
    y = _gains(gains)
    v = np.asarray(variances, dtype=float)
    if v.shape != y.shape or not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("variances must be positive, finite and one per event")
    k = len(y)
    w = 1 / v
    mu_fe = np.sum(w * y) / w.sum()
    q = np.sum(w * (y - mu_fe) ** 2)
    c = w.sum() - np.sum(w**2) / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (v + tau2)
    mu = float(np.sum(ws * y) / ws.sum())
    se = float(math.sqrt(np.sum(ws * (y - mu) ** 2) / ((k - 1) * ws.sum())))
    t = float(st.t.ppf(1 - (1 - level) / 2, k - 1))
    return Interval(mu, mu - t * se, mu + t * se, level, k, k - 1, se, f"hartung-knapp (tau2={tau2:.6g})")
