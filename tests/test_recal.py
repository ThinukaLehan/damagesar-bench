import numpy as np
import pandas as pd
import pytest

from damagesar_bench.recal.prevalence import (RecalError, bbse_prior, c3_gate, em_prior, estimate_all,
                                              estimate_event, platt_apply, platt_fit)


def scores(n, prev, sep, rng):
    """Probe-like probabilities: logit ~ N(+sep/2, 1) for damaged, N(-sep/2, 1) for intact."""
    y = rng.random(n) < prev
    z = np.where(y, sep / 2, -sep / 2) + rng.normal(size=n)
    return 1 / (1 + np.exp(-z)), y


def test_em_recovers_a_shifted_prior_with_exact_posteriors():
    rng = np.random.default_rng(0)
    # exact posteriors under source prior 0.5 for a target whose prior is 0.2
    y = rng.random(200_000) < 0.2
    z = np.where(y, 1.0, -1.0) + rng.normal(size=y.size)
    lik1, lik0 = np.exp(-(z - 1) ** 2 / 2), np.exp(-(z + 1) ** 2 / 2)
    post = 0.5 * lik1 / (0.5 * lik1 + 0.5 * lik0)
    est, it = em_prior(post, 0.5)
    assert abs(est - 0.2) < 0.01 and it > 1


def test_bbse_hand_worked():
    assert bbse_prior(0.5, tpr=0.8, fpr=0.2) == pytest.approx(0.5)
    assert bbse_prior(0.26, tpr=0.8, fpr=0.2) == pytest.approx(0.1)
    assert bbse_prior(0.1, tpr=0.8, fpr=0.2) == 0.0             # clipped
    assert np.isnan(bbse_prior(0.3, tpr=0.2, fpr=0.2))           # no separation


def test_platt_is_monotone_and_needs_two_classes():
    rng = np.random.default_rng(1)
    s, y = scores(5000, 0.3, 2.0, rng)
    a, b = platt_fit(s, y)
    assert a > 0
    p = platt_apply(np.array([0.2, 0.5, 0.8]), a, b)
    assert np.all(np.diff(p) > 0)
    with pytest.raises(RecalError):
        platt_fit(s, np.ones_like(y))


def test_estimate_event_close_on_an_informative_probe_and_beats_the_baseline():
    rng = np.random.default_rng(2)
    s_src, y_src = scores(40_000, 0.3, 3.0, rng)
    s_tgt, y_tgt = scores(20_000, 0.08, 3.0, rng)
    r = estimate_event(s_tgt, s_src, y_src, "t", y_tgt)
    assert not r.abstain
    assert r.rel_error < 0.3 and r.rel_error < r.baseline_rel_error
    assert r.count_estimate == pytest.approx(r.estimate * 20_000)


def test_estimate_event_abstains_on_an_uninformative_probe():
    rng = np.random.default_rng(3)
    s_src, y_src = scores(20_000, 0.3, 0.1, rng)
    s_tgt, y_tgt = scores(10_000, 0.6, 0.1, rng)
    r = estimate_event(s_tgt, s_src, y_src, "t", y_tgt)
    assert r.abstain and "tpr - fpr" in r.reason
    assert np.isnan(r.estimate) and np.isnan(r.rel_error) and np.isnan(r.count_estimate)


def test_estimate_all_uses_other_events_as_source_and_the_gate_counts():
    rng = np.random.default_rng(4)
    parts = []
    for ev, prev in [("a", 0.05), ("b", 0.2), ("c", 0.4), ("d", 0.6), ("e", 0.1)]:
        s, y = scores(6000, prev, 3.0, rng)
        parts.append(pd.DataFrame({"event": ev, "role": "development", "damaged": y, "score": s,
                                   "encoder": "enc", "pooling": "outline"}))
    fs = pd.concat(parts, ignore_index=True)
    est = estimate_all(fs, "enc", "outline")
    assert est["event"].tolist() == ["a", "b", "c", "d", "e"]
    assert est["baseline"].between(0, 1).all()
    g = c3_gate(est)
    assert g.n_events == 5 and g.n_within >= 4 and g.n_beat_baseline >= 4 and g.passed
    with pytest.raises(RecalError):
        estimate_all(fs, "enc", "ring10m")


def test_gate_counts_abstentions_as_misses():
    e = pd.DataFrame({"abstain": [True, False, False, False, False],
                      "rel_error": [np.nan, 0.1, 0.2, 0.6, 0.3],
                      "baseline_rel_error": [0.5, 0.5, 0.1, 0.9, 0.9]})
    g = c3_gate(e)
    assert (g.n_within, g.n_beat_baseline, g.n_abstain, g.passed) == (3, 3, 1, False)
