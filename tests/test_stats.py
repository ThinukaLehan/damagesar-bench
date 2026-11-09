"""Statistics engine: reference values worked by hand or taken from printed tables.

Every expected number here is independent of the code under test: hand arithmetic,
published t/z table values, a brute-force O(m*n) DeLong, scikit-learn, or statsmodels.
"""

import math

import numpy as np
import pytest

from damagesar_bench import stats as S
from damagesar_bench.stats.bootstrap import cluster_bootstrap_paired_auc

# ------------------------------------------------------------------ t-interval


def test_t_interval_hand_worked():
    # gains: mean 0.03; deviations -0.02,0,0.02,-0.01,0.01,0 -> SS 0.001, var 0.0002
    g = [0.01, 0.03, 0.05, 0.02, 0.04, 0.03]
    iv = S.t_interval(g, 0.95)
    se = math.sqrt(0.0002) / math.sqrt(6)            # 0.0057735
    assert iv.estimate == pytest.approx(0.03)
    assert iv.se == pytest.approx(se)
    assert iv.df == 5 and iv.n == 6
    # t(0.975, 5) = 2.5706 (printed tables)
    assert iv.lower == pytest.approx(0.03 - 2.5706 * se, abs=1e-6)
    assert iv.upper == pytest.approx(0.03 + 2.5706 * se, abs=1e-6)


def test_t_interval_bonferroni_level_is_wider():
    g = [0.01, 0.03, 0.05, 0.02, 0.04, 0.03]
    lvl = S.bonferroni_level(2)
    assert lvl == pytest.approx(0.975)
    iv = S.t_interval(g, lvl)
    se = math.sqrt(0.0002) / math.sqrt(6)
    # t(0.9875, 5) = 3.1634 (printed tables)
    assert iv.upper - iv.estimate == pytest.approx(3.1634 * se, abs=1e-5)
    assert iv.lower < S.t_interval(g, 0.95).lower


def test_t_interval_five_events_uses_df4():
    iv = S.t_interval([0.01, 0.02, 0.03, 0.04, 0.05], 0.95)
    se = math.sqrt(0.00025) / math.sqrt(5)
    assert iv.df == 4
    # t(0.975, 4) = 2.7764
    assert iv.upper - iv.estimate == pytest.approx(2.7764 * se, abs=1e-6)


@pytest.mark.parametrize("bad", [[0.1], [], [0.1, float("nan")], [[0.1, 0.2]]])
def test_t_interval_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        S.t_interval(bad)


def test_t_interval_zero_spread_is_a_point():
    iv = S.t_interval([0.03] * 6)
    assert iv.lower == iv.upper == pytest.approx(0.03)


def test_bonferroni_level_validation():
    assert S.bonferroni_level(1) == pytest.approx(0.95)
    with pytest.raises(ValueError):
        S.bonferroni_level(0)

# ------------------------------------------------------------------ decision table


@pytest.mark.parametrize("lo, est, hi, row", [
    (0.010, 0.030, 0.050, 1),   # lower > 0, estimate >= 0.02
    (0.001, 0.020, 0.039, 1),   # estimate exactly at the threshold counts as worthwhile
    (0.005, 0.015, 0.025, 2),   # lower > 0, estimate < 0.02
    (-0.010, 0.000, 0.015, 3),  # lower <= 0, upper < 0.02
    (-0.050, -0.030, -0.010, 3),  # clearly negative is also "not worthwhile" (row 3)
    (0.000, 0.005, 0.010, 3),   # lower exactly 0 is NOT > 0 -> falls to row 3
    (-0.010, 0.010, 0.030, 4),  # crosses 0 and reaches 0.02 -> inconclusive
    (-0.010, 0.005, 0.020, 4),  # upper exactly 0.02 is NOT < 0.02 -> inconclusive
])
def test_decision_rows(lo, est, hi, row):
    assert S.decide(lo, est, hi).row == row


def test_decision_rows_are_exhaustive_and_ordered():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        a, b, c = np.sort(rng.normal(0.01, 0.03, 3))
        v = S.decide(a, b, c)
        # independent re-statement of the table
        if a > 0 and b >= 0.02:
            exp = 1
        elif a > 0:
            exp = 2
        elif c < 0.02:
            exp = 3
        else:
            exp = 4
        assert v.row == exp


def test_decision_ignores_float_noise():
    est = 0.1 - 0.08            # 0.020000000000000004 in floating point
    assert S.decide(0.01, est, 0.03) is S.Verdict.SUPPORTED
    assert S.decide(0.01, 0.02 - 1e-15, 0.03) is S.Verdict.SUPPORTED   # noise below 1e-10
    assert S.decide(0.01, 0.0199, 0.03) is S.Verdict.SMALL_POSITIVE


def test_decision_validation_and_meanings():
    with pytest.raises(ValueError):
        S.decide(0.03, 0.02, 0.01)
    with pytest.raises(ValueError):
        S.decide(float("nan"), 0.0, 0.1)
    assert S.Verdict.INCONCLUSIVE.meaning == "Inconclusive"
    iv = S.t_interval([0.01, 0.03, 0.05, 0.02, 0.04, 0.03])
    assert S.decide_interval(iv) is S.Verdict.SUPPORTED

# ------------------------------------------------------------------ sign-flip


def test_sign_flip_minimum_p_values():
    assert S.sign_flip_test([0.01] * 6).p_value == pytest.approx(2 / 64)
    assert S.sign_flip_test([0.01] * 5).p_value == pytest.approx(2 / 32)
    assert S.sign_flip_test([0.01] * 5).p_value > 0.05     # can never reach 0.05 with 5 events
    assert S.sign_flip_test([0.01] * 6).min_p == pytest.approx(2 / 64)


def test_sign_flip_hand_enumerated():
    # +-1 +-2 +-3: |sums| = 6,0,2,4,4,2,0,6
    assert S.sign_flip_test([1, 2, 3]).p_value == pytest.approx(2 / 8)    # |6|
    assert S.sign_flip_test([1, -2, 3]).p_value == pytest.approx(6 / 8)   # |2|
    assert S.sign_flip_test([1, 2, -3]).p_value == pytest.approx(1.0)     # |0|


def test_sign_flip_is_symmetric_and_bounded():
    g = [0.03, -0.01, 0.02, 0.05, 0.00, 0.01]
    a, b = S.sign_flip_test(g), S.sign_flip_test([-x for x in g])
    assert a.p_value == b.p_value and 0 < a.p_value <= 1
    with pytest.raises(ValueError):
        S.sign_flip_test([0.01] * 21)

# ------------------------------------------------------------------ power, pilot rule


def test_mde_matches_master_numbers():
    # FYP Master §5.2: 1.14 SD (normal) to 1.43 SD (t, 5 df) with 6 events
    assert S.mde(6, dist="normal") == pytest.approx((1.95996 + 0.84162) / math.sqrt(6), abs=1e-4)
    assert round(S.mde(6, dist="normal"), 2) == 1.14
    # t(0.975,5) = 2.5706, t(0.80,5) = 0.9195 -> quick approximation gives 1.42, not 1.43
    assert S.mde(6, dist="t") == pytest.approx((2.5706 + 0.9195) / math.sqrt(6), abs=1e-3)
    assert round(S.mde(6, dist="t"), 2) == 1.42
    # the Master's 1.43 SD is the exact (noncentral t) value; default method
    assert round(S.mde(6), 2) == 1.43
    assert round(S.mde(5), 2) == 1.68
    assert S.mde(6, sd=0.02) == pytest.approx(0.02 * S.mde(6))
    with pytest.raises(ValueError):
        S.mde(1)
    with pytest.raises(ValueError):
        S.mde(6, dist="x")
    with pytest.raises(ValueError):
        S.mde(6, alpha=0)
    with pytest.raises(ValueError):
        S.mde(2)                       # exact method needs >= 3 events
    assert S.mde(2, dist="normal") > 0


def test_exact_mde_really_has_80_percent_power():
    # independent check by simulation: one-sample t-test, true mean = MDE, sd = 1, n = 6
    from scipy import stats as st
    rng = np.random.default_rng(11)
    d = S.mde(6)
    x = rng.normal(d, 1.0, size=(40000, 6))
    p = st.ttest_1samp(x, 0.0, axis=1).pvalue
    assert np.mean(p < 0.05) == pytest.approx(0.80, abs=0.01)


@pytest.mark.parametrize("sd, action", [
    (0.0, S.PilotAction.PROCEED), (0.020, S.PilotAction.PROCEED),
    (0.0201, S.PilotAction.LEAD_WITH_ESTIMATE), (0.035, S.PilotAction.LEAD_WITH_ESTIMATE),
    (0.0351, S.PilotAction.LEAD_WITH_SECONDARY),
])
def test_pilot_sd_rule(sd, action):
    assert S.pilot_sd_rule(sd) is action


def test_pilot_sd_rule_rejects_negative():
    with pytest.raises(ValueError):
        S.pilot_sd_rule(-0.01)

# ------------------------------------------------------------------ Hartung-Knapp


def test_hartung_knapp_equals_t_interval_with_equal_variances():
    # With equal within-event variances all weights are equal, so HK reduces exactly to
    # the ordinary t-interval over the gains (whatever tau^2 is).
    g = [0.01, 0.03, 0.05, 0.02, 0.04, 0.03]
    hk = S.hartung_knapp(g, [0.0004] * 6)
    t = S.t_interval(g)
    assert hk.estimate == pytest.approx(t.estimate)
    assert hk.lower == pytest.approx(t.lower) and hk.upper == pytest.approx(t.upper)


def test_hartung_knapp_hand_worked_tau2():
    # y = 0, 0.1; v = 0.01, 0.01 -> FE mean 0.05, Q = 100*(0.0025+0.0025) = 0.5 < k-1 = 1 -> tau2 = 0
    hk = S.hartung_knapp([0.0, 0.1], [0.01, 0.01])
    assert "tau2=0" in hk.method
    # y = 0, 1; v = 0.01 -> Q = 100*0.5 = 50; C = 200 - 20000/200 = 100; tau2 = 49/100 = 0.49
    hk2 = S.hartung_knapp([0.0, 1.0], [0.01, 0.01])
    assert "tau2=0.49" in hk2.method


def test_hartung_knapp_matches_statsmodels():
    sm = pytest.importorskip("statsmodels.stats.meta_analysis")
    y = np.array([0.031, 0.012, -0.004, 0.045, 0.020, 0.008])
    v = np.array([0.0002, 0.0004, 0.0001, 0.0006, 0.0003, 0.0002])
    res = sm.combine_effects(y, v, method_re="dl", use_t=True)
    frame = res.summary_frame()
    row = frame.loc["random effect wls"]   # statsmodels' HKSJ variance with t quantiles
    hk = S.hartung_knapp(y, v)
    assert hk.estimate == pytest.approx(row["eff"], abs=1e-10)
    assert hk.lower == pytest.approx(row["ci_low"], abs=1e-8)
    assert hk.upper == pytest.approx(row["ci_upp"], abs=1e-8)


def test_hartung_knapp_validation():
    with pytest.raises(ValueError):
        S.hartung_knapp([0.1, 0.2], [0.01, 0.0])
    with pytest.raises(ValueError):
        S.hartung_knapp([0.1, 0.2], [0.01])

# ------------------------------------------------------------------ AUROC, AP, DeLong


def _toy(seed=0, n=300, ties=False):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(int)
    a = y * 0.8 + rng.normal(0, 1, n)
    b = y * 0.4 + rng.normal(0, 1, n) + 0.3 * a
    if ties:
        a, b = np.round(a, 1), np.round(b, 1)
    return y, a, b


def test_auroc_hand_worked():
    # positives 0.9, 0.4 ; negatives 0.4, 0.1 -> pairs: (0.9>0.4)1 (0.9>0.1)1 (0.4=0.4)0.5 (0.4>0.1)1 -> 3.5/4
    assert S.auroc([1, 1, 0, 0], [0.9, 0.4, 0.4, 0.1]) == pytest.approx(0.875)


@pytest.mark.parametrize("ties", [False, True])
def test_auroc_and_ap_match_sklearn(ties):
    skm = pytest.importorskip("sklearn.metrics")
    y, a, _ = _toy(ties=ties)
    assert S.auroc(y, a) == pytest.approx(skm.roc_auc_score(y, a), abs=1e-12)
    assert S.average_precision(y, a) == pytest.approx(skm.average_precision_score(y, a), abs=1e-12)


def test_metrics_reject_single_class_and_bad_labels():
    with pytest.raises(S.MetricError):
        S.auroc([0, 0, 0], [0.1, 0.2, 0.3])
    with pytest.raises(S.MetricError):
        S.auroc([0, 2, 1], [0.1, 0.2, 0.3])
    with pytest.raises(S.MetricError):
        S.auroc([0, 1], [0.1])
    with pytest.raises(S.MetricError):
        S.auroc([0, 1], [0.1, float("inf")])


def _delong_bruteforce(y, scores):
    """Textbook DeLong (1988) with explicit pairwise kernels: O(m*n), independent of ranks."""
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    m, n = len(pos), len(neg)
    v10, v01, aucs = [], [], []
    for s in scores:
        s = np.asarray(s, float)
        psi = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        aucs.append(psi.mean())
        v10.append(psi.mean(axis=1))
        v01.append(psi.mean(axis=0))
    s10 = np.cov(np.array(v10))
    s01 = np.cov(np.array(v01))
    return np.array(aucs), np.atleast_2d(s10) / m + np.atleast_2d(s01) / n


@pytest.mark.parametrize("ties", [False, True])
def test_delong_matches_bruteforce(ties):
    y, a, b = _toy(seed=3, n=200, ties=ties)
    d = S.delong(y, a, b)
    aucs, cov = _delong_bruteforce(y, [a, b])
    np.testing.assert_allclose(d.auc, aucs, atol=1e-12)
    np.testing.assert_allclose(d.cov, cov, atol=1e-12)
    diff, se = S.delong_paired_diff(y, a, b)
    assert diff == pytest.approx(aucs[0] - aucs[1])
    assert se == pytest.approx(math.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))


def test_delong_single_score_and_identical_scores():
    y, a, _ = _toy(seed=4)
    d = S.delong(y, a)
    assert d.auc.shape == (1,) and d.se[0] > 0
    diff, se = S.delong_paired_diff(y, a, a)
    assert diff == 0 and se == pytest.approx(0, abs=1e-12)

# ------------------------------------------------------------------ cluster bootstrap


def test_bootstrap_point_estimate_and_determinism():
    y, a, b = _toy(seed=5, n=400)
    tiles = np.arange(400) // 10                     # 40 tiles of 10 buildings
    r1 = cluster_bootstrap_paired_auc(y, a, b, tiles, n_boot=300, seed=7)
    r2 = cluster_bootstrap_paired_auc(y, a, b, tiles, n_boot=300, seed=7)
    assert r1 == r2
    assert r1.estimate == pytest.approx(S.auroc(y, a) - S.auroc(y, b))
    assert r1.lower < r1.estimate < r1.upper
    assert r1.n_valid + r1.n_skipped == 300 and r1.reliable


def test_bootstrap_identical_scores_give_zero_interval():
    y, a, _ = _toy(seed=6, n=200)
    r = cluster_bootstrap_paired_auc(y, a, a, np.arange(200) // 5, n_boot=100)
    assert r.estimate == r.lower == r.upper == 0


def test_bootstrap_respects_clusters():
    # Strong within-tile correlation: the cluster interval must be wider than the
    # interval that treats every building as its own cluster.
    rng = np.random.default_rng(8)
    tiles = np.repeat(np.arange(30), 20)
    tile_effect = rng.normal(0, 1.0, 30)[tiles]
    y = (rng.random(600) < 0.3).astype(int)
    a = y * 0.6 + rng.normal(0, 0.5, 600) + tile_effect * y
    b = y * 0.6 + rng.normal(0, 0.5, 600)
    wide = cluster_bootstrap_paired_auc(y, a, b, tiles, n_boot=400, seed=1)
    narrow = cluster_bootstrap_paired_auc(y, a, b, np.arange(600), n_boot=400, seed=1)
    assert (wide.upper - wide.lower) > (narrow.upper - narrow.lower)


def test_bootstrap_skips_single_class_replicates_and_flags():
    # one tile holds the only damaged buildings: many replicates miss it entirely
    y = np.r_[np.ones(3), np.zeros(57)].astype(int)
    a = np.r_[np.full(3, 0.9), np.linspace(0, 0.5, 57)]
    b = np.r_[np.full(3, 0.4), np.linspace(0, 0.5, 57)]
    tiles = np.r_[np.zeros(3), np.repeat(np.arange(1, 20), 3)]
    r = cluster_bootstrap_paired_auc(y, a, b, tiles, n_boot=200, seed=2)
    assert r.n_skipped > 0 and not r.reliable


def test_bootstrap_validation():
    with pytest.raises(ValueError):
        cluster_bootstrap_paired_auc([0, 1], [0.1, 0.2], [0.1, 0.2], [0])
    with pytest.raises(ValueError):
        cluster_bootstrap_paired_auc([0, 1], [0.1, 0.2], [0.1, 0.2], [0, 1], level=1.0)

# ------------------------------------------------------------------ Gate 1


DEV = ["beirut", "la_palma", "hawaii", "congo", "libya"]


def test_gate1_pass():
    g = dict(zip(DEV, [0.03, 0.02, 0.04, -0.01, 0.05]))   # mean 0.026, 4 of 5 better
    r = S.gate1(g, null_mean_gains=np.linspace(-0.01, 0.01, 101))
    assert r.mean_gain == pytest.approx(0.026) and r.n_better == 4 and r.passed


def test_gate1_fails_each_condition():
    null = [0.0]
    assert not S.gate1(dict(zip(DEV, [0.02, 0.02, 0.02, 0.0, 0.01])), null).mean_ok   # mean 0.014
    r = S.gate1(dict(zip(DEV, [0.10, 0.05, -0.01, -0.01, 0.01])), null)               # 3 better
    assert r.mean_ok and not r.count_ok and not r.passed
    r = S.gate1(dict(zip(DEV, [0.03] * 5)), [0.04])                                    # null higher
    assert r.mean_ok and r.count_ok and not r.null_ok and not r.passed


def test_gate1_boundaries():
    # mean exactly 0.02 (with float noise) passes the mean condition; a gain of exactly 0 is not "better"
    r = S.gate1(dict(zip(DEV, [0.1 - 0.08] * 4 + [0.02])), [0.0])
    assert r.mean_ok and r.passed
    assert S.gate1(dict(zip(DEV, [0.05, 0.05, 0.0, 0.0, 0.0])), [0.0]).n_better == 2
    with pytest.raises(ValueError):
        S.gate1({}, [0.0])
    with pytest.raises(ValueError):
        S.gate1(dict(zip(DEV, [0.02] * 5)), [])
