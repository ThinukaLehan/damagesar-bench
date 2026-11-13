import numpy as np
import pandas as pd
import pytest

from damagesar_bench import contracts as C
from damagesar_bench.probe.logistic import ProbeError, run_probes
from damagesar_bench.stats.compare import compare, event_metrics
from damagesar_bench.stats.event_level import Verdict


def synth(events_cfg, today_locked=True, flip=(), signal=2.0, n=120, dim=6, seed=0,
          encoder="fake", pooling="outline", events=("beirut", "hawaii", "haiti", "marshall")):
    """Synthetic building features. Dimension 0 carries the damage signal (sign flipped for
    events in ``flip``); the rest is noise. Each event gets its own offset."""
    rng = np.random.default_rng(seed)
    rows = []
    for k, ev in enumerate(events):
        role = events_cfg.events[ev].role
        locked = role == "confirmatory" and today_locked
        y = rng.random(n) < 0.3
        x = rng.normal(0, 1, (n, dim)) + k * 0.5
        x[:, 0] += (-signal if ev in flip else signal) * y
        for i in range(n):
            r = {"event": ev, "tile_id": f"{ev}-t_{i // 10:08d}", "instance_id": i, "role": role,
                 "damaged": pd.NA if locked else bool(y[i]), "labels_locked": locked,
                 "area_m2": 80.0, "encoder": encoder, "pooling": pooling}
            r.update({C.feature_name(j): x[i, j] for j in range(dim)})
            rows.append(r)
    df = pd.DataFrame(rows)
    df["damaged"] = df["damaged"].astype("boolean")
    return df


def test_folds_before_freeze_hold_out_dev_only_and_never_train_on_confirmatory(events_cfg, before_freeze):
    scores, fits = run_probes(synth(events_cfg), events_cfg, before_freeze)
    assert sorted(scores["event"].unique()) == ["beirut", "hawaii"]
    assert {f.held_out: f.train_events for f in fits} == {
        "beirut": ("hawaii", "marshall"), "hawaii": ("beirut", "marshall")}
    assert not scores["train_events"].str.contains("haiti").any()
    assert ((scores["score"] > 0) & (scores["score"] < 1)).all()


def test_probe_learns_the_signal(events_cfg, before_freeze):
    scores, _ = run_probes(synth(events_cfg), events_cfg, before_freeze)
    m = event_metrics(scores, events_cfg, n_boot=200)
    assert (m["auroc"] > 0.8).all()


def test_held_out_event_does_not_leak_into_its_probe(events_cfg, before_freeze):
    # Beirut's damage signal points the other way. A probe trained without Beirut must
    # therefore score Beirut badly (AUROC well below 0.5). Any leak would pull it up.
    scores, _ = run_probes(synth(events_cfg, flip=("beirut",)), events_cfg, before_freeze)
    m = event_metrics(scores, events_cfg, n_boot=200).set_index("event")
    assert m.loc["beirut", "auroc"] < 0.2
    # Hawaii's probe trains on Beirut (flipped) + Marshall (normal): the signals cancel
    assert 0.35 < m.loc["hawaii", "auroc"] < 0.65


def test_matches_a_hand_built_sklearn_pipeline(events_cfg, before_freeze):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = synth(events_cfg)
    scores, _ = run_probes(df, events_cfg, before_freeze)
    feats = C.feature_columns(df)
    train = df[df["event"].isin(["hawaii", "marshall"])]
    test = df[df["event"] == "beirut"]
    ref = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
    ref.fit(train[feats].to_numpy(float), train["damaged"].astype(bool).to_numpy())
    expected = ref.predict_proba(test[feats].to_numpy(float))[:, 1]
    got = scores[scores["event"] == "beirut"].sort_values("instance_id")["score"].to_numpy()
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-9)


def test_deterministic(events_cfg, before_freeze):
    a, _ = run_probes(synth(events_cfg), events_cfg, before_freeze)
    b, _ = run_probes(synth(events_cfg), events_cfg, before_freeze)
    pd.testing.assert_frame_equal(a, b)


def test_after_freeze_every_event_is_held_out(events_cfg, after_freeze):
    df = synth(events_cfg, today_locked=False)
    scores, fits = run_probes(df, events_cfg, after_freeze)
    assert sorted(scores["event"].unique()) == ["beirut", "haiti", "hawaii", "marshall"]
    haiti = next(f for f in fits if f.held_out == "haiti")
    assert haiti.train_events == ("beirut", "hawaii", "marshall")   # mexico has units: none


def test_missing_training_event_stops(events_cfg, before_freeze):
    df = synth(events_cfg, events=("beirut", "hawaii", "haiti"))
    with pytest.raises(ProbeError, match="marshall"):
        run_probes(df, events_cfg, before_freeze)


def test_one_class_training_set_stops(events_cfg, before_freeze):
    df = synth(events_cfg)
    df.loc[df["event"] != "haiti", "damaged"] = False
    with pytest.raises(ProbeError, match="one class"):
        run_probes(df, events_cfg, before_freeze)


def test_leaky_features_table_is_refused(events_cfg, before_freeze):
    df = synth(events_cfg)
    df.loc[df["event"] == "haiti", "damaged"] = True
    with pytest.raises(C.ContractError):
        run_probes(df, events_cfg, before_freeze)


def test_several_settings_and_comparison(events_cfg, after_freeze):
    good = synth(events_cfg, today_locked=False, signal=2.5, encoder="enc_a")
    weak = synth(events_cfg, today_locked=False, signal=0.7, encoder="enc_b")
    scores, fits = run_probes(pd.concat([good, weak], ignore_index=True), events_cfg, after_freeze)
    assert len(fits) == 8
    m = event_metrics(scores, events_cfg, n_boot=200)
    c = compare(m, ("enc_a", "outline"), ("enc_b", "outline"))
    assert len(c.gains) == 4 and (c.gains > 0).all()
    assert c.interval.lower > 0 and c.verdict is Verdict.SUPPORTED
    assert c.sign_flip.p_value == pytest.approx(2 / 16)
    same = compare(m, ("enc_a", "outline"), ("enc_a", "outline"))
    assert (same.gains == 0).all() and same.verdict is Verdict.NOT_WORTHWHILE


def test_compare_fixed_event_set_must_be_complete(events_cfg, after_freeze):
    df = synth(events_cfg, today_locked=False)
    scores, _ = run_probes(df, events_cfg, after_freeze)
    m = event_metrics(scores, events_cfg, n_boot=100)
    with pytest.raises(ValueError, match="missing"):
        compare(m, ("fake", "outline"), ("fake", "outline"), events=["beirut", "libya"])
    with pytest.raises(ValueError, match="at least 2"):
        compare(m[m["event"] == "beirut"], ("fake", "outline"), ("fake", "outline"))


def test_event_metrics_values(events_cfg, before_freeze):
    from sklearn.metrics import average_precision_score, roc_auc_score
    scores, _ = run_probes(synth(events_cfg), events_cfg, before_freeze)
    m = event_metrics(scores, events_cfg, n_boot=300, seed=3).set_index("event")
    b = scores[scores["event"] == "beirut"]
    y = b["damaged"].astype(int)
    assert m.loc["beirut", "auroc"] == pytest.approx(roc_auc_score(y, b["score"]))
    assert m.loc["beirut", "ap"] == pytest.approx(average_precision_score(y, b["score"]))
    assert m.loc["beirut", "boot_lower"] <= m.loc["beirut", "auroc"] <= m.loc["beirut", "boot_upper"]
    assert m.loc["beirut", "n"] == 120


def test_non_convergence_stops(events_cfg, before_freeze, monkeypatch):
    import damagesar_bench.probe.logistic as L
    monkeypatch.setattr(L, "MAX_ITER", 1)
    with pytest.raises(ProbeError, match="converge"):
        run_probes(synth(events_cfg), events_cfg, before_freeze)


def test_no_held_out_event_present(events_cfg, before_freeze):
    df = synth(events_cfg, events=("haiti", "marshall"))
    with pytest.raises(ProbeError, match="no fold"):
        run_probes(df, events_cfg, before_freeze)
