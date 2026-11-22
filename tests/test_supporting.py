import numpy as np
import pandas as pd
import pytest

from damagesar_bench.stats.event_level import sign_flip_test
from damagesar_bench.stats.supporting import one_sided_sign_flip, per_event_paired, summarise


def test_one_sided_sign_flip():
    assert one_sided_sign_flip([0.1, 0.2, 0.05, 0.3, 0.1]) == pytest.approx(1 / 32)   # all positive: 1 pattern of 32
    assert one_sided_sign_flip([-0.1, -0.2, -0.05, -0.3, -0.1]) == pytest.approx(1.0)
    g = [0.3, -0.1, 0.2, 0.05, 0.1]
    assert one_sided_sign_flip(g) <= sign_flip_test(g).p_value          # one-sided never above two-sided here


def _scores(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for ev, strength in [("e1", 2.0), ("e2", 2.0), ("e3", 0.0)]:
        for t in range(30):
            for i in range(20):
                y = rng.random() < 0.3
                base = rng.normal()
                rows.append({"event": ev, "tile_id": f"{ev}_{t}", "instance_id": i, "damaged": y,
                             "enc": base + (strength if y else 0.0), "ref": base + rng.normal() * 3})
    d = pd.DataFrame(rows)
    a = d.assign(encoder="A", pooling="outline", score=d["enc"])
    b = d.assign(encoder="B", pooling="outline", score=d["ref"])
    out = pd.concat([a, b], ignore_index=True)[["event", "tile_id", "instance_id", "damaged", "score", "encoder",
                                                 "pooling"]]
    out["damaged"] = out["damaged"].astype("boolean")
    return out


def test_per_event_paired_counts_clear_wins_only():
    pe = per_event_paired(_scores(), ("A", "outline"), ("B", "outline"), ["e1", "e2", "e3"], n_boot=300)
    r = pe.set_index("event")
    assert r.loc["e1", "counts_for"] == "a" and r.loc["e2", "counts_for"] == "a"
    assert r.loc["e1", "boot_lower"] > 0 and r.loc["e1", "delong_p"] < 0.05
    s = summarise(pe)
    assert s.n_events == 3 and s.n_a_better >= 2 and s.n_b_better == 0


def test_per_event_paired_needs_both_settings():
    with pytest.raises(ValueError, match="no scores"):
        per_event_paired(_scores(), ("A", "outline"), ("C", "outline"), ["e1"])
