import numpy as np
import pandas as pd

from fairdp.metrics.ethics import consistency, group_fairness, proxy_audit


def test_group_fairness_four_fifths_and_gaps():
    # group a: selection 0.5, group b: selection 0.3 -> ratio 0.6 (fails 4/5 rule)
    n = 200
    g = np.array(["a"] * n + ["b"] * n)
    y = np.tile([1, 0], n)
    yhat = np.concatenate([np.r_[np.ones(100), np.zeros(100)], np.r_[np.ones(60), np.zeros(140)]])
    prob = yhat * 0.8 + 0.1
    r = group_fairness(y, yhat, prob, g)
    assert abs(r["demographic_parity_diff"] - 0.2) < 1e-9
    assert abs(r["disparate_impact_ratio"] - 0.6) < 1e-9
    assert r["four_fifths_pass"] is False
    assert r["n_groups_eligible"] == 2


def test_group_fairness_identical_groups_is_fair():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    g = np.array(["a", "b"] * 200)
    r = group_fairness(y, y, y.astype(float), g)
    assert r["equal_opportunity_diff"] == 0.0 and r["four_fifths_pass"] is True


def test_consistency_bounds():
    X = np.random.default_rng(0).normal(size=(300, 3))
    assert consistency(X, np.ones(300)) == 1.0
    c = consistency(X, (X[:, 0] > 0).astype(int))
    assert 0.5 < c < 1.0


def test_proxy_audit_detects_proxy():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.choice(["F", "M"], 600))
    X = pd.DataFrame({"proxy": (s == "F").astype(float) + rng.normal(0, 0.1, 600), "noise": rng.normal(size=600),
                      "cat": rng.choice(["x", "y"], 600)})
    r = proxy_audit(X, s, "F")
    assert r["auc"] > 0.95
    assert r["top_features"][0]["field"] == "proxy"


def test_calibration_gap_is_spread_of_calibration_in_the_large():
    # group a perfectly calibrated (score 0.3, rate 0.3); group b over-scored by 0.2
    n = 100
    g = np.array(["a"] * n + ["b"] * n)
    y = np.r_[np.r_[np.ones(30), np.zeros(70)], np.r_[np.ones(30), np.zeros(70)]]
    prob = np.r_[np.full(n, 0.3), np.full(n, 0.5)]
    r = group_fairness(y, (prob >= 0.4).astype(int), prob, g)
    assert abs(r["calibration_gap"] - 0.2) < 1e-9
