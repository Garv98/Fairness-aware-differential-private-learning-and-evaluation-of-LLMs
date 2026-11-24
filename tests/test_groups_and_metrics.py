import numpy as np
import pandas as pd

from fairdp import groups as G
from fairdp.data.preprocess import prepare
from fairdp.metrics.fairness import (differential_fairness_eps, fairness_report,
                                     group_metrics, dp_disparate_impact)
from fairdp.metrics.stats import bootstrap_ci, holm


def test_credit_binning_matches_notebook_rules():
    train = pd.DataFrame({"person_age": [22, 30, 50, 25, 35, 36],
                          "person_income": [10, 20, 30, 40, 50, 60],
                          "person_home_ownership": ["RENT"] * 6})
    binner = G.CREDIT.fit(train)
    out = binner(train)
    # (0,25] Young, (25,35] Middle, (35,inf) Older  -- right-closed like nb04
    assert out["age"].tolist() == ["Young", "Middle", "Older", "Young", "Middle", "Older"]
    q1, q3 = np.quantile(train["person_income"], [0.25, 0.75])
    assert binner.thresholds["income_q1"] == q1 and binner.thresholds["income_q3"] == q3


def test_prepare_split_sizes_and_intersections(synthetic_credit):
    d = prepare(synthetic_credit, seed=1)
    n = len(synthetic_credit)
    assert len(d.train.y) + len(d.val.y) + len(d.test.y) == n
    assert abs(len(d.test.y) / n - 0.2) < 0.01
    assert d.train.groups["intersection"].str.count(r"\|").eq(2).all()
    # deterministic given the seed
    d2 = prepare(synthetic_credit, seed=1)
    assert np.array_equal(d.test.y, d2.test.y)


def test_cell_index_marginals(synthetic_credit):
    d = prepare(synthetic_credit, seed=0)
    cells = G.CellIndex(d.train.groups, d.group_attrs)
    ids = cells.encode(d.train.groups)
    assert (ids >= 0).all()
    counts = np.bincount(ids, minlength=cells.n_cells)
    # marginal counts derived from cell counts == direct marginal counts
    M = cells.marginal_membership["home"]
    derived = dict(zip(cells.levels["home"], M @ counts))
    direct = d.train.groups["home"].value_counts().to_dict()
    assert all(derived[k] == direct[k] for k in direct)


def test_group_metrics_against_fairlearn():
    fl = __import__("pytest").importorskip("fairlearn.metrics")
    rng = np.random.default_rng(0)
    y, p = rng.integers(0, 2, 500), rng.integers(0, 2, 500)
    g = rng.choice(list("abc"), 500)
    ours = group_metrics(y, p, g).set_index("group")
    mf = fl.MetricFrame(metrics={"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate,
                                 "sel": fl.selection_rate},
                        y_true=y, y_pred=p, sensitive_features=g)
    np.testing.assert_allclose(ours["tpr"], mf.by_group["tpr"].to_numpy())
    np.testing.assert_allclose(ours["fpr"], mf.by_group["fpr"].to_numpy())
    np.testing.assert_allclose(ours["selection_rate"], mf.by_group["sel"].to_numpy())


def test_fairness_report_and_df_eps(synthetic_credit):
    d = prepare(synthetic_credit, seed=0)
    y = d.test.y
    summary, tables = fairness_report(y, y, d.test.groups, d.group_attrs, prob=y.astype(float))
    assert summary["accuracy"] == 1.0
    assert summary["intersection_worst_tpr"] == 1.0
    assert set(tables) == set(d.group_attrs) | {"intersection"}
    # identical selection rates -> eps_DF == 0
    assert differential_fairness_eps(np.ones(100), np.repeat(["a", "b"], 50)) == 0.0


def test_disparate_impact():
    a = pd.DataFrame({"group": ["x", "y"], "n": [100, 100], "n_pos": [50, 50], "tpr": [0.8, 0.8]})
    b = a.assign(tpr=[0.7, 0.4])
    di = dp_disparate_impact(a, b)
    assert np.isclose(di["di_tpr_max"], 0.4) and np.isclose(di["di_tpr_spread"], 0.3)


def test_stats_helpers():
    m, lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0])
    assert lo <= m <= hi and m == 2.5
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == 0.03 and adj["b"] >= adj["c"]
