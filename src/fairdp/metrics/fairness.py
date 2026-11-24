"""Utility, group and intersectional fairness metrics.

Extends the original ``src/fairness.py::evaluate_group_metrics`` with
ROC-AUC/ECE, worst-group summaries, equalized-odds / demographic-parity
differences, intersectional differential fairness (Foulds et al., 2020) and
DP-induced disparate impact.

Convention: positive class = 1 (e.g. *default* for credit). "Recall" = TPR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

MIN_SUPPORT = 30            # min samples for a group to enter summary gaps
MIN_POSITIVES = 10          # min positives for TPR/FNR-based gaps


def expected_calibration_error(y_true, prob, n_bins: int = 15) -> float:
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - y_true[m].mean())
    return float(ece)


def utility_metrics(y_true, y_pred, prob=None) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "selection_rate": float(y_pred.mean()),
    }
    if prob is not None and len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, prob)
        out["ece"] = expected_calibration_error(y_true, prob)
    return {k: float(v) for k, v in out.items()}


def group_metrics(y_true, y_pred, groups, prob=None) -> pd.DataFrame:
    """Per-group table (superset of the original evaluate_group_metrics)."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_pred),
                      "g": pd.Series(groups).astype(str).to_numpy()})
    if prob is not None:
        d["s"] = np.asarray(prob)
    rows = []
    for g, gd in d.groupby("g", sort=True):
        y, p = gd["y"].to_numpy(), gd["p"].to_numpy()
        tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
        tn = int(((y == 0) & (p == 0)).sum()); fn = int(((y == 1) & (p == 0)).sum())
        pos, neg = tp + fn, fp + tn
        row = {
            "group": g, "n": len(gd), "n_pos": pos,
            "accuracy": (tp + tn) / len(gd),
            "precision": tp / (tp + fp) if tp + fp else np.nan,
            "tpr": tp / pos if pos else np.nan,
            "fpr": fp / neg if neg else np.nan,
            "fnr": fn / pos if pos else np.nan,
            "f1": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "selection_rate": p.mean(),
            "base_rate": y.mean(),
        }
        if prob is not None and 0 < pos < len(gd):
            row["roc_auc"] = roc_auc_score(y, gd["s"].to_numpy())
        rows.append(row)
    return pd.DataFrame(rows)


def _eligible(tab: pd.DataFrame, need_pos: bool) -> pd.DataFrame:
    m = tab["n"] >= MIN_SUPPORT
    if need_pos:
        m &= tab["n_pos"] >= MIN_POSITIVES
    return tab[m]


def summarize_groups(tab: pd.DataFrame, prefix: str) -> dict:
    """Worst-group values and max-min gaps over sufficiently large groups."""
    out = {}
    for metric, need_pos in [("accuracy", False), ("tpr", True), ("fpr", False),
                             ("selection_rate", False), ("f1", True)]:
        t = _eligible(tab, need_pos)[metric].dropna()
        if len(t) == 0:
            continue
        out[f"{prefix}_{metric}_gap"] = float(t.max() - t.min())
        if metric in ("accuracy", "tpr", "f1"):
            out[f"{prefix}_worst_{metric}"] = float(t.min())
    t = _eligible(tab, True)
    if len(t):
        # equalized odds difference = max(TPR gap, FPR gap)
        out[f"{prefix}_eodds_diff"] = float(max(t["tpr"].max() - t["tpr"].min(),
                                               t["fpr"].max() - t["fpr"].min()))
    out[f"{prefix}_n_groups_eval"] = int(len(_eligible(tab, False)))
    out[f"{prefix}_n_groups_total"] = int(len(tab))
    return out


def differential_fairness_eps(y_pred, groups, alpha: float = 1.0) -> float:
    """Empirical differential-fairness epsilon (Foulds et al., 2020).

    eps_DF = max_{g,g'} max_{y in {0,1}} |log P(yhat=y|g) - log P(yhat=y|g')|,
    with Dirichlet(alpha) smoothing so tiny intersectional cells stay finite.
    """
    d = pd.DataFrame({"p": np.asarray(y_pred), "g": pd.Series(groups).astype(str).to_numpy()})
    stats = d.groupby("g")["p"].agg(["sum", "count"])
    p1 = (stats["sum"] + alpha) / (stats["count"] + 2 * alpha)
    l1, l0 = np.log(p1.to_numpy()), np.log(1 - p1.to_numpy())
    return float(max(l1.max() - l1.min(), l0.max() - l0.min()))


def fairness_report(y_true, y_pred, groups: pd.DataFrame, attrs: list[str],
                    prob=None) -> tuple[dict, dict[str, pd.DataFrame]]:
    """Return (flat summary dict, per-attribute group tables).

    Group tables include every marginal attribute and ``intersection``.
    """
    summary = utility_metrics(y_true, y_pred, prob)
    tables = {}
    for a in list(attrs) + ["intersection"]:
        tab = group_metrics(y_true, y_pred, groups[a], prob)
        tables[a] = tab
        summary.update(summarize_groups(tab, a))
    summary["intersection_df_eps"] = differential_fairness_eps(y_pred, groups["intersection"])
    return summary, tables


def dp_disparate_impact(tab_nonprivate: pd.DataFrame, tab_private: pd.DataFrame,
                        metric: str = "tpr") -> dict:
    """Per-group cost of privacy: delta_g = m_nonDP(g) - m_DP(g).

    Returns max delta (worst-hit group), spread (max-min delta) and std.
    Only groups meeting the support thresholds in both tables are used.
    """
    need_pos = metric in ("tpr", "fnr", "f1")
    a = _eligible(tab_nonprivate, need_pos).set_index("group")[metric]
    b = _eligible(tab_private, need_pos).set_index("group")[metric]
    delta = (a - b).dropna()
    if delta.empty:
        return {}
    return {f"di_{metric}_max": float(delta.max()),
            f"di_{metric}_spread": float(delta.max() - delta.min()),
            f"di_{metric}_std": float(delta.std(ddof=0)),
            f"di_{metric}_mean": float(delta.mean())}
