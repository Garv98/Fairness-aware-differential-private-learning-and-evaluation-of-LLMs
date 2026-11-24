"""Extended fairness / responsible-AI metrics.

Group fairness (per sensitive attribute, groups below the support threshold
are excluded from gaps and ratios):
  * demographic parity difference and disparate-impact ratio (four-fifths
    rule, EEOC Uniform Guidelines 1978)
  * equal opportunity difference (TPR gap), equalized-odds difference
  * predictive parity difference (precision gap)
  * calibration: calibration-in-the-large per group (mean score - actual rate);
    the gap is the spread of that error across groups. (Per-group binned ECE
    is reported too but not used for the gap: it is biased upward for small
    groups, so its spread would mostly measure group size.)

Individual fairness:
  * consistency (Zemel et al., 2013): 1 - mean |yhat_i - mean_{kNN(i)} yhat_j|

Proxy audit:
  * how well the non-sensitive features predict each sensitive attribute
    (cross-validated AUC) and which features carry the most information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from .fairness import MIN_POSITIVES, MIN_SUPPORT, expected_calibration_error


def _groups(y, yhat, prob, g):
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(yhat), "s": np.asarray(prob), "g": pd.Series(g).astype(str).to_numpy()})
    return d.groupby("g", sort=True)


def group_fairness(y, yhat, prob, g, four_fifths: float = 0.8) -> dict:
    """Fairness metrics for one sensitive attribute."""
    rows = []
    for name, d in _groups(y, yhat, prob, g):
        n, pos = len(d), int((d.y == 1).sum())
        tp = int(((d.y == 1) & (d.p == 1)).sum()); fp = int(((d.y == 0) & (d.p == 1)).sum())
        rows.append({
            "group": name, "n": n, "n_pos": pos,
            "selection_rate": float(d.p.mean()),
            "tpr": tp / pos if pos else None,
            "fpr": fp / (n - pos) if n - pos else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "ece": expected_calibration_error(d.y.to_numpy(), d.s.to_numpy(), n_bins=10) if n >= MIN_SUPPORT else None,
            "mean_score": float(d.s.mean()), "base_rate": float(d.y.mean()),
            "calibration_error": float(d.s.mean() - d.y.mean()),
            "eligible": bool(n >= MIN_SUPPORT and pos >= MIN_POSITIVES),
        })
    t = pd.DataFrame(rows)
    e = t[t.eligible]

    def gap(col):
        v = e[col].dropna()
        return float(v.max() - v.min()) if len(v) > 1 else None

    sr = e["selection_rate"]
    di_ratio = float(sr.min() / sr.max()) if len(sr) > 1 and sr.max() > 0 else None
    eo = [x for x in (gap("tpr"), gap("fpr")) if x is not None]
    return {
        "groups": t.to_dict("records"),
        "demographic_parity_diff": gap("selection_rate"),
        "disparate_impact_ratio": di_ratio,
        "four_fifths_pass": None if di_ratio is None else bool(di_ratio >= four_fifths),
        "equal_opportunity_diff": gap("tpr"),
        "equalized_odds_diff": max(eo) if eo else None,
        "predictive_parity_diff": gap("precision"),
        "calibration_gap": gap("calibration_error"),
        "worst_group_tpr": float(e["tpr"].dropna().min()) if e["tpr"].notna().any() else None,
        "n_groups_eligible": int(len(e)), "n_groups": int(len(t)),
    }


def consistency(X: np.ndarray, yhat: np.ndarray, k: int = 10, max_n: int = 5000, seed: int = 0) -> float:
    """Individual fairness: similar people (k nearest neighbours in feature
    space) should receive similar decisions. 1.0 = perfectly consistent."""
    X, yhat = np.asarray(X), np.asarray(yhat, dtype=float)
    if len(X) > max_n:
        idx = np.random.default_rng(seed).choice(len(X), max_n, replace=False)
        X, yhat = X[idx], yhat[idx]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, ind = nn.kneighbors(X)
    return float(1.0 - np.mean(np.abs(yhat - yhat[ind[:, 1:]].mean(axis=1))))


def proxy_audit(X_nonsensitive: pd.DataFrame, sensitive: pd.Series, positive, top_k: int = 8, seed: int = 0) -> dict:
    """Can the sensitive attribute be recovered from the other features?

    Returns cross-validated AUC of a logistic regression predicting
    ``sensitive == positive`` from one-hot encoded non-sensitive features, and
    the features with the highest mutual information with the attribute.
    AUC near 0.5 = no proxy; near 1.0 = removing the attribute does not hide it.
    """
    Xd = pd.get_dummies(X_nonsensitive, dummy_na=False).astype(float)
    Xd = Xd.fillna(Xd.median(numeric_only=True))
    Xs = (Xd - Xd.mean()) / Xd.std().replace(0, 1)
    target = (sensitive.astype(str) == str(positive)).astype(int).to_numpy()
    prob = cross_val_predict(LogisticRegression(max_iter=2000), Xs.to_numpy(), target, cv=5, method="predict_proba")[:, 1]
    auc = float(roc_auc_score(target, prob))
    mi = mutual_info_classif(Xs.to_numpy(), target, random_state=seed)
    # aggregate one-hot columns back to their source field
    agg: dict[str, float] = {}
    for col, v in zip(Xd.columns, mi):
        field = next((c for c in X_nonsensitive.columns if col == c or col.startswith(f"{c}_")), col)
        agg[field] = agg.get(field, 0.0) + float(v)
    top = sorted(agg.items(), key=lambda kv: -kv[1])[:top_k]
    return {"auc": auc, "positive": str(positive), "base_rate": float(target.mean()),
            "top_features": [{"field": f, "mutual_info": round(v, 5)} for f, v in top]}
