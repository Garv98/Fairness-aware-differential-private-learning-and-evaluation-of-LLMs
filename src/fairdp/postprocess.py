"""Group-specific decision thresholds (equal-opportunity post-processing,
Hardt et al., 2016) fitted on the held-out validation split.

Thresholds are a function of the released model + validation data only, so
the training-data DP guarantee is preserved (post-processing). Validation
data itself is not protected by the guarantee - disclosed in the paper.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics.fairness import MIN_POSITIVES, MIN_SUPPORT


def fit_group_thresholds(prob_val: np.ndarray, y_val: np.ndarray, groups_val: pd.Series,
                         base_threshold: float = 0.5, target: str = "overall_tpr") -> dict:
    """Per group threshold whose validation TPR is closest to the overall TPR
    at ``base_threshold``. Groups with too few samples/positives keep the base
    threshold. Returns {group: threshold, "__default__": base}."""
    prob_val, y_val = np.asarray(prob_val), np.asarray(y_val)
    g = pd.Series(groups_val).astype(str).to_numpy()
    pos = y_val == 1
    target_tpr = float((prob_val[pos] >= base_threshold).mean()) if pos.any() else 0.5
    grid = np.linspace(0.02, 0.98, 97)
    th = {"__default__": base_threshold}
    for grp in np.unique(g):
        m = g == grp
        if m.sum() < MIN_SUPPORT or (m & pos).sum() < MIN_POSITIVES:
            continue
        tprs = np.array([(prob_val[m & pos] >= t).mean() for t in grid])
        # tie-break toward base threshold
        cost = np.abs(tprs - target_tpr) + 1e-4 * np.abs(grid - base_threshold)
        th[grp] = float(grid[np.argmin(cost)])
    return th


def apply_thresholds(prob: np.ndarray, groups: pd.Series, th: dict) -> np.ndarray:
    g = pd.Series(groups).astype(str)
    t = g.map(lambda x: th.get(x, th["__default__"])).to_numpy(dtype=float)
    return (np.asarray(prob) >= t).astype(int)
