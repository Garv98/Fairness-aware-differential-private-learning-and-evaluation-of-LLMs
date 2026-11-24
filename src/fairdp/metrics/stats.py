"""Statistics across seeds: bootstrap CIs and paired significance tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return (np.nan, np.nan, np.nan)
    if len(v) == 1:
        return (float(v[0]), np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return (float(v.mean()), float(lo), float(hi))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    keys = sorted(pvals, key=pvals.get)
    m, adj, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def paired_tests(df: pd.DataFrame, metric: str, reference: str,
                 group_cols=("dataset", "epsilon"),
                 method_col: str = "method", seed_col: str = "seed") -> pd.DataFrame:
    """Wilcoxon signed-rank of each method vs ``reference`` paired by seed,
    Holm-corrected within each (dataset, epsilon) block."""
    rows = []
    for key, block in df.groupby(list(group_cols), dropna=False):
        ref = block[block[method_col] == reference].set_index(seed_col)[metric]
        pv = {}
        diffs = {}
        for m, mb in block.groupby(method_col):
            if m == reference:
                continue
            x = mb.set_index(seed_col)[metric]
            common = ref.index.intersection(x.index)
            if len(common) < 5:
                continue
            d = (x.loc[common] - ref.loc[common]).to_numpy()
            diffs[m] = float(np.mean(d))
            pv[m] = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(d).pvalue)
        for m, p in holm(pv).items():
            rows.append(dict(zip(group_cols, key if isinstance(key, tuple) else (key,)),
                             method=m, metric=metric, mean_diff=diffs[m],
                             p_raw=pv[m], p_holm=p))
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame, metrics: list[str],
              by=("dataset", "epsilon", "method")) -> pd.DataFrame:
    rows = []
    for key, block in df.groupby(list(by), dropna=False):
        row = dict(zip(by, key))
        row["n_seeds"] = block["seed"].nunique() if "seed" in block else len(block)
        for m in metrics:
            if m not in block:
                continue
            mean, lo, hi = bootstrap_ci(block[m].to_numpy())
            row[m] = mean
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
        rows.append(row)
    return pd.DataFrame(rows)
