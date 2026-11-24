"""Per-group membership-inference evaluation.

Loss-threshold attack (Yeom et al., 2018): score = -loss; members = training
rows, non-members = held-out test rows (balanced subsample). Reported per
group as AUC and TPR at 1% / 10% FPR, enabling the *disparate vulnerability*
analysis (Kulynych et al., 2022): does fairness mitigation shift leakage onto
minority cells? A shadow-model LiRA (Carlini et al., 2022) can be added later
using the same interface.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ..models import per_sample_loss

MIN_PER_SIDE = 30


def _tpr_at(y, s, fpr_target):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def loss_threshold_mia(model, train, test, attrs: list[str], seed: int = 0,
                       max_per_side: int = 5000, global_fpr: float = 0.10) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    m = min(len(train.y), len(test.y), max_per_side)
    i_in = rng.choice(len(train.y), m, replace=False)
    i_out = rng.choice(len(test.y), m, replace=False)
    s_in = -per_sample_loss(model, train.X[i_in], train.y[i_in])
    s_out = -per_sample_loss(model, test.X[i_out], test.y[i_out])
    score = np.concatenate([s_in, s_out])
    member = np.concatenate([np.ones(m), np.zeros(m)])
    g = pd.concat([train.groups.iloc[i_in], test.groups.iloc[i_out]], ignore_index=True)

    # One global decision threshold (attacker does not know group membership):
    # the threshold that flags `global_fpr` of all non-members. Disparate
    # vulnerability = per-group advantage (TPR - FPR) at that same threshold.
    thr = float(np.quantile(s_out, 1 - global_fpr))
    flag = score >= thr
    summary = {"mia_auc": float(roc_auc_score(member, score)),
               "mia_tpr_at_1fpr": _tpr_at(member, score, 0.01),
               "mia_global_fpr": global_fpr,
               "mia_advantage": float(flag[member == 1].mean() - flag[member == 0].mean())}
    rows = []
    for a in list(attrs) + ["intersection"]:
        for grp, idx in g.groupby(a).indices.items():
            mm = member[idx]
            if mm.sum() < MIN_PER_SIDE or (1 - mm).sum() < MIN_PER_SIDE:
                continue
            fi = flag[idx]
            tpr, fpr = float(fi[mm == 1].mean()), float(fi[mm == 0].mean())
            rows.append({"attribute": a, "group": grp, "n": len(idx), "n_members": int(mm.sum()),
                         "mia_auc": float(roc_auc_score(mm, score[idx])),
                         "tpr_global": tpr, "fpr_global": fpr, "advantage": tpr - fpr})
    tab = pd.DataFrame(rows)
    if len(tab):
        it = tab[tab.attribute == "intersection"]
        if len(it):
            summary["mia_advantage_worst_cell"] = float(it["advantage"].max())
            summary["mia_advantage_cell_gap"] = float(it["advantage"].max() - it["advantage"].min())
    return summary, tab
