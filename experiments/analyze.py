"""Aggregate a run directory into paper tables + figures.

    python experiments/analyze.py --run main [--reference dpsgd] [--pp both]

Writes to results/runs/<run>/analysis/:
    table_main.csv          mean [95% bootstrap CI] per dataset x eps x method x pp
    tests_<metric>.csv      Wilcoxon (paired by seed) vs reference, Holm-corrected
    fig_pareto_<ds>.png     balanced accuracy vs worst intersectional TPR
    fig_eps_<ds>.png        worst intersectional TPR vs epsilon
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fairdp.metrics.stats import aggregate, paired_tests  # noqa: E402

KEY_METRICS = [
    "accuracy", "balanced_accuracy", "roc_auc", "ece", "recall",
    "intersection_worst_tpr", "intersection_tpr_gap", "intersection_eodds_diff",
    "intersection_worst_accuracy", "intersection_df_eps", "intersection_di_tpr_max",
    "intersection_di_tpr_spread", "mia_auc", "mia_auc_worst_cell",
]


def fmt(row, m):
    if pd.isna(row.get(m)):
        return ""
    lo, hi = row.get(f"{m}_lo"), row.get(f"{m}_hi")
    if pd.isna(lo):
        return f"{row[m]:.3f}"
    return f"{row[m]:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", default="dpsgd")
    args = ap.parse_args()
    run = ROOT / "results" / "runs" / args.run
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    s = pd.read_parquet(run / "summary.parquet")
    # attribute-prefixed DI columns for the marginal attrs exist too; keep all numeric
    metrics = [m for m in KEY_METRICS if m in s.columns]
    for a in ("age", "sex", "race", "income", "home"):
        for suffix in ("_worst_tpr", "_tpr_gap"):
            if a + suffix in s.columns:
                metrics.append(a + suffix)

    agg = aggregate(s, metrics, by=("dataset", "pp", "epsilon", "method"))
    agg.to_csv(out / "table_main_raw.csv", index=False)
    pretty = agg[["dataset", "pp", "epsilon", "method", "n_seeds"]].copy()
    for m in metrics:
        pretty[m] = agg.apply(lambda r: fmt(r, m), axis=1)
    pretty.to_csv(out / "table_main.csv", index=False)

    priv = s[s["private"]]
    for m in ("intersection_worst_tpr", "intersection_tpr_gap", "balanced_accuracy",
              "accuracy", "intersection_di_tpr_max"):
        if m in priv:
            t = paired_tests(priv, m, args.reference, group_cols=("dataset", "pp", "epsilon"))
            t.to_csv(out / f"tests_{m}.csv", index=False)

    for ds, block in agg.groupby("dataset"):
        b = block[~block["pp"]]
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for meth, mb in b.groupby("method"):
            ax.errorbar(mb["balanced_accuracy"], mb["intersection_worst_tpr"],
                        xerr=[mb["balanced_accuracy"] - mb["balanced_accuracy_lo"],
                              mb["balanced_accuracy_hi"] - mb["balanced_accuracy"]],
                        yerr=[mb["intersection_worst_tpr"] - mb["intersection_worst_tpr_lo"],
                              mb["intersection_worst_tpr_hi"] - mb["intersection_worst_tpr"]],
                        fmt="o", capsize=2, label=meth, alpha=0.85)
        ax.set_xlabel("balanced accuracy")
        ax.set_ylabel("worst intersectional TPR")
        ax.set_title(f"{ds}: utility vs worst-cell recall (all eps, no PP)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"fig_pareto_{ds}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        bp = b[b["epsilon"] != float("inf")]
        for meth, mb in bp.groupby("method"):
            mb = mb.sort_values("epsilon")
            ax.plot(mb["epsilon"], mb["intersection_worst_tpr"], marker="o", label=meth)
            ax.fill_between(mb["epsilon"], mb["intersection_worst_tpr_lo"],
                            mb["intersection_worst_tpr_hi"], alpha=0.12)
        for meth, mb in b[b["epsilon"] == float("inf")].groupby("method"):
            ax.axhline(mb["intersection_worst_tpr"].iloc[0], ls="--", lw=1, label=f"{meth} (non-DP)")
        ax.set_xscale("log")
        ax.set_xlabel("epsilon (fixed budget)")
        ax.set_ylabel("worst intersectional TPR")
        ax.set_title(ds)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"fig_eps_{ds}.png", dpi=150)
        plt.close(fig)
    print(f"analysis written to {out}")


if __name__ == "__main__":
    main()
