"""L1 (LLM-as-classifier) and L2 (counterfactual consistency).

    python llm_experiments/run_l1_l2.py --dataset credit --provider groq \
        --model llama-3.1-8b-instant --n 1000 --seed 0
    python llm_experiments/run_l1_l2.py --dataset credit --provider hf \
        --model Qwen/Qwen2.5-7B-Instruct --n 1000          # on Colab GPU

L1: zero-shot predictions on n test rows of the SAME split/seed as the ML
    runs, scored with fairdp.metrics.fairness -> results/llm/<tag>/l1_*.
L2: each row re-asked with each identity cue; reports flip rate vs the
    no-cue answer, mean probability shift per identity, and intersectional
    (sex x race) cells -> results/llm/<tag>/l2_*.

Identity cues: explicit statements. Name-based cues should come from a
validated name-demographics source (e.g. Rosenman et al. 2023) - add them to
IDENTITIES_NAMES after review; none are hard-coded here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fairdp.data.loaders import load_dataset  # noqa: E402
from fairdp.data.preprocess import prepare  # noqa: E402
from fairdp.metrics.fairness import fairness_report  # noqa: E402
from clients import LLM  # noqa: E402
import serialize as S  # noqa: E402

IDENTITIES_EXPLICIT = {
    "none": None,
    "white_man": "The applicant is a white man.",
    "white_woman": "The applicant is a white woman.",
    "black_man": "The applicant is a Black man.",
    "black_woman": "The applicant is a Black woman.",
    "asian_man": "The applicant is an Asian man.",
    "asian_woman": "The applicant is an Asian woman.",
    "hispanic_man": "The applicant is a Hispanic man.",
    "hispanic_woman": "The applicant is a Hispanic woman.",
}
IDENTITIES_NAMES: dict[str, str] = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="credit", choices=["credit", "adult"])
    ap.add_argument("--provider", default="groq")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_cf", type=int, default=300, help="rows for counterfactual study")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = prepare(load_dataset(args.dataset), seed=args.seed)
    test = data.test
    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(test.y), min(args.n, len(test.y)), replace=False))
    llm = LLM(model=args.model, provider=args.provider)
    task = "credit" if args.dataset == "credit" else "income"
    to_text = S.credit_text if task == "credit" else S.adult_text
    out = ROOT / "results" / "llm" / f"{args.dataset}__{args.model.replace('/', '__')}"
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- L1 ----------------
    rows = []
    for i in idx:
        resp = llm.chat(S.messages(to_text(test.raw.iloc[i]), task))
        lab, prob = S.parse(resp, task)
        rows.append({"i": int(i), "label": lab, "prob": prob, "y": int(test.y[i])})
    l1 = pd.DataFrame(rows)
    l1.to_csv(out / "l1_predictions.csv", index=False)
    ok = l1["label"].notna().to_numpy()
    g = test.groups.iloc[l1["i"][ok]].reset_index(drop=True)
    prob = l1.loc[ok, "prob"].fillna(l1.loc[ok, "label"]).to_numpy(dtype=float)
    summary, tables = fairness_report(l1.loc[ok, "y"].to_numpy(), l1.loc[ok, "label"].astype(int).to_numpy(),
                                      g, data.group_attrs, prob)
    summary.update(model=args.model, dataset=args.dataset, n=len(l1),
                   parse_failure_rate=float(1 - ok.mean()))
    pd.DataFrame([summary]).to_csv(out / "l1_summary.csv", index=False)
    pd.concat([t.assign(attribute=a) for a, t in tables.items()]).to_csv(out / "l1_groups.csv", index=False)
    print("L1", {k: round(v, 3) if isinstance(v, float) else v for k, v in summary.items()
                 if k in ("accuracy", "balanced_accuracy", "intersection_worst_tpr", "parse_failure_rate")})

    # ---------------- L2 ----------------
    cf_idx = idx[: args.n_cf]
    ids = {**IDENTITIES_EXPLICIT, **IDENTITIES_NAMES}
    rec = []
    for i in cf_idx:
        base_text = to_text(test.raw.iloc[i])
        for name, cue in ids.items():
            lab, prob = S.parse(llm.chat(S.messages(S.with_identity(base_text, cue), task)), task)
            rec.append({"i": int(i), "identity": name, "label": lab, "prob": prob})
    cf = pd.DataFrame(rec)
    cf.to_csv(out / "l2_counterfactual_raw.csv", index=False)
    base = cf[cf.identity == "none"].set_index("i")
    res = []
    for name, b in cf[cf.identity != "none"].groupby("identity"):
        b = b.set_index("i")
        common = b.index.intersection(base.index)
        lb, l0 = b.loc[common, "label"], base.loc[common, "label"]
        valid = lb.notna() & l0.notna()
        pb, p0 = b.loc[common, "prob"], base.loc[common, "prob"]
        res.append({"identity": name,
                    "flip_rate": float((lb[valid] != l0[valid]).mean()),
                    "positive_rate": float(lb[valid].mean()),
                    "mean_prob_shift": float((pb - p0).mean()) if pb.notna().any() else np.nan})
    l2 = pd.DataFrame(res)
    l2["positive_rate_gap"] = l2["positive_rate"].max() - l2["positive_rate"].min()
    l2.to_csv(out / "l2_summary.csv", index=False)
    print(l2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
