"""Train, evaluate and document every model the demo serves.

    python server/train_models.py            # ~4 min on CPU
    python server/train_models.py --force    # retrain everything

Writes server/artifacts/:
  models/<key>.pt        PyTorch state dicts
  probs/<key>.npz        validation + test probabilities (threshold mitigation)
  evaluation.json        utility, group/intersectional metrics, ethics metrics, MIA, cost of privacy
  training.json          configs, noise plans, per-epoch history, timestamps
  proxy.json             proxy audit + "fairness through unawareness" baseline
  study.json             multi-seed results of the configured experiment run
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

import core
from fairdp.data.loaders import load_dataset
from fairdp.data.preprocess import prepare
from fairdp.metrics.ethics import proxy_audit
from fairdp.metrics.fairness import dp_disparate_impact, fairness_report
from fairdp.models import MLP, predict_proba
from fairdp.train import NonPrivateConfig, train_nonprivate


def reevaluate():
    """Recompute evaluation + probabilities for saved models (no retraining)."""
    evaluation, n = core.read_json("evaluation.json") or {}, 0
    for m in core.METHODS:
        for e in core.eps_list(m["id"]):
            k = core.key(m["id"], e)
            if not core.model_path(m["id"], e).exists():
                continue
            ev = core.full_eval(core.load_model(m["id"], e))
            probs = ev.pop("_probs")
            np.savez_compressed(core.PROBS_DIR / f"{k}.npz", val=probs["val"], test=probs["test"])
            evaluation[k] = {"method": m["id"], "name": m["name"], "private": e is not None, "eps": e, **ev}
            n += 1
    core.save_json("evaluation.json", evaluation)
    print(f"re-evaluated {n} models")
    return evaluation


def train_served(force: bool):
    core.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    core.PROBS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = {} if force else (core.read_json("evaluation.json") or {})
    training = {} if force else (core.read_json("training.json") or {})
    for m in core.METHODS:
        for e in core.eps_list(m["id"]):
            k = core.key(m["id"], e)
            if not force and core.model_path(m["id"], e).exists() and k in evaluation and "ethics" in evaluation[k]:
                print(f"skip {k}")
                continue
            t0, hist = time.time(), []
            model, meta = core.train(m["id"], e, callback=lambda ep, mdl, rec: hist.append({**rec, **core.quick_eval(mdl)}))
            torch.save(model.state_dict(), core.model_path(m["id"], e))
            ev = core.full_eval(model)
            probs = ev.pop("_probs")
            np.savez_compressed(core.PROBS_DIR / f"{k}.npz", val=probs["val"], test=probs["test"])
            evaluation[k] = {"method": m["id"], "name": m["name"], "private": e is not None, "eps": e, **ev}
            training[k] = {**meta, "epochs": hist, "seconds": round(time.time() - t0, 1),
                           "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            s = ev["summary"]
            print(f"{k:24s} acc={s['accuracy']:.3f} bal={s['balanced_accuracy']:.3f} worst={s.get('intersection_worst_tpr') or float('nan'):.3f} "
                  f"4/5(sex)={ev['ethics']['sex']['disparate_impact_ratio']:.2f} cons={ev['ethics']['consistency']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            core.save_json("evaluation.json", evaluation)
            core.save_json("training.json", training)
    return add_cost_of_privacy(evaluation)


def add_cost_of_privacy(evaluation: dict) -> dict:
    ref = evaluation.get(core.key("erm", None))
    if ref:
        for v in evaluation.values():
            v["cost_of_privacy"] = {a: core.clean(dp_disparate_impact(pd.DataFrame(ref["groups"][a]), pd.DataFrame(v["groups"][a]), "tpr")) for a in v["groups"]}
        core.save_json("evaluation.json", evaluation)
    return evaluation


def proxy_and_unawareness(evaluation: dict):
    _, d, *_ = core.data()
    raw = d.train.raw
    nonsens = raw.drop(columns=[c for c in core.SENSITIVE_FIELDS if c in raw.columns] + ["fnlwgt"], errors="ignore")
    sample = np.random.default_rng(core.CFG["seed"]).choice(len(raw), min(len(raw), 12000), replace=False)
    audits = {a: proxy_audit(nonsens.iloc[sample].reset_index(drop=True), raw[a].iloc[sample].reset_index(drop=True), pos, seed=core.CFG["seed"])
              for a, pos in core.CFG["proxy_audit"].items()}
    # "fairness through unawareness": same model, sex and race removed from the inputs
    ds_blind = load_dataset(core.CFG["dataset"], drop_sensitive_features=True)
    db = prepare(ds_blind, seed=core.CFG["seed"], **core.CFG["split"])
    gy = db.train.groups.assign(_y=db.train.y.astype(str))
    from fairdp.groups import CellIndex
    cy = CellIndex(gy, db.group_attrs + ["_y"])
    _, cfg = core.method_cfg("erm", None)
    model = train_nonprivate(MLP(db.train.X.shape[1], tuple(core.CFG["model"]["hidden"])), db.train.X, db.train.y, cy.encode(gy), cy.n_cells, NonPrivateConfig(**cfg))
    p = predict_proba(model, db.test.X)
    yhat = (p >= core.THRESH).astype(int)
    s, _ = fairness_report(db.test.y, yhat, db.test.groups, db.group_attrs, p)
    blind = {"summary": core.clean(s), "ethics": core.clean(core.ethics_block(db.test.y, yhat, p, db.test.groups, db.group_attrs))}
    aware = evaluation.get(core.key("erm", None))
    core.save_json("proxy.json", {"audits": audits, "blind": blind,
                                  "aware": {"summary": aware["summary"], "ethics": {a: {k: v for k, v in aware["ethics"][a].items() if k != "groups"} for a in core.SENSITIVE_FIELDS + ["intersection"]}} if aware else None,
                                  "removed": core.SENSITIVE_FIELDS})
    print("proxy audit:", {a: round(v["auc"], 3) for a, v in audits.items()}, "| blind model worst-group recall:", blind["summary"].get("intersection_worst_tpr"))


def study():
    run = core.ROOT / "results" / "runs" / core.CFG["study_run"]
    if not (run / "summary.parquet").exists():
        print("no study run found:", run)
        return
    S = pd.read_parquet(run / "summary.parquet")
    S = S[~S.pp]
    # drop runs invalidated by the fixed single-attribute ablation bug (see docs)
    S = S[~((S.method == "ig_single_attr") & (S.epsilon > 1) & (S.epsilon < math.inf))]
    cols = ["accuracy", "balanced_accuracy", "recall", "intersection_worst_tpr", "intersection_tpr_gap", "intersection_di_tpr_max", "mia_auc"]
    rows = []
    for (m, e), b in S.groupby(["method", "epsilon"]):
        r = {"method": m, "eps": None if math.isinf(e) else float(e), "seeds": int(b.seed.nunique())}
        for c in cols:
            v = b[c].dropna() if c in b else pd.Series(dtype=float)
            r[c] = float(v.mean()) if len(v) else None
            r[c + "_sd"] = float(v.std()) if len(v) > 1 else None
        rows.append(r)
    core.save_json("study.json", {"run": core.CFG["study_run"], "epochs": None, "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain every model")
    ap.add_argument("--reeval", action="store_true", help="re-evaluate saved models only (after metric changes)")
    args = ap.parse_args()
    ev = add_cost_of_privacy(reevaluate()) if args.reeval else train_served(args.force)
    proxy_and_unawareness(ev)
    study()
    print("done")


if __name__ == "__main__":
    main()
