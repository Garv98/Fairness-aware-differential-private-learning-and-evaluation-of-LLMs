"""Backend logic for the FairDP demo: data, training, evaluation, inference,
mitigation, recourse, documentation.

All settings come from server/config.yaml; all models, accounting and metrics
come from the research package (src/fairdp) and the experiment registry
(experiments/run.py), so the demo serves exactly what the experiments test.
"""
from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from importlib import metadata
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from fairdp.attacks.mia import loss_threshold_mia  # noqa: E402
from fairdp.data.loaders import load_dataset  # noqa: E402
from fairdp.data.preprocess import prepare  # noqa: E402
from fairdp.dp.accounting import default_delta  # noqa: E402
from fairdp.dp.dpsgd import DPConfig, train_dp  # noqa: E402
from fairdp.groups import CellIndex, intersect  # noqa: E402
from fairdp.metrics import fairness as F  # noqa: E402
from fairdp.metrics.ethics import consistency, group_fairness  # noqa: E402
from fairdp.models import MLP, predict_proba  # noqa: E402
from fairdp.postprocess import apply_thresholds, fit_group_thresholds  # noqa: E402
from fairdp.train import NonPrivateConfig, train_nonprivate  # noqa: E402
from run import NONPRIVATE, PRIVATE  # noqa: E402

CFG = yaml.safe_load((ROOT / "server" / "config.yaml").read_text(encoding="utf-8"))
ART = ROOT / "server" / "artifacts"
MODELS_DIR, PROBS_DIR = ART / "models", ART / "probs"
EPSILONS = [float(e) for e in CFG["epsilons"]]
METHODS = CFG["methods"]
NAMES = {m["id"]: m["name"] for m in METHODS}
THRESH = float(CFG["decision_threshold"])
SENSITIVE_FIELDS = ["sex", "race"]


def is_private(method: str) -> bool:
    if method in PRIVATE:
        return True
    if method in NONPRIVATE:
        return False
    raise KeyError(f"unknown method {method!r}")


def eps_list(method: str):
    return EPSILONS if is_private(method) else [None]


def key(method: str, eps) -> str:
    return f"{method}__{'inf' if eps is None else f'{float(eps):g}'}"


# ------------------------------------------------------------------- data
@lru_cache(maxsize=1)
def data():
    ds = load_dataset(CFG["dataset"])
    d = prepare(ds, seed=CFG["seed"], **CFG["split"])
    cells = CellIndex(d.train.groups, d.group_attrs)
    gy = d.train.groups.assign(_y=d.train.y.astype(str))
    cells_y = CellIndex(gy, d.group_attrs + ["_y"])
    return ds, d, cells, cells_y, gy


@lru_cache(maxsize=1)
def schema() -> dict:
    ds, d, *_ = data()
    raw = d.train.raw
    cat_fields = [c for c in ds.categorical if c in raw.columns]
    num_fields = [c for c in ds.numerical if c in raw.columns and c not in ("fnlwgt", "education-num")]
    cats = {f: raw[f].dropna().astype(str).value_counts().index.tolist() for f in cat_fields}
    nums = {f: {"min": float(raw[f].min()), "max": float(raw[f].max()), "typical_max": float(raw[f].quantile(0.995)), "median": float(raw[f].median())} for f in num_fields}
    edu = raw.dropna(subset=["education"]).groupby(raw["education"].astype(str))["education-num"].median()
    return {"categorical": cats, "numeric": nums,
            "education_num": {k: float(v) for k, v in edu.items()},
            "education_order": edu.sort_values().index.tolist(),
            "hidden_defaults": {"fnlwgt": float(raw["fnlwgt"].median())},
            "fields": SENSITIVE_FIELDS + [f for f in num_fields + cat_fields if f not in SENSITIVE_FIELDS]}


def validate_applicant(a: dict) -> list[str]:
    """Return a list of problems (empty = valid) against the data schema."""
    sc, errs = schema(), []
    for f in sc["fields"]:
        if f not in a:
            errs.append(f"missing field '{f}'")
            continue
        v = a[f]
        if f in sc["numeric"]:
            try:
                x = float(v)
            except (TypeError, ValueError):
                errs.append(f"'{f}' must be a number"); continue
            lo, hi = sc["numeric"][f]["min"], sc["numeric"][f]["max"]
            if not math.isfinite(x) or x < lo or x > hi:
                errs.append(f"'{f}' must be between {lo:g} and {hi:g}")
        elif str(v) not in sc["categorical"][f]:
            errs.append(f"'{f}' has unknown value '{v}'")
    return errs


def majority_identity() -> dict:
    raw = data()[1].train.raw
    return {f: str(raw[f].astype(str).value_counts().index[0]) for f in SENSITIVE_FIELDS}


# --------------------------------------------------------------- training
def method_cfg(method: str, eps: float | None, epochs: int | None = None) -> tuple[bool, dict]:
    epochs = int(epochs or CFG["epochs"])
    seed = 1000 * CFG["seed"] + 17
    if not is_private(method):
        return False, {**CFG["nonprivate"], **NONPRIVATE[method], "epochs": epochs, "seed": seed}
    ov = {k: v for k, v in PRIVATE[method].items() if not k.startswith("_")}
    return True, {**CFG["dp"], **ov, "epochs": epochs, "target_epsilon": float(eps), "seed": seed}


def new_model() -> MLP:
    return MLP(data()[1].train.X.shape[1], tuple(CFG["model"]["hidden"]))


def train(method: str, eps: float | None, epochs: int | None = None, callback=None):
    _, d, cells, cells_y, gy = data()
    private, cfg = method_cfg(method, eps, epochs)
    model = new_model()
    if not private:
        model = train_nonprivate(model, d.train.X, d.train.y, cells_y.encode(gy), cells_y.n_cells, NonPrivateConfig(**cfg), callback=callback)
        return model, {"private": False, "config": cfg}
    dcfg = DPConfig(**cfg)
    c_obj, c_ids = (cells_y, cells_y.encode(gy)) if dcfg.uses_label_cells else (cells, cells.encode(d.train.groups))
    model, log = train_dp(model, d.train.X, d.train.y, c_ids, c_obj, dcfg, callback=callback)
    from dataclasses import asdict
    return model, {"private": True, "config": cfg, "noise": asdict(log.noise), "history": log.history,
                   "weights": dict(zip(c_obj.names, log.final_cell_weights))}


# ------------------------------------------------------------- evaluation
def quick_eval(model) -> dict:
    _, d, *_ = data()
    p = predict_proba(model, d.test.X)
    s, _ = F.fairness_report(d.test.y, (p >= THRESH).astype(int), d.test.groups, d.group_attrs, p)
    return {k: s.get(k) for k in ("accuracy", "balanced_accuracy", "recall", "intersection_worst_tpr", "intersection_tpr_gap")}


def ethics_block(y, yhat, prob, groups: pd.DataFrame, attrs, X=None) -> dict:
    four = CFG["scorecard"]["four_fifths_ratio"]
    out = {a: group_fairness(y, yhat, prob, groups[a], four) for a in list(attrs) + ["intersection"]}
    if X is not None:
        out["consistency"] = consistency(X, yhat, seed=CFG["seed"])
    return out


def full_eval(model) -> dict:
    _, d, *_ = data()
    p_test, p_val = predict_proba(model, d.test.X), predict_proba(model, d.val.X)
    yhat = (p_test >= THRESH).astype(int)
    summary, tables = F.fairness_report(d.test.y, yhat, d.test.groups, d.group_attrs, p_test)
    mia, mia_tab = loss_threshold_mia(model, d.train, d.test, d.group_attrs, seed=CFG["seed"])
    return {"summary": clean(summary), "groups": {a: clean(t.to_dict("records")) for a, t in tables.items()},
            "ethics": clean(ethics_block(d.test.y, yhat, p_test, d.test.groups, d.group_attrs, d.test.X)),
            "mia": clean(mia), "mia_groups": clean(mia_tab.to_dict("records")) if len(mia_tab) else [],
            "_probs": {"val": p_val, "test": p_test}}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 5)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def model_path(method: str, eps) -> Path:
    return MODELS_DIR / f"{key(method, eps)}.pt"


@lru_cache(maxsize=64)
def load_model(method: str, eps):
    m = new_model()
    m.load_state_dict(torch.load(model_path(method, eps), map_location="cpu"))
    m.eval()
    return m


@lru_cache(maxsize=64)
def load_probs(k: str) -> dict:
    z = np.load(PROBS_DIR / f"{k}.npz")
    return {"val": z["val"], "test": z["test"]}


def available_models() -> list[dict]:
    out = []
    for m in METHODS:
        for e in eps_list(m["id"]):
            if model_path(m["id"], e).exists():
                out.append({"method": m["id"], "name": m["name"], "private": is_private(m["id"]), "eps": e})
    return out


def save_json(name: str, obj):
    ART.mkdir(parents=True, exist_ok=True)
    (ART / name).write_text(json.dumps(clean(obj), indent=1))


def read_json(name: str):
    p = ART / name
    return json.loads(p.read_text()) if p.exists() else None


# -------------------------------------------------------------- inference
def applicant_frame(rows: list[dict]) -> pd.DataFrame:
    sc = schema()
    out = []
    for a in rows:
        r = {**sc["hidden_defaults"], **{f: a[f] for f in sc["fields"]}}
        r["education-num"] = sc["education_num"].get(str(a["education"]), float(np.median(list(sc["education_num"].values()))))
        for f in sc["numeric"]:
            r[f] = float(r[f])
        out.append(r)
    return pd.DataFrame(out)


def encode(df: pd.DataFrame) -> np.ndarray:
    return np.asarray(data()[1].preprocessor.transform(df), dtype=np.float32)


def score(models: list[tuple[str, float | None]], rows: list[dict]) -> dict:
    X = encode(applicant_frame(rows))
    return {key(m, e): predict_proba(load_model(m, e), X) for m, e in models}


def group_of(a: dict) -> dict:
    _, d, *_ = data()
    g = d.binner(pd.DataFrame([{"age": float(a["age"]), "sex": a["sex"], "race": a["race"]}]))[d.group_attrs]
    return {**{k: str(v) for k, v in g.iloc[0].items()}, "intersection": str(intersect(g, d.group_attrs).iloc[0])}


def explain(model, a: dict) -> list[dict]:
    """Gradient x input on the encoded features, summed back to raw fields."""
    _, d, *_ = data()
    x = torch.tensor(encode(applicant_frame([a])), requires_grad=True)
    torch.sigmoid(model(x)).sum().backward()
    contrib = (x.grad * x).detach().numpy()[0]
    fields = schema()["fields"] + ["education-num"]
    agg: dict[str, float] = {}
    for name, c in zip(d.feature_names, contrib):
        base = name.split("__", 1)[1]
        field = next((f for f in sorted(fields, key=len, reverse=True) if base == f or base.startswith(f + "_")), None)
        if field is None:
            continue
        field = "education" if field == "education-num" else field
        agg[field] = agg.get(field, 0.0) + float(c)
    return [{"field": k, "value": round(v, 4)} for k, v in sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:6]]


def recourse(method: str, eps, a: dict) -> dict | None:
    """Cheapest change to actionable features that flips the decision to
    positive (Ustun et al., 2019 style grid search). None if already positive."""
    rc, sc = CFG["recourse"], schema()
    order = sc["education_order"]
    if str(a["education"]) not in order:
        return {"found": False, "searched": 0}
    cur_e = order.index(str(a["education"]))
    hp = rc["hours-per-week"]
    h0 = float(a["hours-per-week"])
    edu_opts = range(0, min(rc["education"]["max_levels"], len(order) - 1 - cur_e) + 1)
    n_steps = int(max(0.0, hp["max"] - h0) // hp["step"]) if np.isfinite(h0) else 0
    hour_opts = [h0 + k * hp["step"] for k in range(0, min(n_steps, 200) + 1)]
    cands, costs, changes = [], [], []
    for de in edu_opts:
        for h in hour_opts:
            cands.append({**a, "education": order[cur_e + de], "hours-per-week": h})
            costs.append(de * rc["education"]["cost_per_level"] + (h - h0) / hp["step"] * hp["cost_per_step"])
            changes.append({"education": order[cur_e + de] if de else None, "hours-per-week": h if h != h0 else None})
    p = score([(method, eps)], cands)[key(method, eps)]
    if p[0] >= THRESH:
        return None
    ok = [i for i in range(len(cands)) if p[i] >= THRESH]
    if not ok:
        return {"found": False, "searched": len(cands)}
    i = min(ok, key=lambda j: (costs[j], -p[j]))
    return {"found": True, "cost": round(costs[i], 2), "prob": round(float(p[i]), 4),
            "changes": {k: v for k, v in changes[i].items() if v is not None}, "searched": len(cands)}


def predict(a: dict, eps: float) -> dict:
    served = [(m["id"], None if not is_private(m["id"]) else eps) for m in METHODS]
    served = [(m, e) for m, e in served if model_path(m, e).exists()]
    probs = score(served, [a])
    ref = {**a, **majority_identity()}
    out = []
    for m, e in served:
        p = float(probs[key(m, e)][0])
        out.append({"method": m, "name": NAMES[m], "private": e is not None, "eps": e, "prob": round(p, 4),
                    "decision": int(p >= THRESH), "explain": explain(load_model(m, e), a),
                    "recourse": recourse(m, e, a), "recourse_reference": recourse(m, e, ref)})
    sc = schema()
    combos = [{**a, "sex": s, "race": r} for s in sc["categorical"]["sex"] for r in sc["categorical"]["race"]]
    cprobs = score(served, combos)
    cf = [{"method": m, "name": NAMES[m], "cells": [{"sex": c["sex"], "race": c["race"], "prob": round(float(p), 4)} for c, p in zip(combos, cprobs[key(m, e)])]} for m, e in served]
    return {"group": group_of(a), "eps": eps, "threshold": THRESH, "reference_identity": majority_identity(), "models": out, "counterfactual": cf}


_SAMPLES: list[dict] = []


def sample_applicants() -> list[dict]:
    """Pick demo applicants automatically: complete records where the served
    models disagree most (std of predicted probability at the strictest
    epsilon), at most one per intersectional group, largest groups first."""
    _, d, *_ = data()
    sc = schema()
    raw = d.test.raw.reset_index(drop=True)
    ok = raw[sc["fields"]].notna().all(axis=1).to_numpy()
    eps0 = min(EPSILONS)
    served = [(m["id"], eps0 if is_private(m["id"]) else None) for m in METHODS]
    served = [(m, e) for m, e in served if model_path(m, e).exists()]
    if not served:
        return []
    if _SAMPLES:
        return _SAMPLES
    P = np.stack([predict_proba(load_model(m, e), d.test.X) for m, e in served])
    disagreement = P.std(axis=0)
    df = pd.DataFrame({"cell": d.test.groups["intersection"].to_numpy(), "dis": disagreement, "ok": ok, "y": d.test.y})
    df = df[df.ok]
    sizes = d.test.groups["intersection"].value_counts()
    picks = []
    for cell in sizes.index:
        sub = df[df.cell == cell]
        if len(sub) == 0:
            continue
        i = int(sub.dis.idxmax())
        r = raw.iloc[i]
        picks.append({"label": cell.replace("|", " · "), "truth": int(d.test.y[i]), "disagreement": round(float(sub.dis.max()), 4),
                      "applicant": {f: (float(r[f]) if f in sc["numeric"] else str(r[f])) for f in sc["fields"]}})
        if len(picks) >= CFG["demo"]["n_samples"]:
            break
    _SAMPLES[:] = sorted(picks, key=lambda p: -p["disagreement"])
    return _SAMPLES


# ------------------------------------------------------------- mitigation
def _thresholds_for(policy: str, attr: str, p_val, t: float):
    _, d, *_ = data()
    gv = d.val.groups[attr].astype(str).to_numpy()
    if policy == "single":
        return {"__default__": t}
    if policy == "equal_opportunity":
        return fit_group_thresholds(p_val, d.val.y, d.val.groups[attr], base_threshold=t)
    if policy == "demographic_parity":
        target = float((p_val >= t).mean())
        th = {"__default__": t}
        for g in np.unique(gv):
            m = gv == g
            if m.sum() >= F.MIN_SUPPORT:
                th[g] = float(np.quantile(p_val[m], 1 - target))
        return th
    raise ValueError(policy)


def apply_policy(k: str, policy: str, attr: str, t: float) -> dict:
    _, d, *_ = data()
    pr = load_probs(k)
    th = _thresholds_for(policy, attr, pr["val"], t)
    yhat = apply_thresholds(pr["test"], d.test.groups[attr], th)
    summary, _ = F.fairness_report(d.test.y, yhat, d.test.groups, d.group_attrs, pr["test"])
    eth = ethics_block(d.test.y, yhat, pr["test"], d.test.groups, d.group_attrs)
    return clean({"policy": policy, "attribute": attr, "threshold": t, "thresholds": th, "summary": summary, "ethics": eth})


@lru_cache(maxsize=128)
def threshold_sweep(k: str, attr: str) -> list[dict]:
    _, d, *_ = data()
    pr = load_probs(k)
    out = []
    g = CFG["threshold_grid"]
    for t in np.round(np.arange(g["min"], g["max"] + 1e-9, g["step"]), 4):
        yhat = (pr["test"] >= t).astype(int)
        s, _ = F.fairness_report(d.test.y, yhat, d.test.groups, d.group_attrs, None)   # AUC not needed per threshold
        g = group_fairness(d.test.y, yhat, pr["test"], d.test.groups[attr], CFG["scorecard"]["four_fifths_ratio"])
        out.append({"t": float(t), "accuracy": s["accuracy"], "balanced_accuracy": s["balanced_accuracy"],
                    "worst_tpr": s.get("intersection_worst_tpr"), "dp_diff": g["demographic_parity_diff"],
                    "di_ratio": g["disparate_impact_ratio"], "eo_diff": g["equal_opportunity_diff"]})
    return clean(out)


# ------------------------------------------------------ accountability docs
@lru_cache(maxsize=1)
def provenance() -> dict:
    ds, d, *_ = data()
    fp = hashlib.sha256(pd.util.hash_pandas_object(ds.X, index=False).values.tobytes()).hexdigest()
    try:
        rc = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        commit = rc.stdout.strip() if rc.returncode == 0 else None
        st = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, timeout=5) if commit else None
        dirty = bool(st.stdout.strip()) if st is not None and st.returncode == 0 else None
    except Exception:
        commit, dirty = None, None
    vers = {}
    for pkg in ("torch", "opacus", "scikit-learn", "numpy", "pandas", "fastapi"):
        try:
            vers[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return {"dataset": CFG["dataset"], "dataset_rows": int(len(ds)), "dataset_fingerprint": fp[:16],
            "split": CFG["split"], "seed": CFG["seed"], "train_size": int(len(d.train.y)), "val_size": int(len(d.val.y)),
            "test_size": int(len(d.test.y)), "n_features": int(d.train.X.shape[1]), "group_attrs": d.group_attrs,
            "n_cells": int(d.test.groups["intersection"].nunique()), "git_commit": commit, "git_dirty": dirty,
            "python": platform.python_version(), "packages": vers, "config": CFG,
            "min_support": F.MIN_SUPPORT, "min_positives": F.MIN_POSITIVES, "delta": default_delta(len(d.train.y))}


def checks(entry: dict, study: dict | None) -> list[dict]:
    """Responsible-AI checks for one model, each with evidence and pass/fail."""
    sc, eth, s = CFG["scorecard"], entry["ethics"], entry["summary"]
    out = []
    eps = entry.get("eps")
    out.append({"principle": "Privacy", "check": "Formal differential privacy", "value": f"ε = {eps:g}" if eps else "none",
                "pass": eps is not None and eps <= sc["epsilon_max"]})
    out.append({"principle": "Privacy", "check": "Membership attack AUC", "value": entry["mia"].get("mia_auc"),
                "pass": (entry["mia"].get("mia_auc") or 1) <= sc["mia_auc_max"]})
    for a in SENSITIVE_FIELDS:
        e = eth[a]
        out.append({"principle": "Fairness", "check": f"Four-fifths rule ({a})", "value": e["disparate_impact_ratio"], "pass": e["four_fifths_pass"]})
        out.append({"principle": "Fairness", "check": f"Equalized odds ({a})", "value": e["equalized_odds_diff"],
                    "pass": e["equalized_odds_diff"] is not None and e["equalized_odds_diff"] <= sc["equalized_odds_max"]})
        out.append({"principle": "Fairness", "check": f"Calibration gap ({a})", "value": e["calibration_gap"],
                    "pass": e["calibration_gap"] is not None and e["calibration_gap"] <= sc["calibration_gap_max"]})
    out.append({"principle": "Fairness", "check": "Individual consistency (kNN)", "value": eth.get("consistency"),
                "pass": (eth.get("consistency") or 0) >= sc["consistency_min"]})
    out.append({"principle": "Transparency", "check": "Per-decision explanation (gradient × input)", "value": "available", "pass": True})
    out.append({"principle": "Accountability", "check": "Provenance (data hash, config, versions)", "value": "recorded", "pass": True})
    if study:
        r = next((x for x in study.get("rows", []) if x["method"] == entry["method"] and (x["eps"] == eps or (x["eps"] is None and eps is None))), None)
        if r:
            out.append({"principle": "Robustness", "check": f"Worst-group recall across {r['seeds']} seeds", "value": f"{_f(r['intersection_worst_tpr'], 2)} ± {_f(r['intersection_worst_tpr_sd'], 2)}", "pass": None})
    return out


def _f(v, d=3) -> str:
    return "–" if v is None or (isinstance(v, float) and not math.isfinite(v)) else f"{v:.{d}f}"


def model_card(k: str) -> str:
    ev = (read_json("evaluation.json") or {}).get(k)
    if ev is None:
        raise KeyError(k)
    tr = (read_json("training.json") or {}).get(k, {})
    pv = provenance()
    s, eth = ev["summary"], ev["ethics"]
    pct = lambda v: "–" if v is None else f"{v * 100:.1f}%"  # noqa: E731
    lines = [
        f"# Model card: {ev['name']}" + (f" (ε = {ev['eps']:g})" if ev["eps"] else ""),
        "", f"*Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from the FairDP demo backend.*", "",
        "## Model details",
        f"- Architecture: MLP {' → '.join(map(str, CFG['model']['hidden']))} → 1, trained {CFG['epochs']} epochs, seed {CFG['seed']}",
        f"- Training method: `{ev['method']}`" + (f"; DP-SGD with RDP accounting, accounted ε = {_f(tr.get('noise', {}).get('epsilon')) if tr.get('noise') else 'unknown (training log missing)'}, δ = {pv['delta']:.1e}, noise multiplier σ = {_f(tr.get('noise', {}).get('sigma_grad')) if tr.get('noise') else 'unknown'}" if ev["private"] else "; no privacy guarantee"),
        f"- Code version: {pv['git_commit'] or 'unknown'}{' (uncommitted changes)' if pv['git_dirty'] else ''}; torch {pv['packages'].get('torch')}, opacus {pv['packages'].get('opacus')}",
        "", "## Intended use",
        "- Research and teaching demonstration of privacy–fairness trade-offs.",
        f"- Task: predict whether income exceeds $50k (UCI Adult). Decisions at threshold {THRESH}.",
        "- Not for real lending, hiring or benefit decisions. Credit scoring is a high-risk use under the EU AI Act (Annex III) and would need a full conformity assessment.",
        "", "## Data",
        f"- Dataset `{pv['dataset']}`: {pv['dataset_rows']:,} rows (fingerprint `{pv['dataset_fingerprint']}`), split {pv['train_size']:,} / {pv['val_size']:,} / {pv['test_size']:,} (train / val / test).",
        f"- Sensitive attributes: {', '.join(pv['group_attrs'])}; {pv['n_cells']} intersectional groups in the test set.",
        "", "## Performance (test set)",
        f"| Accuracy | Balanced accuracy | Recall | ROC-AUC |", "|---|---|---|---|",
        f"| {pct(s.get('accuracy'))} | {pct(s.get('balanced_accuracy'))} | {pct(s.get('recall'))} | {pct(s.get('roc_auc'))} |",
        "", "## Fairness",
        "| Attribute | Disparate-impact ratio | Four-fifths rule | Equal-opportunity diff | Equalized-odds diff | Calibration gap |", "|---|---|---|---|---|---|",
    ]
    for a in pv["group_attrs"] + ["intersection"]:
        e = eth[a]
        lines.append(f"| {a} | {e['disparate_impact_ratio'] if e['disparate_impact_ratio'] is None else round(e['disparate_impact_ratio'], 3)} | {'pass' if e['four_fifths_pass'] else 'fail' if e['four_fifths_pass'] is not None else '–'} | {pct(e['equal_opportunity_diff'])} | {pct(e['equalized_odds_diff'])} | {pct(e['calibration_gap'])} |")
    lines += [f"", f"- Worst intersectional-group recall: {pct(s.get('intersection_worst_tpr'))}; individual consistency (kNN): {_f(eth.get('consistency'))}",
              "", "## Privacy evaluation",
              f"- Loss-threshold membership-inference AUC: {_f(ev['mia'].get('mia_auc'))} (0.5 = chance).",
              "", "## Limitations and ethical considerations",
              "- The privacy guarantee covers model training only; preprocessing statistics (scaler, encoder) and the list of group categories are not privatised.",
              f"- Groups with fewer than {pv['min_support']} people or {pv['min_positives']} positives are excluded from gap metrics; their estimates are unreliable.",
              "- Group gaps describe outcomes and do not by themselves prove causal discrimination; race and sex categories in census data are coarse.",
              "- Results come from one training seed; see the multi-seed study for variability. Significance claims need ≥ 10 seeds (5-seed Wilcoxon tests cannot reach p < 0.05).",
              "- Method settings were chosen on a pilot run of the same dataset; that selection is not covered by the privacy guarantee.",
              "- Group categories (the cell schema) are treated as public information; IG-DPSGD sees roughly batch-size / #cells people per cell per step, so its noisy group statistics are dominated by noise for small cells.",
              "- Post-processed (group-threshold) results use non-private validation data and are not covered by the same ε."]
    return "\n".join(lines)
