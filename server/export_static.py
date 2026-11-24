"""Export everything the dashboard needs to run WITHOUT the Python backend.

    python server/export_static.py        (after train_models.py)

Writes web/public/static/:
  bundle.json        meta, evaluation, study, training, proxy, provenance, credit,
                     LLM results, scorecards, model cards (exactly the API responses)
  engine.json        preprocessing spec, group spec, recourse/config needed for inference
  models.json        weights of every served model (float32 values, rounded to 7 sig. digits)
  scores.json        validation/test labels, groups and model scores (mitigation, sweep)
  lite.json + lite.bin   a training subset for the in-browser DP-SGD demo
  parity.json        reference outputs from PyTorch for parity tests

Nothing here retrains or modifies the served models; it only reads them.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import core
import app as api  # reuse the exact endpoint implementations
from fairdp.dp.dpsgd import DPConfig
from fairdp.train import NonPrivateConfig

OUT = core.ROOT / "web" / "public" / "static"


def dump(name: str, obj, exact: bool = False) -> None:
    """exact=True keeps full float precision (weights, preprocessing); API
    responses go through core.clean exactly like the live server."""
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    p.write_text(json.dumps(obj if exact else core.clean(obj), separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"{name:14s} {p.stat().st_size / 1024:9.1f} KB")


def r7(a) -> list:
    return [float(f"{v:.7g}") for v in np.asarray(a, dtype=np.float64).ravel()]


def main() -> None:
    ds, d, *_ = core.data()
    sc = core.schema()
    pre = d.preprocessor

    # ---------------------------------------------------------- preprocessing
    num_pipe, num_cols = pre.named_transformers_["num"], list(pre.transformers_[0][2])
    cat_pipe, cat_cols = pre.named_transformers_["cat"], list(pre.transformers_[1][2])
    imp_n, scaler = num_pipe.named_steps["imputer"], num_pipe.named_steps["scaler"]
    imp_c, enc = cat_pipe.named_steps["imputer"], cat_pipe.named_steps["encoder"]
    fields = sc["fields"] + ["education-num"]
    feature_field = []
    for name in d.feature_names:
        base = name.split("__", 1)[1]
        f = next((x for x in sorted(fields, key=len, reverse=True) if base == x or base.startswith(x + "_")), None)
        feature_field.append(None if f is None else ("education" if f == "education-num" else f))
    engine = {
        "numeric": [{"name": c, "median": float(m), "mean": float(mu), "scale": float(s)}
                    for c, m, mu, s in zip(num_cols, imp_n.statistics_, scaler.mean_, scaler.scale_)],
        "categorical": [{"name": c, "mode": str(mo), "categories": [str(x) for x in cats]}
                        for c, mo, cats in zip(cat_cols, imp_c.statistics_, enc.categories_)],
        "feature_names": list(d.feature_names),
        "feature_field": feature_field,
        "hidden_defaults": sc["hidden_defaults"],
        "education_num": sc["education_num"],
        "education_order": sc["education_order"],
        "groups": {"attrs": d.group_attrs, "age_edges": [30, 50], "age_labels": ["<=30", "31-50", ">50"],
                   "race_keep": ["White", "Black"], "race_other": "Other"},
        "threshold": core.THRESH,
        "recourse": core.CFG["recourse"],
        "four_fifths": core.CFG["scorecard"]["four_fifths_ratio"],
        "min_support": 30, "min_positives": 10,
    }
    from fairdp.metrics.fairness import MIN_POSITIVES, MIN_SUPPORT
    engine["min_support"], engine["min_positives"] = MIN_SUPPORT, MIN_POSITIVES
    dump("engine.json", engine, exact=True)

    # ---------------------------------------------------------------- models
    models = {}
    for m in core.available_models():
        mdl = core.load_model(m["method"], m["eps"])
        layers = [mod for mod in mdl.network if isinstance(mod, torch.nn.Linear)]
        models[core.key(m["method"], m["eps"])] = [{"in": L.in_features, "out": L.out_features,
                                                    "W": r7(L.weight.detach().numpy()), "b": r7(L.bias.detach().numpy())} for L in layers]
    dump("models.json", models, exact=True)

    # --------------------------------------------- scores for mitigation/sweep
    def groups_of(split):
        out = {}
        for a in d.group_attrs + ["intersection"]:
            labels = split.groups[a].astype(str)
            cats = sorted(labels.unique())
            out[a] = {"labels": cats, "codes": labels.map({c: i for i, c in enumerate(cats)}).astype(int).tolist()}
        return out
    scores = {"val": {"y": d.val.y.tolist(), "groups": groups_of(d.val)},
              "test": {"y": d.test.y.tolist(), "groups": groups_of(d.test)}, "probs": {}}
    for m in core.available_models():
        k = core.key(m["method"], m["eps"])
        pr = core.load_probs(k)
        scores["probs"][k] = {"val": [round(float(v), 6) for v in pr["val"]], "test": [round(float(v), 6) for v in pr["test"]]}
    dump("scores.json", scores, exact=True)
    # encoded test rows for the parity test only (server/artifacts is gitignored, never deployed)
    (core.ART / "parity_test_X.bin").write_bytes(np.asarray(d.test.X, dtype="<f4").tobytes())

    # ------------------------------------------ API responses (exactly the same)
    meta = api.meta()
    bundle = {
        "meta": meta, "evaluation": api.evaluation(), "study": api.study(), "training": api.training(),
        "proxy": api.proxy(), "provenance": api.provenance(), "credit": api.credit(),
        "scorecard": {str(e): api.scorecard(e) for e in core.EPSILONS},
        "model_cards": {k: core.model_card(k) for k in (core.key(m["method"], m["eps"]) for m in core.available_models())},
    }
    llm = api.llm_info()
    llm.pop("providers", None); llm.pop("models", None)          # live parts come from the server route
    bundle["llm"] = llm
    dump("bundle.json", bundle)

    # ----------------------------------------- in-browser DP-SGD training subset
    rng = np.random.default_rng(core.CFG["seed"])
    n_tr, n_te = 8000, 3000
    itr = np.sort(rng.choice(len(d.train.y), n_tr, replace=False))
    ite = np.sort(rng.choice(len(d.test.y), n_te, replace=False))
    Xtr, Xte = d.train.X[itr].astype(np.float32), d.test.X[ite].astype(np.float32)
    (OUT / "lite.bin").write_bytes(np.concatenate([Xtr.ravel(), Xte.ravel()]).astype("<f4").tobytes())
    gtr = d.train.groups.iloc[itr].reset_index(drop=True)
    gte = d.test.groups.iloc[ite].reset_index(drop=True)
    gy = gtr.assign(_y=d.train.y[itr].astype(str))
    from fairdp.groups import CellIndex
    cells_y = CellIndex(gy, d.group_attrs + ["_y"])
    lite = {
        "n_train": n_tr, "n_test": n_te, "dim": int(Xtr.shape[1]),
        "y_train": d.train.y[itr].tolist(), "y_test": d.test.y[ite].tolist(),
        "cell_train": cells_y.encode(gy).tolist(), "n_cells": cells_y.n_cells, "cell_names": cells_y.names,
        "label_membership": cells_y.marginal_membership["_y"].astype(int).tolist(),
        "attr_membership": [M.astype(int).tolist() for a, M in cells_y.marginal_membership.items()],
        "groups_test": groups_of(type("S", (), {"groups": gte})()),
        "hidden": [32, 16], "batch": 256, "lr": 0.003, "clip": 1.0, "aux_frac": 0.05,
        "delta": 1 / n_tr ** 1.1,
        # same component settings as the full trainer (experiments/run.py registry + dataclass defaults)
        "methods": {m["id"]: {"private": core.is_private(m["id"]),
                              **(core.PRIVATE.get(m["id"]) or core.NONPRIVATE.get(m["id"]))} for m in core.METHODS},
        "dp_defaults": {f.name: f.default for f in dataclasses.fields(DPConfig)
                        if f.name in ("loss_bound", "dro_eta", "dro_warmup_frac", "weight_cap", "smoothing_strength", "ema")},
        "nonprivate": {"batch_size": core.CFG["nonprivate"]["batch_size"], "lr": core.CFG["nonprivate"]["lr"],
                       "dro_eta": NonPrivateConfig().dro_eta},
        "sensitive": core.SENSITIVE_FIELDS,
        "note": f"Lite in-browser training: {n_tr:,} of {len(d.train.y):,} training people, MLP 32→16, evaluated on {n_te:,} test people.",
    }
    dump("lite.json", lite, exact=True)

    # ---------------------------------------------------- parity references
    samples = meta["samples"]
    preds = {}
    for s in samples[:4]:
        for e in core.EPSILONS:
            preds[f"{s['label']}|{e}"] = core.predict(s["applicant"], e)
    acc = {f"{s}|{b}|{ep}": api._accountant(s, b, ep) for s, b, ep in [(1.0, 256, 10), (0.6, 512, 20), (2.5, 128, 5), (1.3, 1024, 40)]}
    mit = {}
    ours = next(m["id"] for m in core.METHODS if m.get("ours"))
    for pol in ("single", "equal_opportunity", "demographic_parity"):
        for attr in ("sex", "race", "intersection"):
            for t in (0.3, 0.5):
                mit[f"{pol}|{attr}|{t}"] = core.apply_policy(core.key(ours, core.EPSILONS[0]), pol, attr, t)
    dump("parity.json", {"predict": preds, "accountant": acc, "mitigation": mit, "ours": ours,
                         "sweep": core.threshold_sweep(core.key(ours, core.EPSILONS[0]), "race")})
    print("done ->", OUT)


if __name__ == "__main__":
    sys.exit(main())
