"""Grid runner: dataset x method x epsilon x seed.

Usage:
    python experiments/run.py --config experiments/configs/smoke.yaml
    python experiments/run.py --config experiments/configs/main.yaml --only-dataset adult

Outputs (resumable; finished runs are skipped):
    results/runs/<name>/raw/<dataset>__<method>__eps<e>__s<seed>.parquet  summary rows
    results/runs/<name>/raw/<...>.groups.parquet                         per-group tables
    results/runs/<name>/raw/<...>.json                                   noise plan + train log
    results/runs/<name>/summary.parquet, groups.parquet                  combined
Every model is evaluated twice: raw threshold 0.5 (pp=False) and with
validation-fitted intersectional thresholds (pp=True).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fairdp.data.loaders import load_dataset            # noqa: E402
from fairdp.data.preprocess import prepare              # noqa: E402
from fairdp.groups import CellIndex                     # noqa: E402
from fairdp.models import MLP, predict_proba            # noqa: E402
from fairdp.train import NonPrivateConfig, train_nonprivate  # noqa: E402
from fairdp.dp.dpsgd import DPConfig, train_dp          # noqa: E402
from fairdp.metrics.fairness import fairness_report, dp_disparate_impact  # noqa: E402
from fairdp.postprocess import fit_group_thresholds, apply_thresholds     # noqa: E402
from fairdp.attacks.mia import loss_threshold_mia                           # noqa: E402

# ----------------------------------------------------------------------------
# Method registry. Non-private methods ignore epsilon (recorded as inf).
# ----------------------------------------------------------------------------
NONPRIVATE = {
    "erm": dict(method="erm"),
    "class_weighted": dict(method="class_weighted"),
    "groupdro": dict(method="groupdro"),
}
PRIVATE = {
    "dpsgd":            dict(clip_mode="flat"),
    "dpsgd_balanced":   dict(clip_mode="flat", balance_labels=True, aux_frac=0.05),
    "dpsgd_f":          dict(clip_mode="dpsgdf", aux_frac=0.05, smoothing=False),
    "global_adapt":     dict(clip_mode="global_adapt", aux_frac=0.05),
    # proposed (default chosen on the Adult pilot: flat clipping) + ablations
    "ig_dpsgd":         dict(clip_mode="flat", dro=True, smoothing=True, aux_frac=0.05),
    "ig_no_smoothing":  dict(clip_mode="flat", dro=True, smoothing=False, aux_frac=0.05),
    "ig_global_adapt":  dict(clip_mode="global_adapt", dro=True, smoothing=True, aux_frac=0.05),
    "ig_single_attr":   dict(clip_mode="flat", dro=True, smoothing=True, aux_frac=0.05,
                             _cells="first_attr"),  # DRO over first attribute x label only
}


def method_settings(name: str, cfg: dict) -> tuple[bool, dict]:
    """Return (is_private, overrides). Supports ``name@aux0.1`` suffixes."""
    base, _, suffix = name.partition("@")
    if base in NONPRIVATE:
        return False, dict(NONPRIVATE[base])
    if base not in PRIVATE:
        raise KeyError(f"unknown method {name!r}")
    ov = dict(PRIVATE[base])
    ov.update(cfg.get("method_overrides", {}).get(base, {}))
    if suffix.startswith("aux"):
        ov["aux_frac"] = float(suffix[3:])
    return True, ov


def run_key(ds: str, method: str, eps, seed: int) -> str:
    e = "inf" if eps is None or (isinstance(eps, float) and math.isinf(eps)) else f"{eps:g}"
    return f"{ds}__{method}__eps{e}__s{seed}"


def evaluate(model, data, tag: dict) -> tuple[list[dict], list[pd.DataFrame]]:
    attrs = data.group_attrs
    p_test = predict_proba(model, data.test.X)
    p_val = predict_proba(model, data.val.X)
    rows, tabs = [], []
    th = fit_group_thresholds(p_val, data.val.y, data.val.groups["intersection"])
    for pp in (False, True):
        y_hat = (apply_thresholds(p_test, data.test.groups["intersection"], th) if pp
                 else (p_test >= 0.5).astype(int))
        summary, tables = fairness_report(data.test.y, y_hat, data.test.groups, attrs, p_test)
        rows.append({**tag, "pp": pp, **summary})
        for a, t in tables.items():
            t = t.copy()
            t.insert(0, "attribute", a)
            for k, v in {**tag, "pp": pp}.items():
                t[k] = v
            tabs.append(t)
    return rows, tabs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only-dataset")
    ap.add_argument("--only-method")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    out = ROOT / "results" / "runs" / cfg["name"]
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))

    for dspec in cfg["datasets"]:
        dspec = dict(dspec)
        ds_name = dspec.pop("name")
        if args.only_dataset and ds_name != args.only_dataset:
            continue
        ds = load_dataset(ds_name, **dspec.pop("loader", {}))
        for seed in cfg["seeds"]:
            data = prepare(ds, seed=seed, group_attrs=dspec.get("group_attrs"))
            cells = CellIndex(data.train.groups, data.group_attrs)
            cid_train = cells.encode(data.train.groups)
            # intersection x label cells (Group DRO convention, Sagawa et al. 2020)
            gy = data.train.groups.assign(_y=data.train.y.astype(str))
            cells_y = CellIndex(gy, data.group_attrs + ["_y"])
            cid_train_y = cells_y.encode(gy)
            for method in cfg["methods"]:
                if args.only_method and method != args.only_method:
                    continue
                private, ov = method_settings(method, cfg)
                eps_list = cfg["epsilons"] if private else [math.inf]
                for eps in eps_list:
                    key = run_key(ds_name, method, eps, seed)
                    f_sum = raw / f"{key}.parquet"
                    if f_sum.exists() and not args.force:
                        continue
                    t0 = time.time()
                    torch_seed = 1000 * seed + 17
                    model = MLP(data.train.X.shape[1], tuple(cfg.get("hidden", [64, 32])))
                    tag = dict(dataset=ds_name, method=method, epsilon=float(eps),
                               seed=seed, private=private)
                    meta = {}
                    ov_run = {k: v for k, v in ov.items() if k != "_cells"}
                    cell_mode = ov.get("_cells", "intersection")
                    if private:
                        dcfg = DPConfig(**{**cfg.get("dp", {}), **ov_run,
                                           "target_epsilon": float(eps), "seed": torch_seed})
                        use_y = dcfg.uses_label_cells
                        c_obj, c_ids = (cells_y, cid_train_y) if use_y else (cells, cid_train)
                        if cell_mode == "first_attr":
                            a0 = [data.group_attrs[0]] + (["_y"] if use_y else [])
                            c_obj = CellIndex(gy, a0)
                            c_ids = c_obj.encode(gy)
                        model, log = train_dp(model, data.train.X, data.train.y,
                                              c_ids, c_obj, dcfg)
                        meta = {"noise": asdict(log.noise), "history": log.history,
                                "final_clip": log.final_clip,
                                "final_cell_weights": dict(zip(c_obj.names, log.final_cell_weights))}
                        tag.update(accounted_epsilon=log.noise.epsilon,
                                   delta=log.noise.delta, sigma_eff=log.noise.sigma_eff,
                                   sigma_grad=log.noise.sigma_grad, aux_frac=log.noise.aux_frac)
                    else:
                        ncfg = NonPrivateConfig(**{**cfg.get("nonprivate", {}), **ov_run,
                                                   "seed": torch_seed})
                        model = train_nonprivate(model, data.train.X, data.train.y,
                                                 cid_train_y, cells_y.n_cells, ncfg)
                    rows, tabs = evaluate(model, data, tag)
                    if cfg.get("mia", True):
                        mia_sum, mia_tab = loss_threshold_mia(model, data.train, data.test,
                                                              data.group_attrs, seed=seed)
                        for r in rows:
                            r.update(mia_sum)
                        if len(mia_tab):
                            mia_tab = mia_tab.assign(**tag)
                            mia_tab.to_parquet(raw / f"{key}.mia.parquet")
                    for r in rows:
                        r["train_seconds"] = time.time() - t0
                    pd.DataFrame(rows).to_parquet(f_sum)
                    pd.concat(tabs, ignore_index=True).to_parquet(raw / f"{key}.groups.parquet")
                    (raw / f"{key}.json").write_text(json.dumps(meta, default=float, indent=1))
                    acc = rows[0]["accuracy"]
                    wt = rows[0].get("intersection_worst_tpr", float("nan"))
                    print(f"[{key}] acc={acc:.4f} worst_int_tpr={wt:.3f} "
                          f"({time.time() - t0:.1f}s)", flush=True)

    combine(out)


def combine(out: Path):
    raw = out / "raw"
    sums = [pd.read_parquet(f) for f in sorted(raw.glob("*.parquet")) if not f.name.endswith((".groups.parquet", ".mia.parquet"))]
    if not sums:
        return
    summary = pd.concat(sums, ignore_index=True)
    groups = pd.concat([pd.read_parquet(f) for f in sorted(raw.glob("*.groups.parquet"))], ignore_index=True)
    # DP-induced disparate impact vs non-private ERM of the same dataset/seed
    di_rows = []
    for (ds, seed, pp), blk in groups.groupby(["dataset", "seed", "pp"]):
        ref = blk[(blk["method"] == "erm")]
        if ref.empty:
            continue
        for (m, e), b in blk.groupby(["method", "epsilon"]):
            row = dict(dataset=ds, seed=seed, pp=pp, method=m, epsilon=e)
            for a in b["attribute"].unique():
                d = dp_disparate_impact(ref[ref.attribute == a], b[b.attribute == a], "tpr")
                row.update({f"{a}_{k}": v for k, v in d.items()})
            di_rows.append(row)
    if di_rows:
        summary = summary.merge(pd.DataFrame(di_rows), on=["dataset", "seed", "pp", "method", "epsilon"], how="left")
    summary.to_parquet(out / "summary.parquet")
    summary.to_csv(out / "summary.csv", index=False)
    groups.to_parquet(out / "groups.parquet")
    mias = sorted(raw.glob("*.mia.parquet"))
    if mias:
        pd.concat([pd.read_parquet(f) for f in mias], ignore_index=True).to_parquet(out / "mia.parquet")
    print(f"combined {len(summary)} rows -> {out / 'summary.parquet'}")


if __name__ == "__main__":
    main()
