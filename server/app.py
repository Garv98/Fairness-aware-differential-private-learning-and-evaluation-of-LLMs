"""FairDP demo API (FastAPI).

    cd server && ../.venv/Scripts/python -m uvicorn app:app --port 8000
    first run: python train_models.py          interactive docs: /docs

Every value the dashboard shows comes from these endpoints; settings come
from server/config.yaml.
"""
from __future__ import annotations

import copy
import os
import re
import sys
from functools import lru_cache
import threading
import time
import uuid
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

import core


def _load_env_file(path):
    """Load KEY=VALUE lines from server/.env (gitignored) without overriding real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env_file(core.ROOT / "server" / ".env")
from fairdp.dp.accounting import calibrate_sigma, default_delta, epsilon_for, plan_noise
from fairdp.dp.dpsgd import DPConfig
from run import PRIVATE


@asynccontextmanager
async def lifespan(_app):
    core.data(); core.schema()          # load + preprocess once at startup
    yield

app = FastAPI(title="FairDP demo API", lifespan=lifespan)
# Allowed dashboard origins: config.yaml, plus deployment overrides via env vars
#   CORS_ORIGINS="https://my-app.vercel.app,https://other.example"   (exact origins)
#   CORS_ORIGIN_REGEX="https://my-app(-[a-z0-9-]+)?\.vercel\.app"   (e.g. Vercel preview URLs)
_origins = core.CFG["cors_origins"] + [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_origin_regex=os.environ.get("CORS_ORIGIN_REGEX") or None,
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def need(name: str):
    v = core.read_json(name)
    if v is None:
        raise HTTPException(503, "Artifacts missing: run `python server/train_models.py` first.")
    return v


def check_attr(attr: str):
    allowed = core.data()[1].group_attrs + ["intersection"]
    if attr not in allowed:
        raise HTTPException(400, f"attr must be one of {allowed}")


def check_model(method: str, eps: float | None):
    if method not in core.NAMES:
        raise HTTPException(400, f"unknown method {method!r}")
    e = eps if core.is_private(method) else None
    if not core.model_path(method, e).exists():
        raise HTTPException(404, f"no trained model for {core.key(method, e)}")
    return e


@app.get("/api/health")
def health():
    return {"ok": True, "models": len(core.available_models())}


@app.get("/api/meta")
def meta():
    pv = core.provenance()
    methods = [{**m, "private": core.is_private(m["id"]), "ours": bool(m.get("ours")),
                "baseline": bool(m.get("baseline")), "aux_frac": PRIVATE.get(m["id"], {}).get("aux_frac", 0.0)} for m in core.METHODS]
    return {"schema": core.schema(), "models": core.available_models(), "methods": methods,
            "epsilons": core.EPSILONS, "train_epsilons": core.CFG["train_epsilons"], "train_epochs": core.CFG["train_epochs"],
            "threshold": core.THRESH, "samples": core.sample_applicants(), "scorecard": core.CFG["scorecard"], "proxy_strength": core.CFG["proxy_strength"], "task": core.CFG["task"], "threshold_grid": core.CFG["threshold_grid"], "accountant_ui": core.CFG["accountant"],
            "recourse": core.CFG["recourse"], "reference_identity": core.majority_identity(),
            "info": {k: pv[k] for k in ("dataset", "dataset_rows", "train_size", "val_size", "test_size", "n_features", "group_attrs",
                                         "n_cells", "min_support", "min_positives", "delta", "seed", "split")}
            | {"hidden": core.CFG["model"]["hidden"], "epochs": core.CFG["epochs"], "dp": core.CFG["dp"]}}


class PredictIn(BaseModel):
    applicant: dict
    eps: float


@app.post("/api/predict")
def predict(body: PredictIn):
    if body.eps not in core.EPSILONS:
        raise HTTPException(400, f"eps must be one of {core.EPSILONS}")
    check_applicant(body.applicant)
    return core.predict(body.applicant, body.eps)


def check_applicant(a: dict):
    errs = core.validate_applicant(a)
    if errs:
        raise HTTPException(422, "; ".join(errs))


@app.get("/api/evaluation")
def evaluation():
    return need("evaluation.json")


@app.get("/api/study")
def study():
    return core.read_json("study.json") or {"rows": []}


@app.get("/api/training")
def training():
    return core.read_json("training.json") or {}


ACC = core.CFG["accountant"]


def _steps(epochs: int, q: float) -> int:
    """Same step count as the DP trainer: epochs x round(1/q)."""
    return int(epochs) * int(round(1 / q))


@lru_cache(maxsize=512)
def _accountant(sigma: float, batch: int, epochs: int) -> dict:
    n = len(core.data()[1].train.y)
    q = batch / n
    steps, delta = _steps(epochs, q), default_delta(n)
    grid = sorted({1, max(1, epochs // 2), epochs, min(ACC["epochs"]["max"], 2 * epochs), ACC["epochs"]["max"]})
    return core.clean({"eps": epsilon_for(sigma, q, steps, delta), "n": n, "q": q, "steps": steps, "delta": delta,
                       "batch": batch, "epochs": epochs,
                       "curve": [[ep, epsilon_for(sigma, q, _steps(ep, q), delta)] for ep in grid],
                       "sigma_for": {str(t): calibrate_sigma(t, delta, q, steps) for t in core.EPSILONS}})


@app.get("/api/accountant")
def accountant(sigma: float = Query(ACC["sigma"]["default"], ge=ACC["sigma"]["min"], le=ACC["sigma"]["max"]),
               batch: int = Query(core.CFG["dp"]["expected_batch_size"], ge=ACC["batch"]["min"], le=ACC["batch"]["max"]),
               epochs: int = Query(core.CFG["epochs"], ge=ACC["epochs"]["min"], le=ACC["epochs"]["max"])):
    return _accountant(round(sigma, 4), batch, epochs)


# ------------------------------------------------------------- mitigation
POLICIES = ("single", "equal_opportunity", "demographic_parity")


@app.get("/api/mitigation")
def mitigation(method: str, eps: float | None = None, policy: str = "single", attr: str = "sex",
               t: float | None = Query(None, ge=core.CFG["threshold_grid"]["min"], le=core.CFG["threshold_grid"]["max"])):
    e = check_model(method, eps)
    if policy not in POLICIES:
        raise HTTPException(400, f"policy must be one of {POLICIES}")
    check_attr(attr)
    return core.apply_policy(core.key(method, e), policy, attr, core.THRESH if t is None else t)


@app.get("/api/sweep")
def sweep(method: str, eps: float | None = None, attr: str = "sex"):
    e = check_model(method, eps)
    check_attr(attr)
    return core.threshold_sweep(core.key(method, e), attr)


# ---------------------------------------------------- responsible AI docs
@app.get("/api/proxy")
def proxy():
    return need("proxy.json")


@app.get("/api/provenance")
def provenance():
    pv = core.provenance()
    tr = core.read_json("training.json") or {}
    return pv | {"models": {k: {"trained_at": v.get("trained_at"), "seconds": v.get("seconds")} for k, v in tr.items()}}


@app.get("/api/scorecard")
def scorecard(eps: float):
    ev, st = need("evaluation.json"), core.read_json("study.json")
    out = []
    for m in core.METHODS:
        e = eps if core.is_private(m["id"]) else None
        k = core.key(m["id"], e)
        if k in ev:
            out.append({"key": k, "method": m["id"], "name": m["name"], "eps": e, "checks": core.checks(ev[k], st)})
    return {"thresholds": core.CFG["scorecard"], "models": out}


@app.get("/api/model_card", response_class=PlainTextResponse)
def model_card(method: str, eps: float | None = None):
    e = check_model(method, eps)
    need("evaluation.json")
    try:
        return core.model_card(core.key(method, e))
    except KeyError:
        raise HTTPException(404, "no evaluation for this model; run train_models.py")


# ---------------------------------------------------------- live training
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


class TrainIn(BaseModel):
    method: str
    eps: float = Field(1.0, gt=0)
    epochs: int


def _run(jid: str, req: TrainIn):
    job = JOBS[jid]
    try:
        priv = core.is_private(req.method)
        if priv:
            _, cfg = core.method_cfg(req.method, req.eps, req.epochs)
            dc = DPConfig(**cfg)
            n = len(core.data()[1].train.y)
            q = dc.expected_batch_size / n
            T = _steps(req.epochs, q)
            k = dc.stats_blocks()
            p = plan_noise(req.eps, default_delta(n), q, T, aux_frac=dc.aux_frac if k else 0.0, stats_blocks=k)
            with LOCK:
                job["noise"] = {"sigma_eff": p.sigma_eff, "sigma_grad": p.sigma_grad, "sample_rate": q, "delta": p.delta, "steps": T, "epsilon": p.epsilon}

        def cb(ep, model, rec):
            row = {"epoch": ep, **{k2: v for k2, v in rec.items() if k2 == "loss"}, **core.quick_eval(model)}
            if priv:
                nz = job["noise"]
                row["eps_spent"] = epsilon_for(nz["sigma_eff"], nz["sample_rate"], _steps(ep, nz["sample_rate"]), nz["delta"])
            with LOCK:
                job["epochs"].append(core.clean(row))

        model, meta = core.train(req.method, req.eps if priv else None, epochs=req.epochs, callback=cb)
        ev = core.full_eval(model)
        with LOCK:
            job["final"] = ev["summary"]
            job["final_ethics"] = {a: {k2: v for k2, v in ev["ethics"][a].items() if k2 != "groups"} for a in core.SENSITIVE_FIELDS}
            job["status"] = "done"
    except Exception as e:  # surfaced to the UI
        with LOCK:
            job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"


@app.post("/api/train")
def start_train(req: TrainIn):
    lim = core.CFG["train_epochs"]
    if req.method not in core.NAMES:
        raise HTTPException(400, "unknown method")
    if not lim["min"] <= req.epochs <= lim["max"]:
        raise HTTPException(400, f"epochs must be {lim['min']}..{lim['max']}")
    if core.is_private(req.method) and req.eps not in [float(e) for e in core.CFG["train_epsilons"]]:
        raise HTTPException(400, f"eps must be one of {core.CFG['train_epsilons']}")
    with LOCK:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise HTTPException(409, "A training job is already running.")
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"id": jid, "status": "running", "method": req.method, "name": core.NAMES[req.method], "private": core.is_private(req.method),
                     "eps": req.eps if core.is_private(req.method) else None, "epochs_total": req.epochs, "epochs": [], "started": time.time()}
    threading.Thread(target=_run, args=(jid, req), daemon=True).start()
    return {"id": jid}


@app.get("/api/train/{jid}")
def train_status(jid: str):
    with LOCK:
        if jid not in JOBS:
            raise HTTPException(404, "no such job")
        return copy.deepcopy(JOBS[jid])


# ------------------------------------------------------------ credit study
@app.get("/api/credit")
def credit():
    R = core.ROOT / "results"

    def t(name):
        p = R / name
        return core.clean(pd.read_csv(p).to_dict("records")) if p.exists() else []

    groups, levels = {}, t("final_dp_privacy_utility_fairness_comparison.csv")
    for attr, stem in (("Age", "age"), ("Income", "income"), ("Homeownership", "homeownership")):
        g = {"Standard MLP": t(f"{stem}_fairness_baseline.csv")}
        for suffix, lv in zip(("", "_noise_1_5", "_noise_2_0"), levels):
            g[f"DP σ={lv.get('Noise_Level')} (ε={lv['Epsilon']:.2f})" if isinstance(lv.get("Epsilon"), (int, float)) else f"DP σ={lv.get('Noise_Level')}"] = t(f"{stem}_fairness_dp{suffix}.csv")
        groups[attr] = g
    test_n = int(sum(r["Samples"] for r in groups["Age"]["Standard MLP"])) if groups["Age"]["Standard MLP"] else None
    return {"performance": t("model_performance_comparison.csv"), "gaps": t("model_fairness_comparison.csv"),
            "dp_levels": levels, "groups": groups, "test_size": test_n}


# --------------------------------------------------------------- LLM audit
sys.path.insert(0, str(core.ROOT / "llm_experiments"))
PROVIDER_KEYS = {"groq": "GROQ_API_KEY", "gemini": "GEMINI_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
LLM_EXPERIMENTS = [
    ("L1", "LLM as classifier on the same test rows", "accuracy, worst-group recall", "run_l1_l2.py", "l1_summary.csv"),
    ("L2", "Counterfactual identity swap", "decision flip rate, probability shift", "run_l1_l2.py", "l2_summary.csv"),
    ("L3", "Multilingual fairness", "cross-language agreement", "run_l3.py", "l3_summary.csv"),
    ("L4", "BBQ + DiscrimEval benchmarks", "bias scores", "run_l4.py", "l4_summary.csv"),
    ("L5", "Privacy leakage (PII in context, Min-K% MIA)", "leakage rate, attack AUC", "run_l5.py", "l5_summary.csv"),
    ("L6", "DP-LoRA fine-tuning with canaries", "memorisation, per-group utility", "run_l6.py", "l6_summary.csv"),
]


PROVIDER_MODELS = {"groq": "https://api.groq.com/openai/v1/models"}
NON_CHAT = ("whisper", "tts", "guard", "orpheus", "prompt-guard", "distil")


@lru_cache(maxsize=8)
def _provider_models(provider: str) -> tuple:
    """Chat models the configured key can use (queried live, cached per run)."""
    url, key = PROVIDER_MODELS.get(provider), os.environ.get(PROVIDER_KEYS.get(provider, ""), "")
    if not url or not key:
        return ()
    import json as _json
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}", "User-Agent": "fairdp-research/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            ids = [m["id"] for m in _json.loads(r.read())["data"]]
    except Exception:
        return ()
    return tuple(sorted(i for i in ids if not any(x in i for x in NON_CHAT)))


L1_COLS = ["model", "dataset", "n", "parse_failure_rate", "accuracy", "balanced_accuracy", "recall", "selection_rate",
           "intersection_worst_tpr", "intersection_tpr_gap", "intersection_n_groups_eval", "sex_tpr_gap", "race_tpr_gap"]


@app.get("/api/llm")
def llm_info():
    d = core.ROOT / "results" / "llm"
    res, l2 = [], []
    if d.exists():
        for f in sorted(d.glob("*/l1_summary.csv")):
            df = pd.read_csv(f)
            res += core.clean(df[[c for c in L1_COLS if c in df.columns]].to_dict("records"))
        for f in sorted(d.glob("*/l2_summary.csv")):
            model = pd.read_csv(f.parent / "l1_summary.csv")["model"].iloc[0] if (f.parent / "l1_summary.csv").exists() else f.parent.name
            for r in pd.read_csv(f).to_dict("records"):
                l2.append(core.clean({"model": model, **r}))
    exps = []
    for eid, name, measures, script, result in LLM_EXPERIMENTS:
        has_code = (core.ROOT / "llm_experiments" / script).exists()
        has_res = d.exists() and any(d.glob(f"*/{result}"))
        exps.append({"id": eid, "name": name, "measures": measures, "status": "results" if has_res else "code ready" if has_code else "planned"})
    return {"results": res, "l2": l2, "experiments": exps,
            "providers": {p: bool(os.environ.get(k)) for p, k in PROVIDER_KEYS.items()},
            "models": {p: list(_provider_models(p)) for p in PROVIDER_KEYS}}


MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,99}$")


class LLMIn(BaseModel):
    applicant: dict
    provider: str
    model: str


@app.post("/api/llm/query")
def llm_query(body: LLMIn):
    if body.provider not in PROVIDER_KEYS:
        raise HTTPException(400, f"provider must be one of {list(PROVIDER_KEYS)}")
    if not MODEL_ID.match(body.model) or ".." in body.model:
        raise HTTPException(400, "invalid model id")
    check_applicant(body.applicant)
    if not os.environ.get(PROVIDER_KEYS[body.provider]):
        raise HTTPException(400, f"Set {PROVIDER_KEYS[body.provider]} before starting the API.")
    import serialize as S
    from clients import LLM
    sc = core.schema()
    row = pd.Series(body.applicant)
    out = []
    try:
        llm = LLM(model=body.model, provider=body.provider, min_interval_s=1.0)
        for sex in sc["categorical"]["sex"]:
            for race in sc["categorical"]["race"]:
                r = row.copy(); r["sex"], r["race"] = sex, race
                resp = llm.chat(S.messages(S.adult_text(r), task="income"))
                lab, prob = S.parse(resp, task="income")
                out.append({"sex": sex, "race": race, "decision": lab, "prob": prob, "raw": resp[:300]})
    except Exception as e:
        raise HTTPException(502, f"LLM call failed: {type(e).__name__}: {e}")
    return {"model": body.model, "provider": body.provider, "results": out, "prompt": S.adult_text(row), "system": S.SYSTEM_INCOME}


class PromptIn(BaseModel):
    applicant: dict


@app.post("/api/llm/prompt")
def llm_prompt(body: PromptIn):
    """The exact prompt the audit sends (llm_experiments/serialize.py)."""
    import serialize as S
    check_applicant(body.applicant)
    return {"system": S.SYSTEM_INCOME, "user": S.adult_text(pd.Series(body.applicant))}
