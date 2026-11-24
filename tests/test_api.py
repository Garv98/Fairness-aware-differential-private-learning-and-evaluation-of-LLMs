"""API contract tests for the demo backend (skipped if models are not trained)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
if not (ROOT / "server" / "artifacts" / "evaluation.json").exists():
    pytest.skip("run server/train_models.py first", allow_module_level=True)

from fastapi.testclient import TestClient  # noqa: E402
import app as api  # noqa: E402

client = TestClient(api.app)


@pytest.fixture(scope="module")
def meta():
    r = client.get("/api/meta")
    assert r.status_code == 200
    return r.json()


def test_meta_is_config_driven(meta):
    cfg = api.core.CFG
    assert meta["epsilons"] == [float(e) for e in cfg["epsilons"]]
    assert meta["threshold"] == cfg["decision_threshold"]
    assert [m["id"] for m in meta["methods"]] == [m["id"] for m in cfg["methods"]]
    assert sum(m["ours"] for m in meta["methods"]) == 1
    assert 0 < len(meta["samples"]) <= cfg["demo"]["n_samples"]


def test_predict_counterfactual_and_recourse(meta):
    a = meta["samples"][0]["applicant"]
    r = client.post("/api/predict", json={"applicant": a, "eps": meta["epsilons"][0]}).json()
    n_ids = len(meta["schema"]["categorical"]["sex"]) * len(meta["schema"]["categorical"]["race"])
    assert len(r["models"]) == len(meta["methods"])
    assert all(len(cf["cells"]) == n_ids for cf in r["counterfactual"])
    for m in r["models"]:
        assert 0 <= m["prob"] <= 1 and m["decision"] == int(m["prob"] >= meta["threshold"])
        assert (m["recourse"] is None) == (m["decision"] == 1)


def test_mitigation_policies_change_outcomes(meta):
    ours = next(m["id"] for m in meta["methods"] if m["ours"])
    e = meta["epsilons"][0]
    res = {p: client.get(f"/api/mitigation?method={ours}&eps={e}&policy={p}&attr=sex").json() for p in ("single", "equal_opportunity", "demographic_parity")}
    assert res["equal_opportunity"]["ethics"]["sex"]["equal_opportunity_diff"] <= res["single"]["ethics"]["sex"]["equal_opportunity_diff"] + 1e-9
    assert res["demographic_parity"]["ethics"]["sex"]["demographic_parity_diff"] <= res["single"]["ethics"]["sex"]["demographic_parity_diff"] + 1e-9


@pytest.mark.parametrize("path,body", [
    ("/api/predict", {"applicant": {}, "eps": 2.5}),
    ("/api/llm/query", {"applicant": {}, "provider": "nope", "model": "x"}),
    ("/api/train", {"method": "nope", "eps": 1, "epochs": 5}),
])
def test_bad_input_is_rejected(path, body):
    assert client.post(path, json=body).status_code == 400


def test_docs_endpoints(meta):
    ours = next(m["id"] for m in meta["methods"] if m["ours"])
    card = client.get(f"/api/model_card?method={ours}&eps={meta['epsilons'][0]}")
    assert card.status_code == 200 and card.text.startswith("# Model card")
    sc = client.get(f"/api/scorecard?eps={meta['epsilons'][0]}").json()
    assert len(sc["models"]) == len(meta["methods"])
    pv = client.get("/api/provenance").json()
    assert len(pv["dataset_fingerprint"]) == 16


# ---- regressions for review findings (each returned 500 or hung before the fix)
@pytest.mark.parametrize("field,value", [("age", "abc"), ("hours-per-week", -1e9), ("age", float("1e308")), ("education", "Wizardry")])
def test_invalid_applicant_values_rejected_fast(meta, field, value):
    import time
    a = {**meta["samples"][0]["applicant"], field: value}
    t0 = time.time()
    r = client.post("/api/predict", json={"applicant": a, "eps": meta["epsilons"][0]})
    assert r.status_code == 422 and time.time() - t0 < 5


@pytest.mark.parametrize("url", [
    "/api/sweep?method=dpsgd&eps=1&attr=foo",
    "/api/accountant?sigma=1&batch=-5",
    "/api/accountant?sigma=1&epochs=100000",
    "/api/mitigation?method=dpsgd&eps=1&policy=demographic_parity&t=5",
])
def test_out_of_range_query_params_rejected(url):
    assert client.get(url).status_code in (400, 422)


def test_llm_model_id_path_traversal_rejected(meta):
    r = client.post("/api/llm/query", json={"applicant": meta["samples"][0]["applicant"], "provider": "groq", "model": "..\..\evil"})
    assert r.status_code == 400


def test_accountant_matches_trainer_step_count(meta):
    r = client.get(f"/api/accountant?sigma=1&epochs={meta['info']['epochs']}").json()
    q = meta["info"]["dp"]["expected_batch_size"] / meta["info"]["train_size"]
    assert r["steps"] == meta["info"]["epochs"] * round(1 / q)
