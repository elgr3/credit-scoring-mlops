from __future__ import annotations

import pytest

from credit_scoring import api


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"].startswith("credit-scoring-xgb:")


def test_predict(client, applicant):
    r = client.post("/predict", json=applicant)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_default"] <= 1
    assert body["decision"] in {"approve", "reject"}
    expected = "reject" if body["probability_default"] >= body["threshold"] else "approve"
    assert body["decision"] == expected
    assert body["model_version"] == client.get("/health").json()["model_version"]


def test_riskier_profile_scores_higher(client, applicant):
    safe = {
        **applicant,
        "checking_status": "no checking",
        "duration": 6,
        "credit_amount": 1000,
        "credit_history": "critical/other existing credit",
        "savings_status": ">=1000",
    }
    risky = {
        **applicant,
        "checking_status": "<0",
        "duration": 60,
        "credit_amount": 15000,
        "credit_history": "no credits/all paid",
        "savings_status": "<100",
    }
    p_safe = client.post("/predict", json=safe).json()["probability_default"]
    p_risky = client.post("/predict", json=risky).json()["probability_default"]
    assert p_risky > p_safe


@pytest.mark.parametrize(
    "patch",
    [
        {"age": 17},
        {"age": "trente"},
        {"duration": 0},
        {"credit_amount": -100},
        {"installment_commitment": 9},
        {"purpose": "crypto"},
        {"checking_status": None},
        {"unexpected_field": 1},
    ],
)
def test_predict_validation_errors(client, applicant, patch):
    r = client.post("/predict", json={**applicant, **patch})
    assert r.status_code == 422


def test_predict_missing_field(client, applicant):
    applicant.pop("age")
    assert client.post("/predict", json=applicant).status_code == 422


def test_batch(client, applicant):
    payload = {"applications": [applicant, {**applicant, "duration": 48}]}
    r = client.post("/predict/batch", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2 and len(body["predictions"]) == 2


def test_batch_empty_rejected(client):
    assert client.post("/predict/batch", json={"applications": []}).status_code == 422


def test_batch_one_invalid_rejects_all(client, applicant):
    payload = {"applications": [applicant, {**applicant, "age": 10}]}
    assert client.post("/predict/batch", json=payload).status_code == 422


def test_metrics_endpoint(client, applicant):
    client.post("/predict", json=applicant)
    client.post("/predict", json={**applicant, "age": 5})
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",path="/predict",status="200"} 1' in text
    assert 'http_requests_total{method="POST",path="/predict",status="422"} 1' in text
    assert "credit_decisions_total" in text
    assert 'model_info{version="credit-scoring-xgb:' in text


def test_model_not_loaded(client, applicant, monkeypatch):
    monkeypatch.setattr(api.service, "pipeline", None)
    assert client.get("/health").json()["status"] == "degraded"
    assert client.post("/predict", json=applicant).status_code == 503
    assert 'model_info{version="none"} 0' in client.get("/metrics").text


def test_load_missing_model(tmp_path):
    svc = api.ModelService()
    svc.load(tmp_path / "absent.joblib", tmp_path / "absent.json")
    assert not svc.loaded and svc.version is None


def test_fallback_to_sklearn_pipeline(client, applicant, monkeypatch):
    fast = client.post("/predict", json=applicant).json()["probability_default"]
    monkeypatch.setattr(api.service, "scorer", None)
    slow = client.post("/predict", json=applicant).json()["probability_default"]
    assert abs(fast - slow) < 1e-6
