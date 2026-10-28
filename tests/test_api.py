import json

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from mlops import config as C
from mlops import data as D
from mlops import serve


@pytest.fixture
def client(isolated, synth, monkeypatch):
    d = D.labelled(D.make_features(synth))
    model = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(n_estimators=10, random_state=0)).fit(d[C.FEATURES], d[C.TARGET])
    C.LATEST_DIR.mkdir(parents=True)
    joblib.dump(model, C.LATEST_DIR / "model.joblib")
    (C.LATEST_DIR / "metadata.json").write_text(json.dumps({"run_id": "abc12345", "test_rmse": 30.0, "persistence_test_rmse": 32.0, "family": "random_forest"}))
    monkeypatch.setattr(C, "TRACKING_URI", "http://127.0.0.1:9")            # unreachable registry: must fall back to the local model
    monkeypatch.setattr(serve, "STATE", {"model": None, "source": None, "version": None, "meta": {}})
    with TestClient(serve.app) as c:
        yield c


def test_health_reports_the_local_fallback(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_source"] == "local-file"


def test_predict_returns_a_plausible_value_and_logs_it(client):
    r = client.post("/predict", json={"station": "Test", "date": "2024-11-05", "features": {"AQI": 250, "PM2_5": 120}})
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["prediction"] <= 500 and body["category"] in {b[1] for b in C.BANDS}
    logged = [json.loads(line) for line in (C.MONITOR_DIR / "predictions.jsonl").read_text().splitlines()]
    assert logged[-1]["request_id"] == body["request_id"] and logged[-1]["features"]["PM2_5"] == 120


def test_missing_optional_features_are_accepted(client):
    assert client.post("/predict", json={"features": {"AQI": 90}}).status_code == 200


@pytest.mark.parametrize("payload", [
    {"features": {"PM2_5": 50}},                  # AQI missing
    {"features": {"AQI": 800}},                   # out of range
    {"features": {"AQI": 100, "BOGUS": 1}},       # unknown feature
    {"features": {"AQI": "high"}},                # wrong type
])
def test_bad_requests_are_rejected(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_actuals_and_metrics(client):
    rid = client.post("/predict", json={"features": {"AQI": 150}}).json()["request_id"]
    assert client.post("/actuals", json={"request_id": rid, "actual_aqi": 160}).json()["stored"] is True
    assert "aqi_predictions_total" in client.get("/metrics").text
    assert (C.MONITOR_DIR / "actuals.jsonl").exists()


def test_no_model_gives_503(isolated, monkeypatch):
    monkeypatch.setattr(C, "TRACKING_URI", "http://127.0.0.1:9")
    monkeypatch.setattr(serve, "STATE", {"model": None, "source": None, "version": None, "meta": {}})
    with TestClient(serve.app) as c:
        assert c.get("/health").json()["status"] == "no-model"
        assert c.post("/predict", json={"features": {"AQI": 100}}).status_code == 503
