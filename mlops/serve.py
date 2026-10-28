"""FastAPI model service. Loads the registry's 'champion' model (falls back to the last accepted local model),
logs every request for monitoring, accepts the true AQI afterwards, and exposes Prometheus metrics."""
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date as _date

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, Field

from . import config as C

PRED_TOTAL = Counter("aqi_predictions_total", "Predictions served")
LATENCY = Histogram("aqi_prediction_latency_seconds", "Prediction latency", buckets=(.005, .01, .025, .05, .1, .25, .5, 1))
PRED_VALUE = Histogram("aqi_predicted_value", "Predicted next-day AQI", buckets=(50, 100, 150, 200, 250, 300, 400, 500))
MODEL_VERSION = Gauge("aqi_model_version", "Version number of the model in use")
ERRORS = Counter("aqi_request_errors_total", "Requests rejected", ["reason"])

STATE = {"model": None, "source": None, "version": None, "meta": {}}
LOCK = threading.Lock()


def load_model():
    """Registry champion first; local copy if the registry is unreachable or empty."""
    import os

    import mlflow
    from mlflow import MlflowClient
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")          # an unreachable registry must not stall start-up
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
    try:
        mlflow.set_tracking_uri(C.TRACKING_URI)
        mv = MlflowClient().get_model_version_by_alias(C.MODEL_NAME, C.ALIAS)
        model = mlflow.sklearn.load_model(f"models:/{C.MODEL_NAME}@{C.ALIAS}")
        meta = {"run_id": mv.run_id, "tags": dict(mv.tags)}
        STATE.update(model=model, source="mlflow-registry", version=int(mv.version), meta=meta)
    except Exception:
        local = C.LATEST_DIR / "model.joblib"
        if not local.exists():
            STATE.update(model=None, source=None, version=None, meta={})
            return False
        meta = json.loads((C.LATEST_DIR / "metadata.json").read_text()) if (C.LATEST_DIR / "metadata.json").exists() else {}
        STATE.update(model=joblib.load(local), source="local-file", version=meta.get("run_id", "local")[:8], meta=meta)
    try:
        MODEL_VERSION.set(float(STATE["version"]))
    except (TypeError, ValueError):
        MODEL_VERSION.set(0)
    return True


@asynccontextmanager
async def lifespan(app):
    load_model()
    C.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="AQI next-day forecast API", version="1.0.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    station: str | None = Field(None, description="Station name, only used for logging")
    date: str | None = Field(None, description="Date of today's readings (YYYY-MM-DD); sets the month features")
    features: dict[str, float | None] = Field(..., description="Today's values; AQI is required, the rest may be missing")


class ActualRequest(BaseModel):
    request_id: str
    actual_aqi: float = Field(..., ge=0, le=500)


def _append(name, record):
    with LOCK:
        C.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
        with open(C.MONITOR_DIR / name, "a") as f:
            f.write(json.dumps(record) + "\n")


@app.get("/health")
def health():
    return {"status": "ok" if STATE["model"] is not None else "no-model", "model_source": STATE["source"], "model_version": STATE["version"]}


@app.get("/model-info")
def model_info():
    if STATE["model"] is None:
        raise HTTPException(503, "no model loaded")
    m = dict(STATE["meta"])
    t = m.get("tags") or {}
    for k in ["test_rmse", "persistence_test_rmse"]:
        if m.get(k) is None and t.get(k) is not None:
            m[k] = float(t[k])
    m.setdefault("family", t.get("family"))
    m.setdefault("trained_at", t.get("trained_at"))
    return {"name": C.MODEL_NAME, "alias": C.ALIAS, "version": STATE["version"], "source": STATE["source"], "features": C.FEATURES,
            "test_rmse": m.get("test_rmse"), "persistence_test_rmse": m.get("persistence_test_rmse"), "family": m.get("family"),
            "trained_at": m.get("trained_at"), "tags": m.get("tags")}


@app.post("/predict")
def predict(req: PredictRequest):
    if STATE["model"] is None:
        ERRORS.labels("no_model").inc()
        raise HTTPException(503, "no model loaded; run the training pipeline first")
    unknown = set(req.features) - set(C.FEATURES)
    if unknown:
        ERRORS.labels("unknown_feature").inc()
        raise HTTPException(422, f"unknown features: {sorted(unknown)}; allowed: {C.FEATURES}")
    aqi = req.features.get("AQI")
    if aqi is None or not (0 <= aqi <= 500):
        ERRORS.labels("bad_aqi").inc()
        raise HTTPException(422, "AQI is required and must be between 0 and 500")
    t0 = time.perf_counter()
    row = {f: np.nan for f in C.FEATURES}
    row.update({k: (np.nan if v is None else v) for k, v in req.features.items()})
    month = (pd.to_datetime(req.date).month if req.date else _date.today().month)
    if np.isnan(row["month_sin"]):
        row["month_sin"], row["month_cos"] = float(np.sin(2 * np.pi * month / 12)), float(np.cos(2 * np.pi * month / 12))
    pred = float(np.clip(STATE["model"].predict(pd.DataFrame([row])[C.FEATURES])[0], 0, 500))
    dt = time.perf_counter() - t0
    rid = uuid.uuid4().hex[:12]
    PRED_TOTAL.inc(); LATENCY.observe(dt); PRED_VALUE.observe(pred)
    _append("predictions.jsonl", {"request_id": rid, "ts": time.time(), "station": req.station, "date": req.date,
                                  "features": {k: (None if np.isnan(v) else float(v)) for k, v in row.items()},
                                  "prediction": pred, "model_version": str(STATE["version"])})
    return {"request_id": rid, "prediction": round(pred, 1), "category": C.category(pred),
            "model_version": STATE["version"], "model_source": STATE["source"]}


@app.post("/actuals")
def actuals(req: ActualRequest):
    """Report the true AQI of the predicted day, so live error can be monitored."""
    _append("actuals.jsonl", {"request_id": req.request_id, "ts": time.time(), "actual_aqi": req.actual_aqi})
    return {"stored": True}


@app.post("/reload")
def reload_model():
    """Pick up a newly promoted champion without restarting the service."""
    ok = load_model()
    return {"reloaded": ok, "model_version": STATE["version"], "model_source": STATE["source"]}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
