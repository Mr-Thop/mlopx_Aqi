"""Monitoring: input drift (PSI and Kolmogorov-Smirnov) against the training distribution, and live prediction error
against the error measured at training time. Exit code 2 signals an alert, which the pipeline uses to trigger retraining."""
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from . import config as C

ORDER = {"OK": 0, "WARNING": 1, "ALERT": 2}


def psi(reference: pd.Series, live: pd.Series, bins: int = 10) -> float:
    """Population Stability Index. Bin edges come from the reference deciles; missing values form their own bin,
    so a change in how often a sensor is missing also counts as drift."""
    ref, liv = reference.dropna().to_numpy(), live.dropna().to_numpy()
    if len(ref) == 0 or len(liv) == 0:
        return 0.0
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.array([ref.min() - 1, np.median(ref), ref.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf

    def shares(x, n_missing, total):
        counts = np.histogram(x, bins=edges)[0].astype(float)
        return np.append(counts, n_missing) / max(total, 1)

    p = shares(ref, reference.isna().sum(), len(reference))
    q = shares(liv, live.isna().sum(), len(live))
    p, q = np.clip(p, 1e-4, None), np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(reference: pd.DataFrame, live: pd.DataFrame, features=None) -> pd.DataFrame:
    rows = []
    for f in features or C.DRIFT_FEATURES:
        if f not in reference or f not in live:
            continue
        v = psi(reference[f], live[f])
        a, b = reference[f].dropna(), live[f].dropna()
        ks_p = float(ks_2samp(a, b).pvalue) if len(a) > 1 and len(b) > 1 else 1.0
        status = "ALERT" if v >= C.PSI_ALERT else "WARNING" if v >= C.PSI_WARN else "OK"
        rows.append({"feature": f, "psi": round(v, 4), "ks_pvalue": round(ks_p, 4), "status": status,
                     "ref_mean": round(float(a.mean()), 3) if len(a) else None, "live_mean": round(float(b.mean()), 3) if len(b) else None,
                     "ref_missing": round(float(reference[f].isna().mean()), 3), "live_missing": round(float(live[f].isna().mean()), 3)})
    return pd.DataFrame(rows)


def performance(pred: pd.DataFrame, actual: pd.DataFrame, reference_rmse: float) -> dict:
    j = pred.merge(actual[["request_id", "actual_aqi"]], on="request_id", how="inner")
    if len(j) < C.MIN_LIVE_ROWS:
        return {"n": int(len(j)), "status": "OK", "note": f"need {C.MIN_LIVE_ROWS} labelled predictions, have {len(j)}"}
    err = j["prediction"] - j["actual_aqi"]
    rmse = float(np.sqrt((err ** 2).mean()))
    status = "ALERT" if reference_rmse and rmse > C.RMSE_ALERT_FACTOR * reference_rmse else "OK"
    return {"n": int(len(j)), "rmse": round(rmse, 3), "mae": round(float(err.abs().mean()), 3), "bias": round(float(err.mean()), 3),
            "reference_rmse": round(reference_rmse, 3), "threshold": round(C.RMSE_ALERT_FACTOR * reference_rmse, 3), "status": status}


def read_jsonl(path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()])


def build_report(reference, live_features, pred, actual, meta) -> dict:
    report = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "live_rows": int(len(live_features)),
              "model_version": meta.get("run_id", "unknown"), "drift": [], "performance": {"status": "OK", "n": 0}}
    if len(live_features) < C.MIN_LIVE_ROWS:
        report.update(status="OK", note=f"only {len(live_features)} live predictions, need {C.MIN_LIVE_ROWS} for drift checks")
        return report
    dt = drift_table(reference, live_features)
    report["drift"] = dt.to_dict("records")
    if len(actual) and len(pred):
        report["performance"] = performance(pred, actual, meta.get("train_rmse_reference", 0.0))
    worst = max([ORDER[s] for s in dt["status"]] + [ORDER[report["performance"]["status"]]])
    report["status"] = [k for k, v in ORDER.items() if v == worst][0]
    report["drifted_features"] = dt.loc[dt["status"] == "ALERT", "feature"].tolist()
    return report


def run() -> dict:
    ref_path, meta_path = C.LATEST_DIR / "reference_sample.parquet", C.LATEST_DIR / "metadata.json"
    if not ref_path.exists():
        raise SystemExit("no reference data: train a model first (python -m mlops.pipeline run)")
    reference = pd.read_parquet(ref_path)
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    pred, actual = read_jsonl(C.MONITOR_DIR / "predictions.jsonl"), read_jsonl(C.MONITOR_DIR / "actuals.jsonl")
    live = pd.DataFrame(list(pred["features"])) if len(pred) else pd.DataFrame(columns=C.FEATURES)
    report = build_report(reference, live, pred, actual, meta)
    C.MONITOR_DIR.mkdir(parents=True, exist_ok=True)
    (C.MONITOR_DIR / "report.json").write_text(json.dumps(report, indent=2))
    with open(C.MONITOR_DIR / "history.jsonl", "a") as f:
        f.write(json.dumps({k: report.get(k) for k in ["generated_at", "status", "live_rows", "drifted_features"]}
                           | {"live_rmse": report["performance"].get("rmse")}) + "\n")
    try:
        import mlflow

        from .tracking import use_experiment
        use_experiment(C.MONITOR_EXPERIMENT)
        with mlflow.start_run(run_name="monitoring-check"):
            mlflow.set_tag("status", report["status"])
            mlflow.log_metric("live_rows", report["live_rows"])
            if report["drift"]:
                mlflow.log_metric("max_psi", max(r["psi"] for r in report["drift"]))
                mlflow.log_metric("n_drifted_features", len(report["drifted_features"]))
            if report["performance"].get("rmse") is not None:
                mlflow.log_metric("live_rmse", report["performance"]["rmse"])
    except Exception as e:                      # monitoring must still work when the tracking server is down
        report["mlflow_note"] = f"not logged to MLflow: {type(e).__name__}"
    return report


if __name__ == "__main__":
    rep = run()
    print(json.dumps({k: rep.get(k) for k in ["status", "live_rows", "drifted_features", "performance", "note"]}, indent=2))
    sys.exit(2 if rep["status"] == "ALERT" else 0)
