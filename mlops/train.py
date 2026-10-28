"""Training with MLflow tracking: persistence baseline, three model families with random-search tuning (nested runs),
selection on a validation period, evaluation on the untouched test period, and a gated registry promotion."""
import json
import os
import shutil
import subprocess
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import data as D
from . import registry
from .tracking import use_experiment


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def metrics(y, p, prefix=""):
    return {f"{prefix}rmse": rmse(y, p), f"{prefix}mae": float(mean_absolute_error(y, p)), f"{prefix}r2": float(r2_score(y, p))}


def git_sha():
    if os.environ.get("GIT_SHA"):
        return os.environ["GIT_SHA"][:12]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def build(family: str, params: dict):
    if family == "ridge":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(**params))
    if family == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **params))
    if family == "hist_gradient_boosting":     # handles missing values natively
        return HistGradientBoostingRegressor(random_state=C.RANDOM_STATE, **params)
    raise ValueError(family)


def search_space(family: str, rng: np.random.Generator) -> dict:
    if family == "ridge":
        return {"alpha": float(10 ** rng.uniform(-2, 3))}
    if family == "random_forest":
        return {"n_estimators": int(rng.choice([100, 200, 300])), "max_depth": int(rng.choice([6, 10, 14, 20])),
                "min_samples_leaf": int(rng.choice([2, 5, 10, 20]))}
    return {"learning_rate": float(10 ** rng.uniform(-2, -0.5)), "max_iter": int(rng.choice([100, 200, 400])),
            "max_depth": int(rng.choice([3, 5, 8])), "l2_regularization": float(10 ** rng.uniform(-3, 1))}


def tune(family, Xtr, ytr, Xva, yva, trials, rng, parent_family_tag):
    """Random search; every trial is a nested MLflow run so the search is fully visible in the UI."""
    best = (None, np.inf)
    for t in range(trials):
        params = search_space(family, rng)
        with mlflow.start_run(run_name=f"{family}-trial-{t}", nested=True):
            mlflow.set_tags({"family": family, "stage": "tuning"})
            mlflow.log_params(params)
            m = build(family, params).fit(Xtr, ytr)
            v = rmse(yva, m.predict(Xva))
            mlflow.log_metric("val_rmse", v)
        if v < best[1]:
            best = (params, v)
    return best


def reference_artifacts(train: pd.DataFrame, out_dir):
    """Training-time distribution of each feature, saved with the model so monitoring has something to compare against."""
    sample = train[C.FEATURES].sample(min(5000, len(train)), random_state=C.RANDOM_STATE)
    sample.to_parquet(out_dir / "reference_sample.parquet", index=False)
    return sample


def run(trials: int = 8, promote: bool = True, path=None) -> dict:
    t0 = time.time()
    use_experiment(C.EXPERIMENT)
    train, test, info = D.prepare(path)
    Xtr_all, ytr_all = train[C.FEATURES], train[C.TARGET]
    Xte, yte = test[C.FEATURES], test[C.TARGET]
    inner_train, val, val_cut = D.split_by_time(train, C.VAL_FRACTION)
    Xtr, ytr, Xva, yva = inner_train[C.FEATURES], inner_train[C.TARGET], val[C.FEATURES], val[C.TARGET]
    rng = np.random.default_rng(C.RANDOM_STATE)

    persistence_test = metrics(yte, test["AQI"], "persistence_test_")
    persistence_val = rmse(yva, val["AQI"])
    results = {}
    cand = C.RUN_DIR / "candidate"
    shutil.rmtree(cand, ignore_errors=True)
    cand.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="training-pipeline") as parent:
        mlflow.set_tags({"git_sha": git_sha(), "stage": "pipeline", "data_hash": info["hash"]})
        mlflow.log_params({"features": ",".join(C.FEATURES), "trials_per_family": trials, "test_fraction": C.TEST_FRACTION,
                           "val_fraction": C.VAL_FRACTION, **{f"data_{k}": v for k, v in info.items()}})
        mlflow.log_metrics({**persistence_test, "persistence_val_rmse": persistence_val})

        for family in ["ridge", "random_forest", "hist_gradient_boosting"]:
            with mlflow.start_run(run_name=family, nested=True) as child:
                mlflow.set_tag("family", family)
                n = 1 if family == "ridge" and trials < 1 else trials
                params, val_rmse = tune(family, Xtr, ytr, Xva, yva, n, rng, family)
                model = build(family, params).fit(Xtr_all, ytr_all)
                pred = model.predict(Xte)
                m = metrics(yte, pred, "test_")
                mlflow.log_params(params)
                mlflow.log_metrics({"val_rmse": val_rmse, **m,
                                    "improvement_vs_persistence_pct": 100 * (1 - m["test_rmse"] / persistence_test["persistence_test_rmse"])})
                results[family] = {"params": params, "val_rmse": val_rmse, "model": model, **m, "run_id": child.info.run_id}

        # selection uses the validation period only; the test period is reported, never used to choose
        best_family = min(results, key=lambda f: results[f]["val_rmse"])
        best = results[best_family]
        mlflow.set_tags({"selected_family": best_family})
        mlflow.log_metrics({"selected_test_rmse": best["test_rmse"], "selected_test_r2": best["test_r2"],
                            "selected_improvement_pct": 100 * (1 - best["test_rmse"] / persistence_test["persistence_test_rmse"])})

        sig = infer_signature(Xte.head(50), best["model"].predict(Xte.head(50)))
        joblib.dump(best["model"], cand / "model.joblib")
        reference_artifacts(train, cand)
        meta = {"family": best_family, "params": best["params"], "features": C.FEATURES, "run_id": parent.info.run_id,
                "test_rmse": best["test_rmse"], "test_mae": best["test_mae"], "test_r2": best["test_r2"],
                "persistence_test_rmse": persistence_test["persistence_test_rmse"], "train_rmse_reference": best["test_rmse"],
                "data": info, "git_sha": git_sha(), "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (cand / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        mlflow.log_artifact(str(cand / "metadata.json"))
        mlflow.log_artifact(str(cand / "reference_sample.parquet"))
        logged = mlflow.sklearn.log_model(best["model"], name="model", signature=sig, input_example=Xte.head(5), serialization_format="cloudpickle")
        model_uri = logged.model_uri

        decision = {"promoted": False, "reason": "promotion disabled"}
        if promote:
            decision = registry.register_and_gate(model_uri, meta, test, persistence_test["persistence_test_rmse"])
        if decision["promoted"] or not promote:       # only an accepted model replaces the one the API falls back to
            shutil.rmtree(C.LATEST_DIR, ignore_errors=True)
            shutil.copytree(cand, C.LATEST_DIR)
        mlflow.set_tags({"promoted": str(decision["promoted"]), "gate_reason": decision["reason"][:250]})
        mlflow.log_metric("duration_s", time.time() - t0)

    summary = {"selected": best_family, "test_rmse": best["test_rmse"], "persistence_rmse": persistence_test["persistence_test_rmse"],
               "improvement_pct": 100 * (1 - best["test_rmse"] / persistence_test["persistence_test_rmse"]),
               "all": {f: {k: v for k, v in r.items() if k != "model"} for f, r in results.items()},
               "decision": decision, "data": info, "run_id": parent.info.run_id, "seconds": round(time.time() - t0, 1)}
    (C.RUN_DIR / "last_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--no-promote", action="store_true")
    a = ap.parse_args()
    s = run(a.trials, not a.no_promote)
    print(json.dumps({k: s[k] for k in ["selected", "test_rmse", "persistence_rmse", "improvement_pct", "decision", "seconds"]}, indent=2))
