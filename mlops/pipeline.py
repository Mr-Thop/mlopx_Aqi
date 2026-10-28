"""Pipeline orchestration: validate data -> train and tune -> gate and register -> smoke-test the served model.
Each stage is timed and logged to runs/pipeline/<timestamp>.json. `monitor-and-retrain` closes the loop."""
import argparse
import json
import os
import time
import traceback

from . import config as C
from . import data as D


class Stage:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def __enter__(self):
        self.t0 = time.time()
        print(f"[stage] {self.name} ...", flush=True)
        return self

    def __exit__(self, et, ev, tb):
        rec = {"stage": self.name, "seconds": round(time.time() - self.t0, 2), "status": "failed" if et else "ok"}
        if et:
            rec["error"] = f"{et.__name__}: {ev}"
            traceback.print_exception(et, ev, tb)
        self.log.append(rec)
        print(f"[stage] {self.name} {rec['status']} in {rec['seconds']}s", flush=True)
        return False


def write_log(log, extra):
    C.PIPELINE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = C.PIPELINE_LOG_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    p.write_text(json.dumps({"stages": log, **extra}, indent=2, default=str))
    return p


def run_pipeline(trials: int, path=None) -> dict:
    from . import tracking, train
    log, extra = [], {"started": time.strftime("%Y-%m-%dT%H:%M:%S"), "data": str(path or C.resolve_data_path())}
    try:
        with Stage(log, "check-tracking-server"):
            tracking.check_server()
        with Stage(log, "validate-data"):
            df = D.load_daily(path)
            problems = D.validate(df)
            if problems:
                raise ValueError("; ".join(problems))
            extra["validated_rows"] = int(len(df))
        with Stage(log, "train-tune-register"):
            summary = train.run(trials=trials, promote=True, path=path)
            extra["result"] = {k: summary[k] for k in ["selected", "test_rmse", "persistence_rmse", "improvement_pct", "decision"]}
        with Stage(log, "smoke-test-model"):
            import joblib
            import pandas as pd
            model = joblib.load(C.LATEST_DIR / "model.joblib")
            row = pd.DataFrame([{f: (150.0 if f == "AQI" else float("nan")) for f in C.FEATURES}])
            p = float(model.predict(row)[0])
            assert 0 <= p <= 600, f"implausible prediction {p}"
            extra["smoke_prediction"] = round(p, 1)
        api = os.environ.get("API_URL")
        if api and extra["result"]["decision"]["promoted"]:
            import requests
            try:
                with Stage(log, "reload-api"):
                    extra["reload"] = requests.post(f"{api}/reload", timeout=10).json()
            except requests.RequestException as e:            # a stopped API must not fail a successful training run
                extra["reload"] = {"skipped": f"API not reachable at {api} ({type(e).__name__}); it loads the champion when it starts"}
        extra["status"] = "success"
    except Exception as e:
        extra["status"] = "failed"
        extra["error"] = str(e)
    extra["log_file"] = str(write_log(log, extra))
    return extra


def main():
    ap = argparse.ArgumentParser(prog="mlops.pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="validate, train, gate, smoke-test")
    r.add_argument("--trials", type=int, default=8)
    r.add_argument("--data", default=None)
    sub.add_parser("check", help="validate the data only")
    sub.add_parser("monitor-and-retrain", help="run monitoring; retrain when it raises an ALERT")
    a = ap.parse_args()

    if a.cmd == "check":
        problems = D.validate(D.load_daily())
        print("OK" if not problems else "\n".join(problems))
        raise SystemExit(1 if problems else 0)
    if a.cmd == "run":
        out = run_pipeline(a.trials, a.data)
        print(json.dumps({k: out.get(k) for k in ["status", "result", "smoke_prediction", "error", "log_file"]}, indent=2, default=str))
        raise SystemExit(0 if out["status"] == "success" else 1)
    if a.cmd == "monitor-and-retrain":
        from . import monitor
        rep = monitor.run()
        print(f"monitoring status: {rep['status']} ({rep.get('drifted_features')})")
        if rep["status"] == "ALERT":
            print("alert raised, retraining ...")
            out = run_pipeline(int(os.environ.get("TRIALS", "8")))
            print(json.dumps({k: out.get(k) for k in ["status", "result", "error"]}, indent=2, default=str))
            raise SystemExit(0 if out["status"] == "success" else 1)


if __name__ == "__main__":
    main()
