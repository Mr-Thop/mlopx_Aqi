"""Export the champion model and a snapshot of tracking, registry and monitoring into streamlit_app/bundle/.

The cloud dashboard (Streamlit Community Cloud) has no MLflow server, so it reads this committed bundle instead.

    python -m mlops.export_bundle                          # uses RUN_DIR/latest and the MLflow store in MLFLOW_TRACKING_URI
    docker compose run --rm export-bundle                  # writes into ./streamlit_app/bundle on your machine
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pandas as pd

from . import config as C

DEFAULT_OUT = C.ROOT / "streamlit_app" / "bundle"


def _num(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else (round(float(v), 4) if isinstance(v, (int, float)) else v)


def tracking_snapshot() -> dict:
    """Experiment runs and registry versions from MLflow; empty with a note when the store cannot be read."""
    snap = {"runs": [], "registry": [], "note": None}
    try:
        import mlflow
        from mlflow import MlflowClient
        mlflow.set_tracking_uri(C.TRACKING_URI)
        exp = mlflow.get_experiment_by_name(C.EXPERIMENT)
        if exp:
            df = mlflow.search_runs([exp.experiment_id], order_by=["start_time ASC"], max_results=2000)
            for _, r in df.iterrows():
                snap["runs"].append({
                    "run_name": r.get("tags.mlflow.runName"), "stage": r.get("tags.stage") if isinstance(r.get("tags.stage"), str) else "family",
                    "family": r.get("tags.family") if isinstance(r.get("tags.family"), str) else None,
                    "start_time": str(r["start_time"])[:19], "val_rmse": _num(r.get("metrics.val_rmse")), "test_rmse": _num(r.get("metrics.test_rmse")),
                    "test_r2": _num(r.get("metrics.test_r2")), "promoted": r.get("tags.promoted") if isinstance(r.get("tags.promoted"), str) else None,
                    "params": {k[7:]: v for k, v in r.items() if k.startswith("params.") and isinstance(v, str) and k != "params.features"}})
        client = MlflowClient()
        aliases = {}
        for alias, ver in client.get_registered_model(C.MODEL_NAME).aliases.items():
            aliases.setdefault(str(ver), []).append(alias)
        for v in client.search_model_versions(f"name='{C.MODEL_NAME}'"):
            snap["registry"].append({"version": int(v.version), "aliases": aliases.get(str(v.version), []), "tags": dict(v.tags),
                                     "created": time.strftime("%Y-%m-%d %H:%M", time.localtime(v.creation_timestamp / 1000))})
    except Exception as e:                                           # the bundle is still useful without the MLflow part
        snap["note"] = f"MLflow store not readable when exported: {type(e).__name__}"
    return snap


def export(out: Path = DEFAULT_OUT, model_dir: Path | None = None) -> dict:
    model_dir = model_dir or C.LATEST_DIR
    if not (model_dir / "model.joblib").exists():
        raise FileNotFoundError(f"No trained model in {model_dir}. Run `python -m mlops.pipeline run` first.")
    out.mkdir(parents=True, exist_ok=True)
    for f in model_dir.iterdir():
        if f.is_file():
            shutil.copyfile(f, out / f.name)           # no chmod: bind mounts on Windows reject it
    snap = {"exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "model_name": C.MODEL_NAME, **tracking_snapshot()}
    rep = C.MONITOR_DIR / "report.json"
    snap["monitor_report"] = json.loads(rep.read_text()) if rep.exists() else None
    hist = C.MONITOR_DIR / "history.jsonl"
    snap["monitor_history"] = [json.loads(x) for x in hist.read_text().splitlines()[-50:] if x.strip()] if hist.exists() else []
    last = C.RUN_DIR / "last_summary.json"
    snap["last_summary"] = json.loads(last.read_text()) if last.exists() else None
    snap["pipeline_runs"] = []
    if C.PIPELINE_LOG_DIR.exists():
        for f in sorted(C.PIPELINE_LOG_DIR.glob("*.json"))[-10:]:
            j = json.loads(f.read_text())
            snap["pipeline_runs"].append({"file": f.name, "status": j.get("status"), "started": j.get("started"), "result": j.get("result"),
                                          "stages": [{"stage": s.get("stage"), "status": s.get("status"), "seconds": s.get("seconds")} for s in j.get("stages", [])]})
    (out / "snapshot.json").write_text(json.dumps(snap, indent=2, default=str))
    return {"out": str(out), "files": sorted(p.name for p in out.iterdir()), "runs": len(snap["runs"]), "registry_versions": len(snap["registry"])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-dir", type=Path, default=None)
    a = p.parse_args(argv)
    print(json.dumps(export(a.out, a.model_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
