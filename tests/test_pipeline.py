import json

import mlflow
import pytest

from mlops import config as C
from mlops import pipeline, train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One real training run on the committed sample, shared by the tests below (about 10 s)."""
    tmp = tmp_path_factory.mktemp("run")
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp)
    mp.setattr(C, "RUN_DIR", tmp / "runs"); mp.setattr(C, "LATEST_DIR", tmp / "runs" / "latest")
    mp.setattr(C, "MONITOR_DIR", tmp / "runs" / "monitoring"); mp.setattr(C, "PIPELINE_LOG_DIR", tmp / "runs" / "pipeline")
    mp.setattr(C, "TRACKING_URI", f"sqlite:///{(tmp / 'mlflow.db').as_posix()}")
    yield train.run(trials=1, promote=True, path=C.SAMPLE_PATH), tmp
    mp.undo()


def test_training_logs_runs_and_params_to_mlflow(trained):
    summary, _ = trained
    assert set(summary["all"]) == {"ridge", "random_forest", "hist_gradient_boosting"}
    parent = mlflow.get_run(summary["run_id"])
    assert parent.data.metrics["persistence_test_rmse"] > 0
    assert parent.data.params["trials_per_family"] == "1"
    assert parent.data.tags["selected_family"] == summary["selected"]
    children = mlflow.search_runs(experiment_ids=[parent.info.experiment_id], filter_string=f"tags.mlflow.parentRunId = '{summary['run_id']}'")
    assert len(children) == 3          # one run per model family, tuning trials are nested one level deeper
    trials = mlflow.search_runs(experiment_ids=[parent.info.experiment_id], filter_string="tags.stage = 'tuning'")
    assert len(trials) >= 3 and trials["metrics.val_rmse"].notna().all()


def test_model_selection_uses_validation_not_test(trained):
    summary, _ = trained
    chosen = min(summary["all"], key=lambda f: summary["all"][f]["val_rmse"])
    assert summary["selected"] == chosen


def test_first_model_is_promoted_to_champion(trained):
    summary, _ = trained
    assert summary["decision"]["promoted"] is True
    mv = mlflow.MlflowClient().get_model_version_by_alias(C.MODEL_NAME, C.ALIAS)
    assert int(mv.version) == summary["decision"]["version"] and mv.tags["gate"] == "promoted"


def test_local_model_and_reference_data_are_written(trained):
    _, tmp = trained
    for f in ["model.joblib", "metadata.json", "reference_sample.parquet"]:
        assert (C.LATEST_DIR / f).exists()
    assert json.loads((C.LATEST_DIR / "metadata.json").read_text())["features"] == C.FEATURES


def test_worse_challenger_is_rejected_and_latest_stays_unchanged(trained, monkeypatch):
    _, tmp = trained
    champion_before = mlflow.MlflowClient().get_model_version_by_alias(C.MODEL_NAME, C.ALIAS).version
    stamp = (C.LATEST_DIR / "metadata.json").read_text()
    from mlops import registry
    monkeypatch.setattr(registry, "register_and_gate", lambda *a, **k: {"promoted": False, "version": 99, "reason": "forced rejection"})
    train.run(trials=1, promote=True, path=C.SAMPLE_PATH)
    assert mlflow.MlflowClient().get_model_version_by_alias(C.MODEL_NAME, C.ALIAS).version == champion_before
    assert (C.LATEST_DIR / "metadata.json").read_text() == stamp


def test_pipeline_fails_fast_on_bad_data(isolated, synth, tmp_path):
    bad = synth.copy()
    bad.loc[bad.index[:10], "AQI"] = 999
    p = tmp_path / "bad.parquet"
    bad.to_parquet(p)
    out = pipeline.run_pipeline(trials=1, path=p)
    assert out["status"] == "failed" and "outside 0-500" in out["error"]
    assert [s["stage"] for s in json.loads(open(out["log_file"]).read())["stages"]] == ["check-tracking-server", "validate-data"]


def test_unreachable_tracking_server_fails_fast(isolated, monkeypatch):
    import time

    from mlops import config as C
    from mlops import pipeline
    monkeypatch.setattr(C, "TRACKING_URI", "http://127.0.0.1:9")
    t = time.time()
    out = pipeline.run_pipeline(trials=1)
    assert out["status"] == "failed" and "not reachable" in out["error"] and "docker compose" in out["error"]
    assert time.time() - t < 20


def test_stopped_api_does_not_fail_training(isolated, synth, tmp_path, monkeypatch):
    p = tmp_path / "d.parquet"
    synth.to_parquet(p)
    monkeypatch.setenv("API_URL", "http://127.0.0.1:9")
    out = pipeline.run_pipeline(trials=1, path=p)
    assert out["status"] == "success" and "skipped" in out["reload"]
