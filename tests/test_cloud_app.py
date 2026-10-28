"""The Streamlit Community Cloud app and the bundle it reads. CI deploys only if these pass."""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from mlops import export_bundle, pipeline

APP = Path(__file__).resolve().parent.parent / "streamlit_app" / "app.py"
PAGES = ["Model", "Predict", "Drift check", "Experiments", "Registry and pipeline"]


def test_export_bundle_contains_model_and_snapshot(isolated, tmp_path):
    pipeline.run_pipeline(trials=1)
    out = tmp_path / "bundle"
    info = export_bundle.export(out)
    assert {"model.joblib", "metadata.json", "reference_sample.parquet", "snapshot.json"} <= set(info["files"])
    snap = json.loads((out / "snapshot.json").read_text())
    assert snap["registry"][0]["aliases"] == ["champion"] and len(snap["runs"]) >= 4
    assert snap["pipeline_runs"][-1]["status"] == "success"


def test_export_fails_clearly_without_a_model(isolated, tmp_path):
    with pytest.raises(FileNotFoundError, match="pipeline run"):
        export_bundle.export(tmp_path / "b")


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_error(page):
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]


def test_prediction_is_a_valid_aqi():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Predict").run()
    at.button[0].click().run()
    assert not at.exception
    assert 0 <= float(at.metric[0].value) <= 500


def test_drift_page_raises_alert_when_pollution_is_scaled():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.radio[0].set_value("Drift check").run()
    assert at.metric[0].value == "OK"
    at.slider[0].set_value(1.8).run()
    assert at.metric[0].value == "ALERT"
