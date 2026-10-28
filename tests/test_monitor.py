import numpy as np
import pandas as pd

from mlops import config as C
from mlops import monitor as M


def _frame(rng, n, shift=1.0, missing=0.0):
    x = rng.normal(100, 25, n) * shift
    x = np.where(rng.random(n) < missing, np.nan, x)
    return pd.Series(x)


def test_psi_is_near_zero_for_the_same_distribution():
    rng = np.random.default_rng(1)
    assert M.psi(_frame(rng, 5000), _frame(rng, 500)) < 0.05


def test_psi_flags_a_mean_shift():
    rng = np.random.default_rng(1)
    assert M.psi(_frame(rng, 5000), _frame(rng, 500, shift=1.6)) > C.PSI_ALERT


def test_psi_flags_a_sensor_that_starts_going_missing():
    rng = np.random.default_rng(1)
    assert M.psi(_frame(rng, 5000), _frame(rng, 500, missing=0.4)) > C.PSI_ALERT


def _ref_live(shift):
    rng = np.random.default_rng(2)
    ref = pd.DataFrame({f: rng.normal(100, 25, 3000) for f in C.DRIFT_FEATURES})
    live = pd.DataFrame({f: rng.normal(100, 25, 300) for f in C.DRIFT_FEATURES})
    live["PM2_5"] = live["PM2_5"] * shift
    return ref, live


def test_report_is_ok_without_drift_and_names_the_drifted_feature_otherwise():
    ref, live = _ref_live(1.0)
    assert M.build_report(ref, live, pd.DataFrame(), pd.DataFrame(), {})["status"] == "OK"
    ref, live = _ref_live(1.8)
    rep = M.build_report(ref, live, pd.DataFrame(), pd.DataFrame(), {})
    assert rep["status"] == "ALERT" and rep["drifted_features"] == ["PM2_5"]


def test_report_waits_for_enough_live_rows():
    ref, live = _ref_live(1.8)
    rep = M.build_report(ref, live.head(5), pd.DataFrame(), pd.DataFrame(), {})
    assert rep["status"] == "OK" and "need" in rep["note"]


def test_performance_alert_when_error_exceeds_the_threshold():
    n = 100
    pred = pd.DataFrame({"request_id": [str(i) for i in range(n)], "prediction": np.full(n, 100.0)})
    good = pd.DataFrame({"request_id": pred.request_id, "actual_aqi": np.full(n, 105.0)})
    bad = pd.DataFrame({"request_id": pred.request_id, "actual_aqi": np.full(n, 220.0)})
    assert M.performance(pred, good, reference_rmse=20.0)["status"] == "OK"
    assert M.performance(pred, bad, reference_rmse=20.0)["status"] == "ALERT"
