import numpy as np
import pandas as pd

from mlops import config as C
from mlops import data as D


def _mini():
    dates = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-05", "2024-01-06"])   # 4 Jan is missing
    return pd.DataFrame({"site": "A", "station": "x", "date": dates, "AQI": [100.0, 110.0, 120.0, 150.0, 160.0]})


def test_label_is_next_calendar_day_only():
    d = D.make_features(_mini())
    by = dict(zip(d["date"].dt.day, d[C.TARGET]))
    assert by[1] == 110 and by[2] == 120
    assert np.isnan(by[3])            # next row is 5 Jan, not 4 Jan: no false pair across the gap
    assert by[5] == 160


def test_lag_feature_skips_gaps():
    d = D.make_features(_mini())
    lag = dict(zip(d["date"].dt.day, d["AQI_lag1"]))
    assert np.isnan(lag[1]) and lag[2] == 100 and lag[3] == 110
    assert np.isnan(lag[5])


def test_features_do_not_cross_stations():
    a, b = _mini(), _mini().assign(site="B", AQI=[300.0] * 5)
    d = D.make_features(pd.concat([a, b]))
    assert d[(d.site == "A") & (d.date.dt.day == 6)][C.TARGET].isna().all()      # last day of A must not borrow B's first day


def test_split_is_strictly_by_time(synth):
    d = D.labelled(D.make_features(synth))
    train, test, cutoff = D.split_by_time(d, 0.2)
    assert train["date"].max() <= cutoff < test["date"].min()
    assert 0.15 < len(test) / len(d) < 0.25


def test_validation_passes_on_good_data(synth):
    assert D.validate(synth) == []


def test_validation_catches_problems(synth):
    bad = synth.copy()
    bad.loc[bad.index[:5], "AQI"] = 900
    assert any("outside 0-500" in p for p in D.validate(bad))
    dup = pd.concat([synth, synth.head(3)])
    assert any("duplicate" in p for p in D.validate(dup))
    assert any("only" in p for p in D.validate(synth.head(100)))
    assert any("missing columns" in p for p in D.validate(synth.drop(columns=["PM10"])))


def test_prepare_on_the_committed_sample():
    train, test, info = D.prepare(C.SAMPLE_PATH)
    assert info["train_rows"] > 500 and info["test_rows"] > 100
    assert train["date"].max() < test["date"].min()
    assert train[C.TARGET].notna().all()
