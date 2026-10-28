import numpy as np
import pandas as pd
import pytest

from mlops import config as C


@pytest.fixture
def synth():
    """Two years of daily data for 4 stations with seasonality, noise and gaps."""
    rng = np.random.default_rng(0)
    frames = []
    for s in range(4):
        dates = pd.date_range("2023-01-01", periods=730, freq="D")
        base = 120 + 60 * np.cos(2 * np.pi * (dates.dayofyear - 20) / 365) + 15 * s
        aqi = np.clip(base + rng.normal(0, 20, len(dates)), 15, 480)
        df = pd.DataFrame({"site": f"S{s}", "station": f"Station {s}", "city": f"City {s}", "date": dates, "AQI": aqi,
                           "PM2_5": aqi * 0.5 + rng.normal(0, 5, len(dates)), "PM10": aqi * 0.9 + rng.normal(0, 8, len(dates)),
                           "NO2": rng.uniform(10, 90, len(dates)), "SO2": rng.uniform(2, 30, len(dates)),
                           "CO8": rng.uniform(0.3, 3, len(dates)), "O38": rng.uniform(10, 120, len(dates)),
                           "RH": rng.uniform(20, 95, len(dates)), "WS": rng.uniform(0.3, 4, len(dates)), "SR": rng.uniform(20, 300, len(dates))})
        df = df.drop(df.sample(frac=0.08, random_state=s).index)       # missing days
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Point every output (models, logs, MLflow store) at a temporary folder."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(C, "RUN_DIR", tmp_path / "runs")
    monkeypatch.setattr(C, "LATEST_DIR", tmp_path / "runs" / "latest")
    monkeypatch.setattr(C, "MONITOR_DIR", tmp_path / "runs" / "monitoring")
    monkeypatch.setattr(C, "PIPELINE_LOG_DIR", tmp_path / "runs" / "pipeline")
    monkeypatch.setattr(C, "TRACKING_URI", f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}")
    return tmp_path
