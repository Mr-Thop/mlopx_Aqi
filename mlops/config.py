"""Central configuration. Every value can be overridden with an environment variable (used by Docker and CI)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("MLOPS_HOME", Path(__file__).resolve().parent.parent))

DATA_PATH = os.environ.get("DATA_PATH", "")                       # Parquet file or folder with station-day AQI
SAMPLE_PATH = ROOT / "sample" / "daily_aqi_sample.parquet"          # tiny committed sample (CI and fallback)
RUN_DIR = Path(os.environ.get("RUN_DIR", ROOT / "runs"))            # local outputs: latest model, monitoring logs, pipeline logs
TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{(RUN_DIR / 'mlflow.db').as_posix()}")
EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "aqi-next-day-forecast")
MONITOR_EXPERIMENT = os.environ.get("MLFLOW_MONITOR_EXPERIMENT", "aqi-monitoring")
MODEL_NAME = os.environ.get("MODEL_NAME", "aqi-next-day")
ALIAS = os.environ.get("MODEL_ALIAS", "champion")

LATEST_DIR = RUN_DIR / "latest"                                     # always written, so the API works without a registry
MONITOR_DIR = RUN_DIR / "monitoring"
PIPELINE_LOG_DIR = RUN_DIR / "pipeline"

# features used by the model (all optional at prediction time except AQI)
BASE_FEATURES = ["AQI", "PM2_5", "PM10", "NO2", "SO2", "CO8", "O38", "RH", "WS", "SR"]
DERIVED_FEATURES = ["AQI_lag1", "AQI_mean7", "month_sin", "month_cos"]
FEATURES = BASE_FEATURES + DERIVED_FEATURES
TARGET = "AQI_next"
DRIFT_FEATURES = [f for f in FEATURES if not f.startswith("month_")]   # calendar features change deterministically, not by drift

TEST_FRACTION = 0.2
VAL_FRACTION = 0.2          # share of the training dates held out for model selection and tuning
RANDOM_STATE = 42

# monitoring thresholds
PSI_WARN, PSI_ALERT = 0.10, 0.20
MIN_LIVE_ROWS = 30
RMSE_ALERT_FACTOR = 1.25    # alert when live RMSE exceeds 1.25 x the RMSE measured at training time

BANDS = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"), (300, "Poor"), (400, "Very Poor"), (500, "Severe")]


def category(aqi: float) -> str:
    for hi, name in BANDS:
        if aqi <= hi:
            return name
    return "Severe"


def resolve_data_path() -> Path:
    """DATA_PATH if it exists and has data, otherwise the committed sample (so CI and demos always run)."""
    if DATA_PATH and Path(DATA_PATH).exists():
        p = Path(DATA_PATH)
        if p.is_file() or any(p.rglob("*.parquet")):
            return p
    return SAMPLE_PATH
