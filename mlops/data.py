"""Loading, validation, feature engineering and the time-based split. No future information may reach the training set."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

REQUIRED = ["site", "station", "date", "AQI"] + [c for c in C.BASE_FEATURES if c != "AQI"]


def load_daily(path=None) -> pd.DataFrame:
    path = Path(path) if path else C.resolve_data_path()
    df = pd.read_parquet(path)
    if "city" in df.columns:
        df["city"] = df["city"].astype(str)
    df["date"] = pd.to_datetime(df["date"])
    return df


def fingerprint(path=None) -> dict:
    """Identifies the data a model was trained on (row count, date range, size hash of the files)."""
    path = Path(path) if path else C.resolve_data_path()
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.name}:{f.stat().st_size}".encode())
    return {"path": str(path), "files": len(files), "hash": h.hexdigest()[:12]}


def validate(df: pd.DataFrame, min_rows: int = 500) -> list[str]:
    """Data-quality gate: returns a list of problems (empty list = pass)."""
    problems = []
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows, need at least {min_rows}")
    if df["AQI"].isna().mean() > 0.05:
        problems.append(f"AQI null share {df['AQI'].isna().mean():.1%} above 5%")
    bad = ((df["AQI"] < 0) | (df["AQI"] > 500)).sum()
    if bad:
        problems.append(f"{bad} AQI values outside 0-500")
    if df.duplicated(["site", "date"]).any():
        problems.append("duplicate (site, date) rows")
    if df["date"].isna().any():
        problems.append("unparseable dates")
    if df["site"].nunique() < 2:
        problems.append("fewer than 2 stations")
    return problems


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per station-day with today's values, short history and the NEXT day's AQI as the label.
    The label and the lag are only set when the neighbouring calendar day exists, so gaps never create false pairs."""
    d = df.dropna(subset=["AQI"]).sort_values(["site", "date"]).reset_index(drop=True).copy()
    g = d.groupby("site", sort=False)
    gap_next = (g["date"].shift(-1) - d["date"]).dt.days
    d[C.TARGET] = np.where(gap_next == 1, g["AQI"].shift(-1), np.nan)
    gap_prev = (d["date"] - g["date"].shift(1)).dt.days
    d["AQI_lag1"] = np.where(gap_prev == 1, g["AQI"].shift(1), np.nan)
    # mean AQI of the last 7 calendar days including today (at least 3 observations)
    roll = (d.set_index("date").groupby("site", sort=False)["AQI"].rolling("7D", min_periods=3).mean())
    d["AQI_mean7"] = roll.to_numpy()
    month = d["date"].dt.month
    d["month_sin"] = np.sin(2 * np.pi * month / 12)
    d["month_cos"] = np.cos(2 * np.pi * month / 12)
    for c in C.BASE_FEATURES:
        if c not in d.columns:
            d[c] = np.nan
    return d


def labelled(d: pd.DataFrame) -> pd.DataFrame:
    return d.dropna(subset=[C.TARGET]).reset_index(drop=True)


def split_by_time(d: pd.DataFrame, frac: float):
    """Earliest (1 - frac) of the DATES for training, latest frac for testing."""
    cutoff = d["date"].quantile(1 - frac)
    return d[d["date"] <= cutoff].reset_index(drop=True), d[d["date"] > cutoff].reset_index(drop=True), cutoff


def prepare(path=None):
    """Full preparation: returns (train, test, info). Raises ValueError if validation fails."""
    df = load_daily(path)
    problems = validate(df)
    if problems:
        raise ValueError("data validation failed: " + "; ".join(problems))
    d = labelled(make_features(df))
    train, test, cutoff = split_by_time(d, C.TEST_FRACTION)
    info = {"rows": int(len(d)), "train_rows": int(len(train)), "test_rows": int(len(test)),
            "stations": int(d["site"].nunique()), "train_end": str(cutoff.date()),
            "date_min": str(d["date"].min().date()), "date_max": str(d["date"].max().date()), **fingerprint(path)}
    return train, test, info
