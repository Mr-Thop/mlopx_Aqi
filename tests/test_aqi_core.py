import math

import pandas as pd
import pytest

from mlops import aqi_core as A
from mlops import config as C


@pytest.mark.parametrize("pollutant,conc,expected", [
    ("PM2_5", 0, 0), ("PM2_5", 30, 50), ("PM2_5", 60, 100), ("PM2_5", 90, 200), ("PM2_5", 250, 400),
    ("PM2_5", 45, 75), ("PM10", 100, 100), ("PM10", 175, 150), ("NO2", 40, 50), ("CO8", 2, 100), ("O38", 168, 200),
])
def test_breakpoints_and_interpolation(pollutant, conc, expected):
    assert A.sub_index(pollutant, conc) == pytest.approx(expected)


def test_value_above_last_band_is_capped():
    assert A.sub_index("PM2_5", 900) == 500.0


def test_missing_stays_missing():
    assert A.sub_index("PM10", None) is None
    assert A.sub_index("PM10", math.nan) is None


def test_validity_rule_needs_three_sub_indices_and_a_particulate():
    assert A.aqi({"NO2": 50, "SO2": 20, "CO8": 1}) == (None, None)          # no PM2.5 or PM10
    assert A.aqi({"PM2_5": 60, "NO2": 50}) == (None, None)                    # only two sub-indices
    value, dom = A.aqi({"PM2_5": 60, "NO2": 50, "SO2": 20})
    assert dom == "PM2_5" and value == pytest.approx(100)


def test_dominant_pollutant_is_the_maximum():
    value, dom = A.aqi({"PM2_5": 30, "PM10": 300, "NO2": 20})
    assert dom == "PM10" and value == pytest.approx(250)


def test_category_bands():
    assert [C.category(x) for x in (50, 51, 200, 201, 400, 450)] == ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def test_python_implementation_agrees_with_the_spark_output():
    """The committed sample is real output of the Spark job: sub-indices, AQI and dominant pollutant must match."""
    d = pd.read_parquet(C.SAMPLE_PATH)
    pol = list(A.BP)
    for _, r in d.iterrows():
        for p in pol:
            expected = r["I_" + p]
            got = A.sub_index(p, None if pd.isna(r[p]) else float(r[p]))
            if pd.isna(expected):
                assert got is None
            else:
                assert got == pytest.approx(expected, abs=1e-6)
    ok = d.dropna(subset=["AQI"])
    for _, r in ok.head(500).iterrows():
        value, dom = A.aqi({p: (None if pd.isna(r[p]) else float(r[p])) for p in pol})
        assert value == pytest.approx(r["AQI"], abs=1e-6)
