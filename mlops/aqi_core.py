"""Pure-Python CPCB AQI sub-index, identical to the Spark job's breakpoints. Used by tests and the API's category lookup."""
import math

# (conc_lo, conc_hi, index_lo, index_hi); same table as aqi_cpcb_v2.py
BP = {
    "PM2_5": [(0, 30, 0, 50), (30, 60, 50, 100), (60, 90, 100, 200), (90, 120, 200, 300), (120, 250, 300, 400), (250, 380, 400, 500)],
    "PM10":  [(0, 50, 0, 50), (50, 100, 50, 100), (100, 250, 100, 200), (250, 350, 200, 300), (350, 430, 300, 400), (430, 510, 400, 500)],
    "NO2":   [(0, 40, 0, 50), (40, 80, 50, 100), (80, 180, 100, 200), (180, 280, 200, 300), (280, 400, 300, 400), (400, 500, 400, 500)],
    "SO2":   [(0, 40, 0, 50), (40, 80, 50, 100), (80, 380, 100, 200), (380, 800, 200, 300), (800, 1600, 300, 400), (1600, 2000, 400, 500)],
    "NH3":   [(0, 200, 0, 50), (200, 400, 50, 100), (400, 800, 100, 200), (800, 1200, 200, 300), (1200, 1800, 300, 400), (1800, 2400, 400, 500)],
    "CO8":   [(0, 1, 0, 50), (1, 2, 50, 100), (2, 10, 100, 200), (10, 17, 200, 300), (17, 34, 300, 400), (34, 50, 400, 500)],
    "O38":   [(0, 50, 0, 50), (50, 100, 50, 100), (100, 168, 100, 200), (168, 208, 200, 300), (208, 748, 300, 400), (748, 1000, 400, 500)],
}


def sub_index(pollutant: str, conc):
    """Piecewise-linear sub-index. None/NaN stays None, a value above the last band is capped at 500."""
    if conc is None or (isinstance(conc, float) and math.isnan(conc)):
        return None
    bands = BP[pollutant]
    if conc > bands[-1][1]:
        return 500.0
    for lo, hi, ilo, ihi in bands:
        if conc <= hi:
            return (ihi - ilo) / (hi - lo) * (conc - lo) + ilo
    return 500.0


def aqi(values: dict):
    """AQI of one station-day from a dict of daily pollutant values, applying the CPCB validity rule
    (at least 3 sub-indices, one of them PM2.5 or PM10). Returns (aqi, dominant) or (None, None)."""
    subs = {p: sub_index(p, values.get(p)) for p in BP}
    subs = {p: v for p, v in subs.items() if v is not None}
    if len(subs) < 3 or not ({"PM2_5", "PM10"} & subs.keys()):
        return None, None
    dom = max(subs, key=subs.get)
    return subs[dom], dom
