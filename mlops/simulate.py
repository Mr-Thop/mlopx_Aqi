"""Replays real test-period station-days through the API, optionally with a pollution shift, so monitoring can be demonstrated.
  python -m mlops.simulate --n 200                 in-distribution traffic, then the true next-day AQI
  python -m mlops.simulate --n 200 --drift 1.8     PM2.5, PM10, NO2 and AQI inputs multiplied by 1.8 (a pollution regime change)"""
import argparse

import numpy as np
import requests

from . import config as C
from . import data as D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--drift", type=float, default=1.0)
    ap.add_argument("--no-actuals", action="store_true")
    ap.add_argument("--keep-actuals", action="store_true", help="with --drift: leave the true labels unscaled, so the model's error grows (concept shift)")
    ap.add_argument("--source", choices=["train", "test"], default="train",
                    help="train = in-distribution traffic (monitoring should stay quiet); test = later period (shows genuine temporal shift)")
    a = ap.parse_args()

    train, test, _ = D.prepare()
    pool = train if a.source == "train" else test
    rows = pool.sample(min(a.n, len(pool)), random_state=int(np.random.default_rng().integers(1_000_000)))
    sent = ok = 0
    for _, r in rows.iterrows():
        feats = {f: (None if np.isnan(r[f]) else float(r[f])) for f in C.FEATURES}
        if a.drift != 1.0:
            for f in ["AQI", "PM2_5", "PM10", "NO2", "AQI_lag1", "AQI_mean7"]:
                if feats.get(f) is not None:
                    feats[f] = min(feats[f] * a.drift, 500.0) if "AQI" in f else feats[f] * a.drift
        resp = requests.post(f"{a.api}/predict", json={"station": r["station"], "date": str(r["date"].date()), "features": feats}, timeout=10)
        sent += 1
        if resp.ok:
            ok += 1
            if not a.no_actuals:
                actual = float(r[C.TARGET]) if (a.drift == 1.0 or a.keep_actuals) else min(float(r[C.TARGET]) * a.drift, 500.0)
                requests.post(f"{a.api}/actuals", json={"request_id": resp.json()["request_id"], "actual_aqi": actual}, timeout=10)
    print(f"sent {sent} requests, {ok} accepted, drift factor {a.drift}")


if __name__ == "__main__":
    main()
