# AQI MLOps: next-day AQI forecast, tracked, gated, served and monitored

MLOps layer for the AQI Big Data mini project. The Spark job produces `daily_aqi` (one row per station per day).
This project turns that table into a **versioned, tested, containerised forecasting service** with drift monitoring and automatic retraining.

**Be honest about the model.** On the real data the best model beats "tomorrow = today" by only about 3% RMSE.
The point of this project is the *process* (tracking, registry gate, CI/CD, serving, monitoring), not a strong forecast.

## Architecture

```
daily_aqi (Parquet from Spark)
      |
      v
 validate  ->  train + tune  ->  register + gate  ->  smoke test  ->  reload API
 (data.py)     (train.py,        (registry.py)        (pipeline.py)    (serve.py)
               MLflow nested
               runs)
                                      |
                                      v
 FastAPI /predict  --logs-->  predictions.jsonl   <-- /actuals (true AQI arrives next day)
                                      |
                                      v
 monitor.py: PSI + KS drift on inputs, live RMSE vs training RMSE  -> OK / WARN / ALERT (exit code 2)
                                      |
                       ALERT -> pipeline monitor-and-retrain -> new candidate -> gate
```

| Piece | File | What it does |
|---|---|---|
| Config | `mlops/config.py` | Paths, feature list, thresholds, all env-driven |
| AQI maths | `mlops/aqi_core.py` | CPCB breakpoint table, same as the Spark job (tested against Spark output) |
| Data | `mlops/data.py` | Load, fingerprint, validate, features, time split (no label leakage) |
| Training | `mlops/train.py` | Persistence baseline, random search for Ridge / RandomForest / HistGradientBoosting, MLflow nested runs |
| Registry | `mlops/registry.py` | Registers version, tags it, sets alias `champion` only if the gate passes |
| Serving | `mlops/serve.py` | FastAPI: `/predict /actuals /health /model-info /reload /metrics` |
| Monitoring | `mlops/monitor.py` | PSI, KS test, live RMSE; writes `report.json`, logs to MLflow |
| Pipeline | `mlops/pipeline.py` | Stages with fail-fast and a run log; `run`, `check`, `monitor-and-retrain` |
| Demo traffic | `mlops/simulate.py` | Replays rows to the API, optionally drifted |
| Dashboard | `mlops/dashboard.py` | Streamlit: model, experiments, registry, monitoring, try the API, pipeline runs |

**Promotion gate.** A new model becomes `champion` only if it beats the persistence baseline *and* the current champion re-evaluated on the same test period.
Otherwise it is registered with `gate=rejected` and the serving model is unchanged.

**Model selection** uses validation RMSE only (last 20% of training dates). The test period (last 20% of all dates) is touched once.

**Drift rules.** PSI on deciles: warn at 0.10, alert at 0.20. Performance alert when live RMSE exceeds 1.25 x training RMSE (needs at least 30 labelled rows).

## Run it locally (no Docker)

Needs **Python 3.12 or newer** (the pinned numpy 2.5 does not support 3.11).

```bash
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
pytest -q && ruff check .                           # 45 tests

python -m mlops.pipeline run --trials 8             # uses sample/ unless DATA_PATH is set
set DATA_PATH=path\to\daily_aqi                     # Windows; export DATA_PATH=... on Linux/Mac
```

With no `DATA_PATH` the committed sample (6 stations, 2,748 station-days) is used, so everything works on a fresh clone.

## Run it with Docker

Needs the `aqi_bda` project to have been run once, so the Docker volume `aqi-bda_work` holds `out/daily_aqi`.
If you have not, create an empty one (`docker volume create aqi-bda_work`) and the pipeline falls back to the sample.

```bash
docker compose up -d mlflow                    # tracking server       http://localhost:5000
docker compose run --rm trainer                # train, tune, register, gate
docker compose up -d api dashboard             # API :8000, dashboard  http://localhost:8502
```

## Demo script (about 5 minutes)

```bash
docker compose run --rm simulate --n 300                 # normal traffic
docker compose run --rm monitor                          # expect OK, PSI < 0.10
docker compose run --rm simulate --n 300 --drift 1.8     # pollution up 80%
docker compose run --rm monitor                          # expect ALERT on AQI, PM2_5, PM10, NO2, lags (exit code 2)
docker compose run --rm retrain                          # monitor-and-retrain: retrain, gate, reload API
```

Then show in the MLflow UI (http://localhost:5000): the experiment with nested trial runs, the model registry with versions, tags and the `champion` alias,
and the `aqi-monitoring` experiment. In the dashboard open Monitoring and Registry.

## CI/CD (GitHub Actions)

* `.github/workflows/ci.yml`: on every push and pull request run `ruff`, `pytest`, a full training run on the sample, build the Docker image and smoke-test the container.
  On `main`, once everything passes, the image is published to GitHub Container Registry (`ghcr.io`) and the tested commit is promoted to the `deploy` branch that Streamlit Community Cloud watches.
* `.github/workflows/retrain.yml`: manual or weekly retrain with the gate; the run folder is uploaded as an artifact.

## Cloud deployment (GitHub to Streamlit Community Cloud)

The dashboard for the champion model runs on **Streamlit Community Cloud**, connected to the GitHub repository. Nothing is deployed by hand:

```
git push to main  ->  GitHub Actions: ruff, pytest (incl. the cloud app), pipeline smoke test, Docker build
                  ->  all green: the tested commit is promoted to branch `deploy`
                  ->  Streamlit Community Cloud watches `deploy` and redeploys the app
```

A failing check stops the chain, so a broken commit never reaches the cloud.

**What the cloud app is.** `streamlit_app/app.py` reads a committed *bundle* (`streamlit_app/bundle/`): the champion model, a reference sample, and a snapshot of
the MLflow runs, registry and monitoring report. Pages: Model, Predict, Drift check (interactive PSI), Experiments, Registry and pipeline.
Streamlit Cloud cannot run MLflow or Docker, which is why the app reads a snapshot. Training, tracking and monitoring stay in Docker on your machine.

**One-time setup**
1. Push this folder to a GitHub repository (branch `main`). The first CI run creates the `deploy` branch.
2. Go to share.streamlit.io, sign in with GitHub, **Create app**, choose the repository, branch **`deploy`**, main file **`streamlit_app/app.py`**.
3. In *Advanced settings* choose **Python 3.12**. Streamlit installs `streamlit_app/requirements.txt` (pinned, no MLflow).
4. Deploy. Later, every push to `main` that passes CI updates the app within a minute or two.

**Putting your real model into the cloud app.** The committed bundle comes from the small sample. After training on the full data with Docker:

```bash
docker compose run --rm export-bundle      # writes champion + snapshot into ./streamlit_app/bundle
git add streamlit_app/bundle && git commit -m "Update champion bundle" && git push
```

Notes: free Streamlit apps go to sleep after a period of no visits, and wake when opened. The app has no request logging, so drift monitoring on live traffic stays with the local API (`simulate`, `monitor`).

## Rubric mapping

| Rubric item | Where it is |
|---|---|
| Cloud services (CO1, 2) | GitHub (source, Actions, container registry ghcr.io) and Streamlit Community Cloud (hosted app, auto-deployed from the `deploy` branch). See "Cloud deployment". |
| ML pipeline and CI/CD (CO2, 2) | `pipeline.py`, `.github/workflows/`, `Dockerfile`, `docker-compose.yml`, `tests/` |
| Tracking, monitoring, optimisation (CO3, 2) | MLflow runs and registry, random search, `monitor.py`, `/metrics`, dashboard |
| Documentation, deployment, presentation (CO4, 4) | This README, the MLOps report and deck (next step) |
| Viva (5) | Viva Q&A sheet (next step) |

## Known limits

* Sandbox testing used the 6-station sample only; run on the full `daily_aqi` for real numbers.
* Seasonal differences between train and live periods look like drift; PSI cannot tell seasonality from a real shift.
* Single machine, sqlite or a single MLflow container; no authentication on the API.
