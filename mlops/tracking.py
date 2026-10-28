"""Tracking helpers: make local (sqlite) runs store artifacts under RUN_DIR instead of the current directory."""
import mlflow

from . import config as C


def use_experiment(name: str):
    mlflow.set_tracking_uri(C.TRACKING_URI)
    if C.TRACKING_URI.startswith("http"):             # tracking server decides where artifacts go
        return mlflow.set_experiment(name)
    if mlflow.get_experiment_by_name(name) is None:
        art = (C.RUN_DIR / "artifacts" / name).resolve()
        art.mkdir(parents=True, exist_ok=True)
        mlflow.create_experiment(name, artifact_location=art.as_uri())
    return mlflow.set_experiment(name)


def check_server(timeout: float = 5.0):
    """Fail fast with a clear message when the MLflow tracking server cannot be reached (instead of minutes of retries)."""
    if not C.TRACKING_URI.startswith("http"):
        return
    import requests
    try:
        requests.get(C.TRACKING_URI.rstrip("/") + "/health", timeout=timeout).raise_for_status()
    except Exception as e:
        raise RuntimeError(f"MLflow tracking server {C.TRACKING_URI} is not reachable ({type(e).__name__}). "
                           "Start it with `docker compose up -d mlflow` and check `docker compose logs mlflow`.") from e
