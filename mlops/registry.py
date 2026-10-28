"""Model registry gate. A new model becomes the 'champion' alias only if it
  1. beats the persistence baseline on the test period, and
  2. is not worse than the current champion evaluated on the SAME test period."""
import mlflow
from mlflow import MlflowClient

from . import config as C


def current_champion():
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(C.MODEL_NAME, C.ALIAS)
    except Exception:
        return None
    return mv


def register_and_gate(model_uri: str, meta: dict, test, persistence_rmse: float) -> dict:
    from .train import rmse
    client = MlflowClient()
    new = mlflow.register_model(model_uri, C.MODEL_NAME)
    client.set_model_version_tag(C.MODEL_NAME, new.version, "test_rmse", f"{meta['test_rmse']:.4f}")
    client.set_model_version_tag(C.MODEL_NAME, new.version, "family", meta["family"])
    client.set_model_version_tag(C.MODEL_NAME, new.version, "git_sha", meta["git_sha"])
    client.set_model_version_tag(C.MODEL_NAME, new.version, "persistence_test_rmse", f"{persistence_rmse:.4f}")
    client.set_model_version_tag(C.MODEL_NAME, new.version, "trained_at", meta["trained_at"])

    if meta["test_rmse"] >= persistence_rmse:
        reason = f"rejected: test RMSE {meta['test_rmse']:.2f} does not beat persistence {persistence_rmse:.2f}"
        client.set_model_version_tag(C.MODEL_NAME, new.version, "gate", "rejected")
        return {"promoted": False, "version": new.version, "reason": reason}

    champ = current_champion()
    if champ is not None and str(champ.version) != str(new.version):
        old = mlflow.sklearn.load_model(f"models:/{C.MODEL_NAME}@{C.ALIAS}")
        old_rmse = rmse(test[C.TARGET], old.predict(test[C.FEATURES]))
        if meta["test_rmse"] > old_rmse:
            reason = f"rejected: new RMSE {meta['test_rmse']:.2f} worse than champion v{champ.version} {old_rmse:.2f} on the same test period"
            client.set_model_version_tag(C.MODEL_NAME, new.version, "gate", "rejected")
            return {"promoted": False, "version": new.version, "reason": reason}
        reason = f"promoted: new RMSE {meta['test_rmse']:.2f} <= champion v{champ.version} {old_rmse:.2f}"
    else:
        reason = f"promoted: first champion, RMSE {meta['test_rmse']:.2f} beats persistence {persistence_rmse:.2f}"
    client.set_registered_model_alias(C.MODEL_NAME, C.ALIAS, new.version)
    client.set_model_version_tag(C.MODEL_NAME, new.version, "gate", "promoted")
    return {"promoted": True, "version": new.version, "reason": reason}
