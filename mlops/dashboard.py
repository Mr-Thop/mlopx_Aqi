"""Streamlit view of the MLOps layer: experiments, registry, monitoring, live test of the API, pipeline history."""
import json
import os
import sys
from pathlib import Path

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from mlflow import MlflowClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlops import config as C  # noqa: E402

API = os.environ.get("API_URL", "http://localhost:8000")
BLUE, ORANGE = "#3b8bea", "#eb6834"
st.set_page_config(page_title="AQI MLOps", layout="wide")


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=14), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.6)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)", zeroline=False)
    return fig


mlflow.set_tracking_uri(C.TRACKING_URI)
st.sidebar.title("AQI MLOps")
page = st.sidebar.radio("Page", ["Model", "Experiments", "Registry", "Monitoring", "Try the API", "Pipeline runs"])
st.sidebar.caption(f"Tracking: {C.TRACKING_URI}")


@st.cache_data(ttl=10, show_spinner=False)
def runs():
    exp = mlflow.get_experiment_by_name(C.EXPERIMENT)
    return mlflow.search_runs([exp.experiment_id], order_by=["start_time DESC"]) if exp else pd.DataFrame()


if page == "Model":
    st.title("Production model")
    try:
        info = requests.get(f"{API}/model-info", timeout=3).json()
        c = st.columns(4)
        c[0].metric("Version", info["version"]); c[1].metric("Family", info["family"])
        c[2].metric("Test RMSE", round(info["test_rmse"], 2) if info["test_rmse"] else "-")
        c[3].metric("Persistence RMSE", round(info["persistence_test_rmse"], 2) if info["persistence_test_rmse"] else "-")
        st.caption(f"Served from {info['source']}; alias '{info['alias']}'; trained {info.get('trained_at')}")
    except Exception:
        st.warning("The API is not reachable. Start it with: docker compose -f docker-compose.mlops.yml up -d api")
    p = C.RUN_DIR / "last_summary.json"
    if p.exists():
        s = json.loads(p.read_text())
        st.subheader("Last training run")
        st.write(f"Selected **{s['selected']}**: test RMSE {s['test_rmse']:.2f} against persistence {s['persistence_rmse']:.2f} "
                 f"({s['improvement_pct']:.1f}% better). Gate decision: {s['decision']['reason']}")

elif page == "Experiments":
    st.title("Experiment tracking")
    r = runs()
    if r.empty:
        st.info("No runs yet. Run: python -m mlops.pipeline run")
    else:
        fam = r[r["tags.stage"].isna() & r["tags.family"].notna() & r["metrics.test_rmse"].notna()] if "tags.family" in r else pd.DataFrame()
        if len(fam):
            last = fam.sort_values("start_time", ascending=False).head(3)
            base = r.loc[r["metrics.persistence_test_rmse"].notna(), "metrics.persistence_test_rmse"].iloc[0]
            fig = go.Figure([go.Bar(x=["persistence"] + list(last["tags.family"]), y=[base] + list(last["metrics.test_rmse"]),
                                    marker_color=["#9a9a96", BLUE, BLUE, BLUE][:len(last) + 1])])
            fig.update_layout(title="Test RMSE of the latest run (lower is better)")
            st.plotly_chart(style(fig), width="stretch")
        cols = [c for c in ["start_time", "tags.mlflow.runName", "tags.family", "tags.stage", "metrics.val_rmse", "metrics.test_rmse",
                            "metrics.test_r2", "tags.promoted", "tags.git_sha"] if c in r]
        st.dataframe(r[cols], hide_index=True, width="stretch")
        st.caption("Tuning trials appear as nested runs (stage = tuning) in the MLflow UI on port 5000.")

elif page == "Registry":
    st.title("Model registry")
    try:
        client = MlflowClient()
        by_alias = {}
        for alias, ver in client.get_registered_model(C.MODEL_NAME).aliases.items():
            by_alias.setdefault(str(ver), []).append(alias)
        rows = [{"version": int(v.version), "aliases": ", ".join(by_alias.get(str(v.version), [])),
                 **{k: v.tags.get(k) for k in ["family", "test_rmse", "gate", "git_sha"]}}
                for v in client.search_model_versions(f"name='{C.MODEL_NAME}'")]
        st.dataframe(pd.DataFrame(rows).sort_values("version", ascending=False), hide_index=True, width="stretch")
        st.caption("A version gets the alias 'champion' only if it passes the gate: it beats persistence and the previous champion on the same test period.")
    except Exception as e:
        st.info(f"Registry not available: {type(e).__name__}")

elif page == "Monitoring":
    st.title("Monitoring")
    p = C.MONITOR_DIR / "report.json"
    if not p.exists():
        st.info("No report yet. Send traffic (python -m mlops.simulate) and run python -m mlops.monitor.")
    else:
        rep = json.loads(p.read_text())
        icon = {"OK": "OK", "WARNING": "WARNING", "ALERT": "ALERT"}[rep["status"]]
        st.metric("Status", icon, help=f"generated {rep['generated_at']}, {rep['live_rows']} live predictions")
        if rep.get("note"):
            st.info(rep["note"])
        if rep["drift"]:
            d = pd.DataFrame(rep["drift"])
            fig = px.bar(d, x="feature", y="psi", color="status", color_discrete_map={"OK": BLUE, "WARNING": "#eda100", "ALERT": ORANGE},
                         title="Input drift: population stability index per feature")
            fig.add_hline(y=C.PSI_WARN, line_dash="dot", annotation_text="warning 0.10")
            fig.add_hline(y=C.PSI_ALERT, line_dash="dash", annotation_text="alert 0.20")
            st.plotly_chart(style(fig), width="stretch")
            st.dataframe(d, hide_index=True, width="stretch")
        perf = rep["performance"]
        if perf.get("rmse") is not None:
            st.subheader("Live prediction error")
            st.write(f"{perf['n']} labelled predictions: RMSE **{perf['rmse']}** (alert above {perf['threshold']}), MAE {perf['mae']}, bias {perf['bias']}.")
        h = C.MONITOR_DIR / "history.jsonl"
        if h.exists():
            st.subheader("History of checks")
            st.dataframe(pd.read_json(h, lines=True).tail(20), hide_index=True, width="stretch")

elif page == "Try the API":
    st.title("Try the prediction API")
    c = st.columns(3)
    aqi = c[0].number_input("Today's AQI", 0, 500, 180)
    pm25 = c[1].number_input("PM2.5", 0.0, 1000.0, 90.0)
    pm10 = c[2].number_input("PM10", 0.0, 1500.0, 160.0)
    if st.button("Predict tomorrow's AQI"):
        try:
            r = requests.post(f"{API}/predict", json={"features": {"AQI": aqi, "PM2_5": pm25, "PM10": pm10}}, timeout=5)
            st.json(r.json())
        except Exception as e:
            st.error(f"API not reachable: {e}")

else:
    st.title("Pipeline runs")
    files = sorted(C.PIPELINE_LOG_DIR.glob("*.json"), reverse=True) if C.PIPELINE_LOG_DIR.exists() else []
    if not files:
        st.info("No pipeline runs yet.")
    for f in files[:10]:
        j = json.loads(f.read_text())
        with st.expander(f"{f.stem}: {j.get('status')}"):
            st.dataframe(pd.DataFrame(j["stages"]), hide_index=True, width="stretch")
            if "result" in j:
                st.json(j["result"])
