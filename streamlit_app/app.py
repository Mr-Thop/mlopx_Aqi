"""Cloud dashboard for the AQI next-day forecast (Streamlit Community Cloud).

Reads the bundle committed in streamlit_app/bundle/ (model, reference sample, snapshot of MLflow tracking, registry and monitoring).
CI deploys it only after lint, tests and the pipeline smoke test have passed.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlops import config as C  # noqa: E402
from mlops.monitor import drift_table  # noqa: E402

BUNDLE = Path(__file__).resolve().parent / "bundle"
BLUE, ORANGE, AMBER, GREY = "#3b8bea", "#eb6834", "#eda100", "#9a9a96"
st.set_page_config(page_title="AQI MLOps", layout="wide")


def style(fig, h=380):
    fig.update_layout(height=h, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(size=14), legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.6)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.25)", zeroline=False)
    return fig


@st.cache_resource(show_spinner=False)
def load_bundle():
    meta = json.loads((BUNDLE / "metadata.json").read_text()) if (BUNDLE / "metadata.json").exists() else {}
    snap = json.loads((BUNDLE / "snapshot.json").read_text()) if (BUNDLE / "snapshot.json").exists() else {}
    ref = pd.read_parquet(BUNDLE / "reference_sample.parquet") if (BUNDLE / "reference_sample.parquet").exists() else pd.DataFrame()
    model = joblib.load(BUNDLE / "model.joblib") if (BUNDLE / "model.joblib").exists() else None
    return model, meta, snap, ref


model, meta, snap, ref = load_bundle()
st.sidebar.title("AQI MLOps")
page = st.sidebar.radio("Page", ["Model", "Predict", "Drift check", "Experiments", "Registry and pipeline"])
st.sidebar.caption("Deployed by CI from GitHub after the checks pass.")
if model is None:
    st.error("No model bundle found. Run `python -m mlops.export_bundle` and commit streamlit_app/bundle/.")
    st.stop()
data_info = meta.get("data", {})
st.sidebar.caption(f"Model: {meta.get('family')}  |  trained {str(meta.get('trained_at', ''))[:10]}")

if page == "Model":
    st.title("Champion model")
    rm, pr = meta.get("test_rmse"), meta.get("persistence_test_rmse")
    c = st.columns(4)
    c[0].metric("Family", meta.get("family", "-"))
    c[1].metric("Test RMSE", round(rm, 2) if rm else "-")
    c[2].metric("Persistence RMSE", round(pr, 2) if pr else "-")
    c[3].metric("Improvement", f"{100 * (1 - rm / pr):.1f}%" if rm and pr else "-")
    if rm and pr:
        fig = go.Figure(go.Bar(x=["Persistence (tomorrow = today)", f"Champion ({meta.get('family')})"], y=[pr, rm], marker_color=[GREY, BLUE],
                               text=[f"{pr:.2f}", f"{rm:.2f}"], textposition="outside"))
        fig.update_layout(title="Test RMSE on the held-out latest 20% of dates (lower is better)", yaxis_title="RMSE (AQI points)")
        st.plotly_chart(style(fig), width="stretch")
    st.write(f"Features: {', '.join(meta.get('features', C.FEATURES))}")
    if data_info:
        st.caption(f"Trained on {data_info.get('rows', '?')} station-days from {data_info.get('stations', '?')} stations "
                   f"({data_info.get('date_min', '?')} to {data_info.get('date_max', '?')}).")
    st.caption("The model is only a few percent better than persistence. This page shows the MLOps process, not an operational forecast.")

elif page == "Predict":
    st.title("Predict tomorrow's AQI")
    med = ref.median(numeric_only=True) if len(ref) else pd.Series(dtype=float)
    c = st.columns(4)
    aqi = c[0].number_input("Today's AQI", 0, 500, int(med.get("AQI", 150)))
    pm25 = c[1].number_input("PM2.5", 0.0, 1000.0, float(round(med.get("PM2_5", 80), 1)))
    pm10 = c[2].number_input("PM10", 0.0, 1500.0, float(round(med.get("PM10", 150), 1)))
    day = c[3].date_input("Date of the readings")
    lag1 = c[0].number_input("Yesterday's AQI", 0, 500, int(med.get("AQI_lag1", aqi)))
    mean7 = c[1].number_input("Mean AQI of last 7 days", 0, 500, int(med.get("AQI_mean7", aqi)))
    with st.expander("Optional readings (leave blank if unknown)"):
        extra, cols = {}, st.columns(3)
        for i, f in enumerate(["NO2", "SO2", "CO8", "O38", "RH", "WS", "SR"]):
            txt = cols[i % 3].text_input(f, "")
            try:
                extra[f] = float(txt)
            except ValueError:
                extra[f] = np.nan
    if st.button("Predict", type="primary"):
        row = {f: np.nan for f in C.FEATURES}
        row.update({"AQI": aqi, "PM2_5": pm25, "PM10": pm10, "AQI_lag1": lag1, "AQI_mean7": mean7, **extra,
                    "month_sin": float(np.sin(2 * np.pi * day.month / 12)), "month_cos": float(np.cos(2 * np.pi * day.month / 12))})
        pred = float(np.clip(model.predict(pd.DataFrame([row])[C.FEATURES])[0], 0, 500))
        st.metric("Predicted AQI tomorrow", round(pred, 1), help=f"CPCB band: {C.category(pred)}")
        st.write(f"CPCB category: **{C.category(pred)}**. Typical error of this model is about {meta.get('test_rmse', 0):.0f} AQI points.")

elif page == "Drift check":
    st.title("Drift check")
    st.write("Compare incoming data with the training distribution using the population stability index (warning 0.10, alert 0.20). "
             "Scale the reference sample to see how a pollution change is detected.")
    c = st.columns(2)
    factor = c[0].slider("Scale pollutant levels by", 0.5, 2.5, 1.0, 0.1)
    n = c[1].slider("Rows in the simulated live batch", 100, int(max(len(ref), 200)), min(500, max(len(ref), 100)), 50)
    if len(ref) == 0:
        st.info("No reference sample in the bundle.")
    else:
        live = ref.sample(min(n, len(ref)), random_state=1).copy()
        for col in ["AQI", "PM2_5", "PM10", "NO2", "AQI_lag1", "AQI_mean7"]:
            if col in live:
                live[col] = live[col] * factor
        d = drift_table(ref, live)
        top = d.sort_values("psi", ascending=False)
        status = "ALERT" if (top.psi >= C.PSI_ALERT).any() else "WARNING" if (top.psi >= C.PSI_WARN).any() else "OK"
        st.metric("Status", status)
        fig = px.bar(top, x="feature", y="psi", color="status", color_discrete_map={"OK": BLUE, "WARNING": AMBER, "ALERT": ORANGE},
                     title="Input drift: population stability index per feature")
        fig.add_hline(y=C.PSI_WARN, line_dash="dot", annotation_text="warning 0.10")
        fig.add_hline(y=C.PSI_ALERT, line_dash="dash", annotation_text="alert 0.20")
        st.plotly_chart(style(fig), width="stretch")
    rep = snap.get("monitor_report")
    if rep:
        st.subheader("Last monitoring report from the Docker run")
        st.write(f"Status **{rep.get('status')}** on {rep.get('live_rows')} live predictions at {rep.get('generated_at')}. "
                 f"Drifted: {', '.join(rep.get('drifted_features') or []) or 'none'}.")
        perf = rep.get("performance", {})
        if perf.get("rmse") is not None:
            st.write(f"Live RMSE {perf['rmse']} against the alert limit {perf['threshold']} (training RMSE {perf['reference_rmse']}).")
        if rep.get("drift"):
            st.dataframe(pd.DataFrame(rep["drift"]), hide_index=True, width="stretch")
    if snap.get("monitor_history"):
        st.subheader("History of checks")
        st.dataframe(pd.DataFrame(snap["monitor_history"]), hide_index=True, width="stretch")

elif page == "Experiments":
    st.title("Experiment tracking")
    runs = pd.DataFrame(snap.get("runs", []))
    if runs.empty:
        st.info(snap.get("note") or "No runs in the bundle.")
    else:
        fam = runs[runs.stage == "family"].dropna(subset=["test_rmse"])
        if len(fam):
            last = fam.sort_values("start_time").groupby("family").tail(1)
            fig = px.bar(last, x="family", y="test_rmse", text="test_rmse", title="Test RMSE per model family (latest run, lower is better)")
            fig.update_traces(marker_color=BLUE, textposition="outside")
            st.plotly_chart(style(fig), width="stretch")
        trials = runs[runs.stage == "tuning"].dropna(subset=["val_rmse"])
        if len(trials):
            fig = px.strip(trials, x="family", y="val_rmse", title="Tuning trials: validation RMSE per random-search trial")
            fig.update_traces(marker=dict(color=BLUE, size=9))
            st.plotly_chart(style(fig), width="stretch")
        show = runs.drop(columns=["params"]).copy()
        show["params"] = runs["params"].map(lambda p: json.dumps(p) if p else "")
        st.dataframe(show, hide_index=True, width="stretch")
        st.caption(f"{len(runs)} MLflow runs exported at {snap.get('exported_at')}.")

else:
    st.title("Registry and pipeline")
    reg = snap.get("registry", [])
    if reg:
        st.subheader("Model registry")
        st.dataframe(pd.DataFrame([{"version": v["version"], "aliases": ", ".join(v["aliases"]), "created": v["created"],
                                    **{k: v["tags"].get(k) for k in ["family", "test_rmse", "gate", "git_sha"]}} for v in reg]).sort_values("version", ascending=False),
                     hide_index=True, width="stretch")
        st.caption("A version gets the alias 'champion' only if it beats persistence and the previous champion on the same test period.")
    else:
        st.info(snap.get("note") or "No registry data in the bundle.")
    pr = snap.get("pipeline_runs", [])
    if pr:
        st.subheader("Pipeline runs")
        for run in reversed(pr):
            with st.expander(f"{run['file']}  |  {run['status']}"):
                st.dataframe(pd.DataFrame(run["stages"]), hide_index=True, width="stretch")
                if run.get("result"):
                    st.json(run["result"])
    ls = snap.get("last_summary")
    if ls:
        st.subheader("Last training run")
        st.write(f"Selected **{ls['selected']}**: test RMSE {ls['test_rmse']:.2f} against persistence {ls['persistence_rmse']:.2f} "
                 f"({ls['improvement_pct']:.1f}% better). Gate: {ls['decision']['reason']}")
