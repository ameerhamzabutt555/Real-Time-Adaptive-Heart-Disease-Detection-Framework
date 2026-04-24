from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import streamlit as st
import httpx

TRACKING_DIR = Path("experiments/tracking")
ADAPTIVE_JSON = TRACKING_DIR / "adaptive_metrics.json"
BASELINE_JSON = TRACKING_DIR / "baseline_metrics.json"

st.set_page_config(page_title="Heart Disease RT Dashboard", layout="wide")
st.title("Real-Time Adaptive Heart Disease Dashboard")
st.caption("Live experiment monitoring panel for baseline and adaptive runs.")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

def api_get(base_url: str, path: str) -> tuple[int | None, dict | None, str | None]:
    try:
        with httpx.Client(base_url=base_url, timeout=5.0) as client:
            r = client.get(path)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        return r.status_code, data, None
    except Exception as exc:  # streamlit UI helper
        return None, None, str(exc)


def api_post(base_url: str, path: str, payload: dict) -> tuple[int | None, dict | None, str | None]:
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            r = client.post(path, json=payload)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        return r.status_code, data, None
    except Exception as exc:  # streamlit UI helper
        return None, None, str(exc)


baseline_metrics = load_json(BASELINE_JSON)
adaptive_metrics = load_json(ADAPTIVE_JSON)
adaptive_summary = adaptive_metrics.get("summary", {})
detector_summary = adaptive_summary.get("detectors", {})
best_detector = adaptive_summary.get("best_detector")

tabs = st.tabs(["Experiments (offline metrics)", "Real-time API (baseline + adaptive)"])

with tabs[0]:
    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Baseline Accuracy", f"{baseline_metrics.get('accuracy', 0):.4f}")
    if best_detector and best_detector in detector_summary:
        top_mid.metric("Adaptive Accuracy (best)", f"{detector_summary[best_detector].get('accuracy', 0):.4f}")
        top_right.metric("Drift Events (best)", detector_summary[best_detector].get("drift_events", 0))
    else:
        top_mid.metric("Adaptive Accuracy (best)", "0.0000")
        top_right.metric("Drift Events (best)", 0)

    if detector_summary:
        st.subheader("Detector Comparison Summary")
        detector_rows = []
        for name, values in detector_summary.items():
            detector_rows.append(
                {
                    "detector": name,
                    "accuracy": values.get("accuracy", 0.0),
                    "f1": values.get("f1", 0.0),
                    "drift_events": values.get("drift_events", 0),
                }
            )
        st.dataframe(pd.DataFrame(detector_rows), use_container_width=True)
    else:
        st.info("Run `make run-adaptive` to generate detector comparison metrics.")

    st.subheader("Adaptive Run Progress by Detector")
    progress_files = adaptive_metrics.get("progress_files", {})
    if progress_files:
        detector_options = sorted(progress_files.keys())
        selected_detector = st.selectbox("Select detector", detector_options, index=0)
        progress_csv = Path(progress_files[selected_detector])
        if progress_csv.exists():
            progress_df = pd.read_csv(progress_csv)
            st.line_chart(progress_df.set_index("step")[["metric_accuracy", "metric_f1"]], use_container_width=True)
            drift_points = progress_df[progress_df["drift_flag"] == 1]
            if not drift_points.empty:
                st.write(f"Detected drift points: {drift_points['step'].tolist()}")
        else:
            st.warning(f"Progress CSV not found for detector: {selected_detector}")
    else:
        st.warning("Progress CSVs not found yet. Run adaptive loop first.")

    st.subheader("Generated Plot Artifacts")
    plots = adaptive_metrics.get("plots", {})
    plot_columns = st.columns(2)
    for idx, (key, plot_path) in enumerate(plots.items()):
        target_col = plot_columns[idx % 2]
        path_obj = Path(plot_path)
        if path_obj.exists():
            target_col.image(str(path_obj), caption=key.replace("_", " ").title())

with tabs[1]:
    st.subheader("API connection")
    default_api_url = os.getenv("HEART_API_URL", "http://localhost:8000")
    api_url = st.text_input("API base URL", value=default_api_url, help="Example: http://localhost:8001")

    col_a, col_b, col_c = st.columns(3)
    if col_a.button("Check /health"):
        code, data, err = api_get(api_url, "/health")
        if err:
            st.error(err)
        else:
            st.write({"status_code": code, "data": data})

    if col_b.button("Get /model-info"):
        code, data, err = api_get(api_url, "/model-info")
        if err:
            st.error(err)
        else:
            st.write({"status_code": code, "data": data})

    if col_c.button("Get /adaptive/status"):
        code, data, err = api_get(api_url, "/adaptive/status")
        if err:
            st.error(err)
        else:
            st.write({"status_code": code, "data": data})

    st.divider()
    st.subheader("Try a prediction (Baseline vs Adaptive)")

    with st.form("predict_form"):
        left, right = st.columns(2)
        age = left.number_input("age", min_value=1, max_value=120, value=53, step=1)
        sex = left.selectbox("sex (0=female, 1=male)", options=[0, 1], index=1)
        resting_bp = left.number_input("resting_bp", min_value=1.0, value=132.0)
        cholesterol = left.number_input("cholesterol", min_value=1.0, value=246.0)
        max_heart_rate = left.number_input("max_heart_rate", min_value=1.0, value=151.0)

        fasting_blood_sugar = right.selectbox("fasting_blood_sugar", options=[0, 1], index=0)
        restecg = right.selectbox("restecg", options=[0, 1, 2], index=1)
        exercise_angina = right.selectbox("exercise_angina", options=[0, 1], index=1)
        chest_pain_type = right.selectbox("chest_pain_type", options=[0, 1, 2, 3], index=2)
        oldpeak = right.number_input("oldpeak", min_value=0.0, value=1.3)
        slope = right.selectbox("slope", options=[0, 1, 2], index=1)
        ca = right.selectbox("ca", options=[0, 1, 2, 3, 4], index=0)
        thal = right.selectbox("thal", options=[0, 1, 2, 3], index=2)

        submit = st.form_submit_button("Run baseline + adaptive predict")

    patient_payload = {
        "age": int(age),
        "sex": int(sex),
        "resting_bp": float(resting_bp),
        "cholesterol": float(cholesterol),
        "max_heart_rate": float(max_heart_rate),
        "fasting_blood_sugar": int(fasting_blood_sugar),
        "restecg": int(restecg),
        "exercise_angina": int(exercise_angina),
        "chest_pain_type": int(chest_pain_type),
        "oldpeak": float(oldpeak),
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal),
    }

    if submit:
        with st.spinner("Calling /predict ..."):
            b_code, b_data, b_err = api_post(api_url, "/predict", patient_payload)
        if b_err:
            st.error(f"/predict error: {b_err}")
        else:
            st.write({"endpoint": "/predict", "status_code": b_code, "data": b_data})

        with st.spinner("Calling /adaptive/predict ..."):
            a_code, a_data, a_err = api_post(api_url, "/adaptive/predict", patient_payload)
        if a_err:
            st.error(f"/adaptive/predict error: {a_err}")
        else:
            st.write({"endpoint": "/adaptive/predict", "status_code": a_code, "data": a_data})

    st.divider()
    st.subheader("Online learning (Adaptive /adaptive/learn)")
    st.caption("Send labeled feedback to update the adaptive model in-memory.")
    with st.form("learn_form"):
        label = st.selectbox("true label", options=[0, 1], index=1)
        threshold_override = st.number_input("threshold override (optional)", min_value=0.0, max_value=1.0, value=0.5)
        use_override = st.checkbox("Send threshold override", value=False)
        learn_submit = st.form_submit_button("Send /adaptive/learn")

    if learn_submit:
        learn_payload: dict = {"patient": patient_payload, "label": int(label)}
        if use_override:
            learn_payload["threshold"] = float(threshold_override)
        with st.spinner("Calling /adaptive/learn ..."):
            l_code, l_data, l_err = api_post(api_url, "/adaptive/learn", learn_payload)
        if l_err:
            st.error(f"/adaptive/learn error: {l_err}")
        else:
            st.write({"endpoint": "/adaptive/learn", "status_code": l_code, "data": l_data})
