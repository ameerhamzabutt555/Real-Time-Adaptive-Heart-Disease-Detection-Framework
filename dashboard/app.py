from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st

TRACKING_DIR = Path("experiments/tracking")
ADAPTIVE_JSON = TRACKING_DIR / "adaptive_metrics.json"
BASELINE_JSON = TRACKING_DIR / "baseline_metrics.json"
DEFAULT_API_BASE_URL = os.getenv("ADAPTIVE_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Heart Disease RT Dashboard", layout="wide")
st.title("Real-Time Adaptive Heart Disease Dashboard")
st.caption("Live experiment monitoring panel for baseline and adaptive runs.")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_url(base_url: str, endpoint: str) -> str:
    return f"{base_url.rstrip('/')}{endpoint}"


def fetch_api_json(base_url: str, endpoint: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(_build_url(base_url, endpoint))
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                return None, "API response was not a JSON object."
            return payload, None
    except Exception as exc:  # pragma: no cover - UI path
        return None, str(exc)


def post_api_json(base_url: str, endpoint: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(_build_url(base_url, endpoint), json=payload)
            response.raise_for_status()
            response_payload = response.json()
            if not isinstance(response_payload, dict):
                return None, "API response was not a JSON object."
            return response_payload, None
    except Exception as exc:  # pragma: no cover - UI path
        return None, str(exc)


baseline_metrics = load_json(BASELINE_JSON)
adaptive_metrics = load_json(ADAPTIVE_JSON)
adaptive_summary = adaptive_metrics.get("summary", {})
detector_summary = adaptive_summary.get("detectors", {})
best_detector = adaptive_summary.get("best_detector")

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

st.divider()
st.subheader("Live Adaptive API Panel")
st.caption("Monitor and interact with online adaptive learning endpoints in real time.")

if "api_base_url" not in st.session_state:
    st.session_state["api_base_url"] = DEFAULT_API_BASE_URL
if "live_status" not in st.session_state:
    st.session_state["live_status"] = None
if "live_status_error" not in st.session_state:
    st.session_state["live_status_error"] = None
if "live_predict_result" not in st.session_state:
    st.session_state["live_predict_result"] = None
if "live_predict_error" not in st.session_state:
    st.session_state["live_predict_error"] = None
if "live_learn_result" not in st.session_state:
    st.session_state["live_learn_result"] = None
if "live_learn_error" not in st.session_state:
    st.session_state["live_learn_error"] = None

api_col, refresh_col = st.columns([3, 1])
api_base_url = api_col.text_input("Adaptive API Base URL", key="api_base_url")
if refresh_col.button("Refresh status", use_container_width=True):
    payload, error = fetch_api_json(api_base_url, "/adaptive/status")
    st.session_state["live_status"] = payload
    st.session_state["live_status_error"] = error

if st.session_state["live_status"] is None and st.session_state["live_status_error"] is None:
    payload, error = fetch_api_json(api_base_url, "/adaptive/status")
    st.session_state["live_status"] = payload
    st.session_state["live_status_error"] = error

if st.session_state["live_status_error"]:
    st.warning(f"Live status unavailable: {st.session_state['live_status_error']}")

live_status = st.session_state.get("live_status")
if isinstance(live_status, dict):
    stat_1, stat_2, stat_3, stat_4 = st.columns(4)
    stat_1.metric("Seen Samples", int(live_status.get("seen_samples", 0)))
    stat_2.metric("Drift Events", int(live_status.get("drift_events", 0)))
    stat_3.metric("Online Accuracy", f"{float(live_status.get('online_accuracy', 0.0)):.4f}")
    stat_4.metric("Online F1", f"{float(live_status.get('online_f1', 0.0)):.4f}")
    st.write(
        f"Model: `{live_status.get('model_version', 'n/a')}` | "
        f"Detector: `{live_status.get('detector', 'n/a')}` | "
        f"Threshold: `{float(live_status.get('threshold', 0.5)):.2f}`"
    )

default_patient = {
    "age": 53,
    "sex": 1,
    "resting_bp": 132.0,
    "cholesterol": 246.0,
    "fasting_blood_sugar": 0,
    "restecg": 1,
    "max_heart_rate": 151.0,
    "exercise_angina": 1,
    "chest_pain_type": 2,
    "oldpeak": 1.3,
    "slope": 1,
    "ca": 0,
    "thal": 2,
}

if "live_patient_json" not in st.session_state:
    st.session_state["live_patient_json"] = json.dumps(default_patient, indent=2)
if "live_true_label" not in st.session_state:
    st.session_state["live_true_label"] = 1
if "live_threshold_override" not in st.session_state:
    st.session_state["live_threshold_override"] = -1.0

with st.expander("Run Live Adaptive Predict / Learn", expanded=False):
    st.text_area("Patient JSON", key="live_patient_json", height=240)
    controls_left, controls_mid, controls_right = st.columns([1, 1, 2])
    controls_left.selectbox("True label", options=[0, 1], key="live_true_label")
    controls_mid.number_input(
        "Threshold override",
        min_value=-1.0,
        max_value=1.0,
        value=st.session_state["live_threshold_override"],
        step=0.01,
        key="live_threshold_override",
        help="Set -1 to ignore and use service default threshold.",
    )
    controls_right.caption("Use Predict for inference only; Learn performs online update using true label.")

    try:
        patient_payload = json.loads(st.session_state["live_patient_json"])
        if not isinstance(patient_payload, dict):
            raise ValueError("Patient JSON must be an object.")
    except Exception as exc:
        patient_payload = None
        st.error(f"Invalid patient JSON: {exc}")

    btn_predict, btn_learn = st.columns(2)
    if btn_predict.button("Adaptive Predict", use_container_width=True):
        if patient_payload is None:
            st.session_state["live_predict_result"] = None
            st.session_state["live_predict_error"] = "Invalid patient JSON."
        else:
            result, error = post_api_json(api_base_url, "/adaptive/predict", patient_payload)
            st.session_state["live_predict_result"] = result
            st.session_state["live_predict_error"] = error

    if btn_learn.button("Adaptive Learn", use_container_width=True):
        if patient_payload is None:
            st.session_state["live_learn_result"] = None
            st.session_state["live_learn_error"] = "Invalid patient JSON."
        else:
            learn_payload: dict[str, Any] = {
                "patient": patient_payload,
                "label": int(st.session_state["live_true_label"]),
            }
            if st.session_state["live_threshold_override"] >= 0.0:
                learn_payload["threshold"] = float(st.session_state["live_threshold_override"])
            result, error = post_api_json(api_base_url, "/adaptive/learn", learn_payload)
            st.session_state["live_learn_result"] = result
            st.session_state["live_learn_error"] = error
            status_payload, status_error = fetch_api_json(api_base_url, "/adaptive/status")
            st.session_state["live_status"] = status_payload
            st.session_state["live_status_error"] = status_error

if st.session_state["live_predict_error"]:
    st.warning(f"Adaptive Predict error: {st.session_state['live_predict_error']}")
if isinstance(st.session_state.get("live_predict_result"), dict):
    st.write("Adaptive Predict Response")
    st.json(st.session_state["live_predict_result"])

if st.session_state["live_learn_error"]:
    st.warning(f"Adaptive Learn error: {st.session_state['live_learn_error']}")
if isinstance(st.session_state.get("live_learn_result"), dict):
    st.write("Adaptive Learn Response")
    st.json(st.session_state["live_learn_result"])
