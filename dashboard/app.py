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

def list_metric_runs() -> dict[str, dict[str, Path]]:
    """Discover available offline metric files for selection in the dashboard."""
    runs: dict[str, dict[str, Path]] = {}

    # Default UCI-303 run (existing behavior)
    runs["UCI-303 (default tracking)"] = {
        "baseline": BASELINE_JSON,
        "adaptive": ADAPTIVE_JSON,
    }

    # Heart 1025 evaluation export folder (contains baseline_vs_adaptive.csv etc.)
    heart1025_dir = TRACKING_DIR / "heart1025"
    if heart1025_dir.exists():
        # Use baseline_vs_adaptive.csv if present; otherwise fallback to standalone metrics files.
        runs["heart.csv (1025) - evaluation export"] = {
            "baseline_vs_adaptive": heart1025_dir / "baseline_vs_adaptive.csv",
            "false_negative": heart1025_dir / "false_negative_analysis.json",
            "stat": heart1025_dir / "statistical_comparison.json",
        }

    # Standalone metrics produced by manual training for heart.csv (1025)
    logreg_1025 = TRACKING_DIR / "baseline_metrics_heart1025_logreg.json"
    hgb_1025 = TRACKING_DIR / "baseline_metrics_heart1025_hgb.json"
    adaptive_1025 = TRACKING_DIR / "adaptive_metrics_heart1025.json"
    adaptive_1025_arf = TRACKING_DIR / "adaptive_metrics_heart1025_arf.json"
    if logreg_1025.exists() and adaptive_1025.exists():
        runs["heart.csv (1025) - baseline logreg + adaptive"] = {
            "baseline": logreg_1025,
            "adaptive": adaptive_1025,
        }
    if logreg_1025.exists() and adaptive_1025_arf.exists():
        runs["heart.csv (1025) - baseline logreg + adaptive ARF"] = {
            "baseline": logreg_1025,
            "adaptive": adaptive_1025_arf,
        }
    if hgb_1025.exists():
        runs["heart.csv (1025) - baseline HGB only"] = {
            "baseline": hgb_1025,
        }
    if hgb_1025.exists() and adaptive_1025_arf.exists():
        runs["heart.csv (1025) - baseline HGB + adaptive ARF"] = {
            "baseline": hgb_1025,
            "adaptive": adaptive_1025_arf,
        }

    return runs

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


def api_post_empty(base_url: str, path: str) -> tuple[int | None, dict | None, str | None]:
    try:
        with httpx.Client(base_url=base_url, timeout=10.0) as client:
            r = client.post(path)
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else None
        return r.status_code, data, None
    except Exception as exc:
        return None, None, str(exc)


def feature_streams_to_dataframe(streams: dict) -> pd.DataFrame:
    if not streams:
        return pd.DataFrame(columns=["feature", "n", "mean", "variance", "last"])
    rows = []
    for feat in sorted(streams.keys()):
        stats = streams[feat]
        if not isinstance(stats, dict):
            continue
        rows.append(
            {
                "feature": feat,
                "n": stats.get("n", 0),
                "mean": stats.get("mean", 0.0),
                "variance": stats.get("variance", 0.0),
                "last": stats.get("last", 0.0),
            }
        )
    return pd.DataFrame(rows)


runs = list_metric_runs()
selected_run = st.sidebar.selectbox("Offline metrics run", list(runs.keys()), index=0)
run_files = runs[selected_run]

baseline_metrics = load_json(run_files.get("baseline", BASELINE_JSON))
adaptive_path = run_files.get("adaptive")
adaptive_metrics = load_json(adaptive_path) if adaptive_path is not None else {}
adaptive_summary = adaptive_metrics.get("summary", {})
detector_summary = adaptive_summary.get("detectors", {})
best_detector = adaptive_summary.get("best_detector")

tabs = st.tabs(["Experiments (offline metrics)", "Real-time API (baseline + adaptive)"])

with tabs[0]:
    top_left, top_mid, top_right = st.columns(3)
    top_left.metric("Baseline Accuracy", f"{baseline_metrics.get('accuracy', 0):.4f}")
    if adaptive_path is None:
        top_mid.metric("Adaptive Accuracy (best)", "N/A")
        top_right.metric("Drift Events (best)", "N/A")
    elif best_detector and best_detector in detector_summary:
        top_mid.metric("Adaptive Accuracy (best)", f"{detector_summary[best_detector].get('accuracy', 0):.4f}")
        top_right.metric("Drift Events (best)", detector_summary[best_detector].get("drift_events", 0))
    else:
        top_mid.metric("Adaptive Accuracy (best)", "0.0000")
        top_right.metric("Drift Events (best)", 0)

    st.caption(f"Showing offline metrics from: {selected_run}")

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

    col_a, col_b, col_c, col_d, col_e, col_f = st.columns(6)
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

    if col_c.button("Adaptive status"):
        code, data, err = api_get(api_url, "/adaptive/status")
        if err:
            st.error(err)
        elif isinstance(data, dict):
            st.session_state["adaptive_status"] = data
            st.success(f"status {code}")
        else:
            st.warning(str(data))

    if col_d.button("Feature streams"):
        code, data, err = api_get(api_url, "/adaptive/feature-tracking")
        if err:
            st.error(err)
        elif isinstance(data, dict) and "streams" in data:
            st.session_state["feature_streams"] = data["streams"]
            st.success(f"feature-tracking {code}")
        else:
            st.warning(str(data))

    if col_e.button("List checkpoints"):
        code, data, err = api_get(api_url, "/adaptive/checkpoints")
        if err:
            st.error(err)
        elif isinstance(data, list):
            st.session_state["checkpoints_list"] = data
            st.success(f"checkpoints {code} ({len(data)} files)")
        else:
            st.warning(str(data))

    if col_f.button("Flush learn queue"):
        code, data, err = api_post_empty(api_url, "/adaptive/learn/flush")
        if err:
            st.error(err)
        else:
            st.write({"endpoint": "/adaptive/learn/flush", "status_code": code, "data": data})

    st.subheader("Adaptive live state")
    status_data = st.session_state.get("adaptive_status")
    if isinstance(status_data, dict):
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Seen (learn)", status_data.get("seen_samples", 0))
        m2.metric("Learn pending", status_data.get("learn_pending", 0))
        m3.metric("Batch size", status_data.get("commit_batch_size", 1))
        m4.metric("Drift events", status_data.get("drift_events", 0))
        m5.metric("Online Acc", f"{float(status_data.get('online_accuracy', 0)):.4f}")
        m6.metric("Online F1", f"{float(status_data.get('online_f1', 0)):.4f}")
        st.caption(
            f"Detector: {status_data.get('detector')} | model: {status_data.get('model_type')} | "
            f"checkpoint_dir: {status_data.get('checkpoint_dir') or '—'}"
        )
        streams_inline = status_data.get("feature_streams")
        if isinstance(streams_inline, dict) and streams_inline:
            st.session_state["feature_streams"] = streams_inline
    else:
        st.info("Click **Adaptive status** to load batch settings, counters, and feature snapshot.")

    streams = st.session_state.get("feature_streams")
    if isinstance(streams, dict) and streams:
        st.subheader("Per-feature stream tracking (all inputs)")
        st.caption("Running n / mean / variance / last value for every canonical feature after API traffic.")
        st.dataframe(feature_streams_to_dataframe(streams), use_container_width=True)
    else:
        st.caption("Load **Feature streams** or **Adaptive status** to populate the table.")

    ck_list = st.session_state.get("checkpoints_list")
    if isinstance(ck_list, list) and ck_list:
        st.subheader("Rollback checkpoint")
        names = [c.get("filename", "") for c in ck_list if isinstance(c, dict) and c.get("filename")]
        if names:
            pick = st.selectbox("checkpoint file", options=names, key="ck_pick")
            if st.button("Restore selected checkpoint", key="ck_restore"):
                code, data, err = api_post(api_url, "/adaptive/checkpoint/restore", {"filename": pick})
                if err:
                    st.error(err)
                else:
                    st.write({"endpoint": "/adaptive/checkpoint/restore", "status_code": code, "data": data})
                    st.session_state.pop("adaptive_status", None)
                    st.session_state.pop("feature_streams", None)

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

        _, ft_data, ft_err = api_get(api_url, "/adaptive/feature-tracking")
        if not ft_err and isinstance(ft_data, dict) and "streams" in ft_data:
            st.session_state["feature_streams"] = ft_data["streams"]
            st.subheader("Feature streams (after this predict)")
            st.dataframe(feature_streams_to_dataframe(ft_data["streams"]), use_container_width=True)

    st.divider()
    st.subheader("Online learning (Adaptive /adaptive/learn)")
    st.caption(
        "Send labeled feedback. If API `ADAPTIVE_COMMIT_BATCH_SIZE` > 1, updates queue until the batch is full; "
        "use **Flush learn queue** on the API panel above to commit early."
    )
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
            if isinstance(l_data, dict):
                st.caption(
                    f"learn_pending={l_data.get('learn_pending', '—')} | "
                    f"learn_applied_now={l_data.get('learn_applied_now', '—')}"
                )
        _, ft_data, ft_err = api_get(api_url, "/adaptive/feature-tracking")
        if not ft_err and isinstance(ft_data, dict) and "streams" in ft_data:
            st.session_state["feature_streams"] = ft_data["streams"]
            st.subheader("Feature streams (after learn)")
            st.dataframe(feature_streams_to_dataframe(ft_data["streams"]), use_container_width=True)

    st.divider()
    st.subheader("Observe (Predict + Auto-learn when label exists) /adaptive/observe")
    st.caption("Use this when labels arrive later: send predict-only first, then send the same patient again with label to auto-learn.")
    with st.form("observe_form"):
        obs_patient_id = st.text_input("patient_id (optional)", value="patient-001")
        obs_send_label = st.checkbox("Include label (learn)", value=False)
        obs_label = st.selectbox("label (if included)", options=[0, 1], index=1)
        obs_submit = st.form_submit_button("Send /adaptive/observe")

    if obs_submit:
        observe_payload: dict = {"patient": patient_payload}
        if obs_patient_id.strip():
            observe_payload["patient_id"] = obs_patient_id.strip()
        if obs_send_label:
            observe_payload["label"] = int(obs_label)
        with st.spinner("Calling /adaptive/observe ..."):
            o_code, o_data, o_err = api_post(api_url, "/adaptive/observe", observe_payload)
        if o_err:
            st.error(f"/adaptive/observe error: {o_err}")
        else:
            st.write({"endpoint": "/adaptive/observe", "status_code": o_code, "data": o_data})
            if isinstance(o_data, dict) and o_data.get("learned"):
                st.caption(
                    f"learn_pending={o_data.get('learn_pending', '—')} | "
                    f"learn_applied_now={o_data.get('learn_applied_now', '—')}"
                )
        _, ft_data, ft_err = api_get(api_url, "/adaptive/feature-tracking")
        if not ft_err and isinstance(ft_data, dict) and "streams" in ft_data:
            st.session_state["feature_streams"] = ft_data["streams"]
            st.subheader("Feature streams (after observe)")
            st.dataframe(feature_streams_to_dataframe(ft_data["streams"]), use_container_width=True)
