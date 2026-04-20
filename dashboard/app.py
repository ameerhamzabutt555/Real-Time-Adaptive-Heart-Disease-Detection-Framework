from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

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
