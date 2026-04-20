import pandas as pd

from heart_disease_rt.monitoring.reporting import summarize_comparison


def test_summarize_comparison_selects_best_final_accuracy() -> None:
    frame = pd.DataFrame(
        {
            "detector": ["adwin", "adwin", "ddm", "ddm"],
            "step": [1, 2, 1, 2],
            "metric_accuracy": [0.5, 0.6, 0.4, 0.8],
            "drift_flag": [0, 0, 0, 1],
        }
    )
    summary = summarize_comparison(frame)
    assert summary["best_detector"] == "ddm"
    assert summary["detectors"]["ddm"]["drift_events"] == 1
