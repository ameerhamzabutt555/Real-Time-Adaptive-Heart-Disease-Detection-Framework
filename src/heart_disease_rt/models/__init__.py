"""Model package (static and adaptive)."""

from heart_disease_rt.models.adaptive import (
    SUPPORTED_DETECTORS,
    compare_drift_detectors,
    run_adaptive_comparison,
    run_adaptive_experiment,
    run_adaptive_training,
)
from heart_disease_rt.models.baseline import run_baseline_training
from heart_disease_rt.models.cv_benchmark import run_repeated_cv_benchmark
from heart_disease_rt.monitoring.drift_scenarios import (
    evaluate_detector_scenarios,
    simulate_drift_scenarios,
)

__all__ = [
    "SUPPORTED_DETECTORS",
    "compare_drift_detectors",
    "run_adaptive_comparison",
    "run_adaptive_experiment",
    "run_adaptive_training",
    "run_baseline_training",
    "run_repeated_cv_benchmark",
    "simulate_drift_scenarios",
    "evaluate_detector_scenarios",
]
