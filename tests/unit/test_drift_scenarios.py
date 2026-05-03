from pathlib import Path

from heart_disease_rt.monitoring.drift_scenarios import (
    ScenarioConfig,
    evaluate_detector_on_scenario,
    generate_drift_scenario,
    run_drift_scenario_benchmark,
)


def test_generate_drift_scenario_has_truth_columns() -> None:
    scenario = generate_drift_scenario(
        ScenarioConfig(name="sudden", size=120, drift_start=60, transition=40),
        seed=1,
    )
    assert "drift_truth" in scenario.columns
    assert "is_post_drift" in scenario.columns
    assert set(scenario["scenario"].unique()) == {"sudden"}


def test_evaluate_detector_on_scenario_has_required_metrics() -> None:
    scenario = generate_drift_scenario(
        ScenarioConfig(name="gradual", size=140, drift_start=70, transition=35),
        seed=2,
    )
    progress, metrics = evaluate_detector_on_scenario(
        scenario,
        detector_name="adwin",
        scenario_name="gradual",
        drift_start_step=70,
    )
    required = {
        "detector",
        "scenario",
        "drift_events",
        "first_detection_step",
        "detection_delay",
        "recovery_delay",
        "final_accuracy",
        "final_f1",
    }
    assert required.issubset(set(metrics.keys()))
    assert len(progress) == len(scenario)


def test_run_drift_scenario_benchmark_creates_outputs(tmp_path: Path) -> None:
    outputs = run_drift_scenario_benchmark(output_dir=tmp_path, detectors=("adwin",), seed=3)
    assert outputs["metrics_csv"].exists()
    assert outputs["progress_csv"].exists()
