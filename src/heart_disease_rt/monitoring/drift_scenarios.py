"""Synthetic drift scenario generation and evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from pathlib import Path

import numpy as np
import pandas as pd
from river import compose, drift, linear_model, metrics, preprocessing

from heart_disease_rt.data.schema import FEATURE_COLUMNS, LABEL_COLUMN


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    size: int = 1000
    drift_start: int = 500
    transition: int = 180


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


def _sample_features(rng: np.random.Generator) -> dict[str, float]:
    age = float(rng.uniform(30, 80))
    sex = float(rng.integers(0, 2))
    chest_pain_type = float(rng.integers(0, 4))
    resting_bp = float(rng.normal(130, 18))
    cholesterol = float(rng.normal(240, 45))
    fasting_blood_sugar = float(rng.integers(0, 2))
    restecg = float(rng.integers(0, 3))
    max_heart_rate = float(rng.normal(150, 20))
    exercise_angina = float(rng.integers(0, 2))
    oldpeak = float(max(0.0, rng.normal(1.1, 1.0)))
    slope = float(rng.integers(0, 3))
    ca = float(rng.integers(0, 4))
    thal = float(rng.integers(0, 4))
    return {
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_bp": resting_bp,
        "cholesterol": cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "restecg": restecg,
        "max_heart_rate": max_heart_rate,
        "exercise_angina": exercise_angina,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }


def _risk_probability(features: dict[str, float], weights: dict[str, float], bias: float) -> float:
    linear = bias
    for feature in FEATURE_COLUMNS:
        linear += weights.get(feature, 0.0) * float(features[feature])
    return _sigmoid(linear)


def _weights_before() -> dict[str, float]:
    return {
        "age": 0.03,
        "resting_bp": 0.01,
        "cholesterol": 0.006,
        "exercise_angina": 0.7,
        "oldpeak": 0.55,
        "ca": 0.5,
        "thal": 0.3,
        "max_heart_rate": -0.01,
    }


def _weights_after() -> dict[str, float]:
    return {
        "age": 0.02,
        "resting_bp": 0.004,
        "cholesterol": 0.001,
        "exercise_angina": 0.9,
        "oldpeak": 0.7,
        "ca": 0.65,
        "thal": 0.35,
        "max_heart_rate": -0.008,
        "chest_pain_type": 0.3,
    }


def _mix_weight(step: int, config: ScenarioConfig) -> float:
    if config.name == "sudden":
        return 0.0 if step < config.drift_start else 1.0
    if config.name == "gradual":
        start = config.drift_start
        end = config.drift_start + config.transition
        if step <= start:
            return 0.0
        if step >= end:
            return 1.0
        return (step - start) / max(1, config.transition)
    if config.name == "recurring":
        cycle = max(50, config.transition)
        in_second_regime = ((step // cycle) % 2) == 1
        return 1.0 if in_second_regime else 0.0
    raise ValueError(f"Unsupported scenario: {config.name}")


def generate_drift_scenario(
    config: ScenarioConfig,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic stream data with labeled drift regime."""
    rng = np.random.default_rng(seed)
    before = _weights_before()
    after = _weights_after()
    rows: list[dict[str, float | int | str]] = []
    for step in range(config.size):
        features = _sample_features(rng)
        alpha = _mix_weight(step, config)
        weights = {k: (1 - alpha) * before.get(k, 0.0) + alpha * after.get(k, 0.0) for k in set(before) | set(after)}
        bias = -4.5 + (1.2 * alpha)
        prob = _risk_probability(features, weights, bias)
        label = int(rng.random() < prob)
        row: dict[str, float | int | str] = {
            **features,
            LABEL_COLUMN: label,
            "step": step + 1,
            "scenario": config.name,
            "is_post_drift": int(alpha >= 0.5),
            "drift_truth": int(alpha >= 0.5),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _build_detector(detector_name: str):
    normalized = detector_name.strip().lower()
    if normalized == "adwin":
        return drift.ADWIN()
    if normalized == "ddm":
        return drift.binary.DDM()
    if normalized in {"page_hinkley", "pagehinkley", "ph"}:
        return drift.PageHinkley()
    raise ValueError(f"Unsupported detector '{detector_name}'")


def evaluate_detector_on_scenario(
    df: pd.DataFrame,
    detector_name: str,
    scenario_name: str,
    drift_start_step: int,
    recovery_window: int = 100,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Run detector on scenario stream and compute delay/recovery metrics."""
    detector = _build_detector(detector_name)
    model = compose.Pipeline(preprocessing.StandardScaler(), linear_model.LogisticRegression())
    metric_acc = metrics.Accuracy()
    metric_f1 = metrics.F1()

    rows: list[dict[str, float | int | str]] = []
    first_detection_step: int | None = None

    for _, row in df.iterrows():
        features = {f: float(row[f]) for f in FEATURE_COLUMNS}
        y_true = int(row[LABEL_COLUMN])
        step = int(row["step"])
        y_pred_raw = model.predict_one(features)
        y_pred = int(y_pred_raw) if y_pred_raw is not None else 0

        metric_acc.update(y_true, y_pred)
        metric_f1.update(y_true, y_pred)
        err = int(y_pred != y_true)
        detector.update(err)
        drift_flag = int(detector.drift_detected)
        if drift_flag and first_detection_step is None and step >= drift_start_step:
            first_detection_step = step

        rows.append(
            {
                "scenario": scenario_name,
                "detector": detector_name,
                "step": step,
                "y_true": y_true,
                "y_pred": y_pred,
                "metric_accuracy": float(metric_acc.get()),
                "metric_f1": float(metric_f1.get()),
                "drift_flag": drift_flag,
                "drift_truth": int(row["drift_truth"]),
                "is_post_drift": int(row["is_post_drift"]),
            }
        )
        model.learn_one(features, y_true)

    progress_df = pd.DataFrame(rows)
    post = progress_df[progress_df["step"] >= drift_start_step]
    pre = progress_df[progress_df["step"] < drift_start_step]

    baseline_pre = float(pre["metric_accuracy"].mean()) if not pre.empty else 0.0
    target_accuracy = baseline_pre * 0.95 if baseline_pre > 0 else 0.0
    recovery_step: int | None = None
    if not post.empty and target_accuracy > 0:
        hit = post[post["metric_accuracy"] >= target_accuracy]
        if not hit.empty:
            recovery_step = int(hit["step"].iloc[0])

    metrics_payload: dict[str, float | int | str] = {
        "scenario": scenario_name,
        "detector": detector_name,
        "steps": int(len(progress_df)),
        "final_accuracy": float(progress_df["metric_accuracy"].iloc[-1]) if not progress_df.empty else 0.0,
        "final_f1": float(progress_df["metric_f1"].iloc[-1]) if not progress_df.empty else 0.0,
        "drift_events": int(progress_df["drift_flag"].sum()),
        "first_detection_step": int(first_detection_step) if first_detection_step is not None else -1,
        "detection_delay": int(first_detection_step - drift_start_step) if first_detection_step is not None else -1,
        "recovery_step": int(recovery_step) if recovery_step is not None else -1,
        "recovery_delay": int(recovery_step - drift_start_step) if recovery_step is not None else -1,
        "pre_drift_accuracy_mean": baseline_pre,
        "post_drift_accuracy_mean": float(post["metric_accuracy"].mean()) if not post.empty else 0.0,
        "window_size_for_recovery": int(recovery_window),
    }
    return progress_df, metrics_payload


def run_drift_scenario_benchmark(
    output_dir: Path,
    detectors: tuple[str, ...] = ("adwin", "ddm", "page_hinkley"),
    seed: int = 42,
) -> dict[str, Path]:
    """Generate all scenarios and benchmark detectors."""
    output_dir.mkdir(parents=True, exist_ok=True)
    scenario_cfgs = [
        ScenarioConfig(name="sudden", size=1000, drift_start=500, transition=120),
        ScenarioConfig(name="gradual", size=1000, drift_start=420, transition=280),
        ScenarioConfig(name="recurring", size=1000, drift_start=300, transition=160),
    ]
    all_metrics: list[dict[str, float | int | str]] = []
    all_progress: list[pd.DataFrame] = []

    for cfg in scenario_cfgs:
        scenario_df = generate_drift_scenario(cfg, seed=seed)
        scenario_path = output_dir / f"scenario_{cfg.name}.csv"
        scenario_df.to_csv(scenario_path, index=False)
        for detector_name in detectors:
            progress, metrics_payload = evaluate_detector_on_scenario(
                scenario_df,
                detector_name=detector_name,
                scenario_name=cfg.name,
                drift_start_step=cfg.drift_start,
            )
            all_metrics.append(metrics_payload)
            all_progress.append(progress)

    metrics_df = pd.DataFrame(all_metrics)
    progress_df = pd.concat(all_progress, ignore_index=True) if all_progress else pd.DataFrame()
    metrics_csv = output_dir / "drift_scenarios_report.csv"
    progress_csv = output_dir / "drift_scenarios_progress.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    progress_df.to_csv(progress_csv, index=False)
    return {
        "metrics_csv": metrics_csv,
        "progress_csv": progress_csv,
    }


def simulate_drift_scenarios(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create scenario datasets using observed feature distributions."""
    del frame  # synthetic generation currently independent of input frame shape stats
    cfgs = [
        ScenarioConfig(name="sudden", size=800, drift_start=400, transition=100),
        ScenarioConfig(name="gradual", size=800, drift_start=320, transition=220),
        ScenarioConfig(name="recurring", size=800, drift_start=260, transition=140),
    ]
    out: dict[str, pd.DataFrame] = {}
    for idx, cfg in enumerate(cfgs):
        df = generate_drift_scenario(cfg, seed=42 + idx)
        df = df.rename(columns={"drift_truth": "drift_ground_truth"})
        out[cfg.name] = df
    return out


def evaluate_detector_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility wrapper returning scenario-level detector metrics."""
    scenarios = simulate_drift_scenarios(frame)
    rows: list[dict[str, float | int | str]] = []
    for scenario_name, scenario_df in scenarios.items():
        if scenario_name == "sudden":
            drift_start = 400
        elif scenario_name == "gradual":
            drift_start = 320
        else:
            drift_start = 260
        for detector_name in ("adwin", "ddm", "page_hinkley"):
            progress, payload = evaluate_detector_on_scenario(
                scenario_df.rename(columns={"drift_ground_truth": "drift_truth"}),
                detector_name=detector_name,
                scenario_name=scenario_name,
                drift_start_step=drift_start,
            )
            false_alarms = int(((progress["step"] < drift_start) & (progress["drift_flag"] == 1)).sum())
            rows.append(
                {
                    "detector": detector_name,
                    "scenario": scenario_name,
                    "drift_events": int(payload["drift_events"]),
                    "first_detection_delay": int(payload["detection_delay"]),
                    "recovery_steps": int(payload["recovery_delay"]),
                    "false_alarms_before_drift": false_alarms,
                    "final_accuracy": float(payload["final_accuracy"]),
                    "final_f1": float(payload["final_f1"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_drift_scenarios(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return scenario summary and per-step event logs."""
    scenarios = simulate_drift_scenarios(frame)
    summary_rows: list[dict[str, float | int | str]] = []
    all_logs: list[pd.DataFrame] = []

    for scenario_name, scenario_df in scenarios.items():
        drift_start = 400 if scenario_name == "sudden" else 320 if scenario_name == "gradual" else 260
        for detector_name in ("adwin", "ddm", "page_hinkley"):
            progress, payload = evaluate_detector_on_scenario(
                scenario_df.rename(columns={"drift_ground_truth": "drift_truth"}),
                detector_name=detector_name,
                scenario_name=scenario_name,
                drift_start_step=drift_start,
            )
            false_alarms = int(((progress["step"] < drift_start) & (progress["drift_flag"] == 1)).sum())
            summary_rows.append(
                {
                    "detector": detector_name,
                    "scenario": scenario_name,
                    "drift_events": int(payload["drift_events"]),
                    "first_detection_delay": int(payload["detection_delay"]),
                    "recovery_steps": int(payload["recovery_delay"]),
                    "false_alarms_before_drift": false_alarms,
                    "final_accuracy": float(payload["final_accuracy"]),
                    "final_f1": float(payload["final_f1"]),
                }
            )
            all_logs.append(progress)

    summary_df = pd.DataFrame(summary_rows)
    events_df = pd.concat(all_logs, ignore_index=True) if all_logs else pd.DataFrame()
    return summary_df, events_df
