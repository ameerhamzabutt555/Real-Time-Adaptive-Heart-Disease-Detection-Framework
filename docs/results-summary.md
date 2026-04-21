# Results Summary (Filled from Current Run)

This file summarizes the generated results from:

```bash
make run-all INPUT=data/raw/heart.csv
```

## 1) Baseline vs Adaptive

Source: `experiments/tracking/baseline_vs_adaptive.csv`

- Baseline accuracy: **0.6154**
- Adaptive accuracy (best detector): **0.6393**
- Accuracy difference (adaptive - baseline): **+0.02396**
- Baseline F1: **0.6667**
- Adaptive F1: **0.6944**

Observation:
- Adaptive run shows improved final performance compared to static baseline.

## 2) Detector Comparison

Source: `experiments/tracking/detector_comparison.csv`

- ADWIN: accuracy 0.6393, F1 0.6944, drift_events 0
- DDM: accuracy 0.6393, F1 0.6944, drift_events 1
- Page-Hinkley: accuracy 0.6393, F1 0.6944, drift_events 0

Observation:
- Accuracy/F1 are similar in this run; ADWIN selected as best with zero drift alarms.

## 3) Drift Scenario Findings

Source: `experiments/tracking/drift_scenarios_report.csv`

- Evaluated scenarios: sudden, gradual, recurring
- Final accuracy range: **0.6400 to 0.6888**
- Final F1 range: **0.7198 to 0.7556**
- `first_detection_delay` remained `-1` across rows in this run (no confirmed post-drift trigger)
- `false_alarms_before_drift` was 0 in all rows

Observation:
- Adaptive behavior remains stable; detector trigger sensitivity should be tuned for stronger explicit detections under current synthetic setup.

## 4) Clinical Error Analysis

Source: `experiments/tracking/false_negative_analysis.json`

- False negatives: **2**
- True positives: **5**
- False negative rate: **0.2857**

Interpretation:
- Model still misses some positive cases; threshold tuning toward higher recall should be evaluated for clinical settings.

## 5) Statistical Note

Source: `experiments/tracking/statistical_comparison.json`

- Metric: `accuracy_difference(adaptive-baseline)`
- Mean difference: **+0.02396**
- CI (current implementation): **[+0.02396, +0.02396]**

Interpretation:
- Adaptive outperforms baseline in this run.
- For publication-strength inference, use repeated-seed experiments and bootstrap across runs.

## 6) Repeated Cross-Validation Benchmark (Literature Alignment)

Source: `experiments/tracking/cv_benchmark.csv`

- Logistic Regression (5-fold x 20 repeats):
  - Accuracy mean: **0.7413**
  - Accuracy std: **0.1284**
  - F1 mean: **0.7436**
  - ROC-AUC mean: **0.7927**
- Random Forest (5-fold x 20 repeats):
  - Accuracy mean: **0.7027**
  - Accuracy std: **0.1138**
  - F1 mean: **0.7217**
  - ROC-AUC mean: **0.7893**

Interpretation:
- Repeated CV provides a more stable estimate than a single split.
- Current dataset size (**n=61**) is small, so 90%+ stable accuracy is not expected.
- Literature values above 90% often use larger datasets, optimized feature engineering, or different evaluation setups.

## 7) Dataset Clarification (UCI)

- Yes, the current schema (`age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`) matches the **UCI Cleveland Heart Disease-style format**.
- However, this local working file currently has only **61 records**, not the full larger UCI variants commonly used in publications.

