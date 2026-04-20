# Results Summary Template

Populate this file after running:

```bash
make run-all INPUT=data/raw/heart.csv
```

Artifacts to reference:

- `experiments/tracking/baseline_vs_adaptive.csv`
- `experiments/tracking/detector_comparison.csv`
- `experiments/tracking/drift_scenarios_report.csv`
- `experiments/tracking/false_negative_analysis.json`
- `experiments/tracking/statistical_comparison.json`

## 1) Baseline vs Adaptive

- Which model had highest final accuracy?
- Was adaptive model more stable under drift?
- How many drift events were detected by best detector?

## 2) Drift Scenario Findings

For each scenario (`sudden`, `gradual`, `recurring`):

- best detector by final accuracy
- first detection delay
- recovery delay
- false alarms before drift

## 3) Clinical Error Analysis

Use `false_negative_analysis.json`:

- false negative rate
- interpretation for clinical safety
- suggested threshold adjustment if FNR is high

## 4) Statistical Note

Use `statistical_comparison.json`:

- mean difference
- confidence interval
- interpretation text

