# Metrics Definition

This file defines the key metrics used to validate thesis claims.

## Classification Metrics

- **Accuracy**: overall correct predictions ratio.
- **Precision**: true positives among predicted positives.
- **Recall (Sensitivity)**: true positives among actual positives.
- **F1**: harmonic mean of precision and recall.
- **ROC-AUC**: ranking quality of positive vs negative classes.

## Drift Metrics

- **drift_events**: number of times detector raised drift.
- **first_detection_delay**: steps from known drift start to first detection.
- **recovery_steps**: steps required to recover near pre-drift performance.
- **false_alarms_before_drift**: drift flags before true drift region.

## Clinical Error Focus

- **false_negative_rate**: missed positive cases among all actual positives.
  This is emphasized because missed disease cases are high-risk clinically.

## Statistical Comparison

- **accuracy_difference(adaptive-baseline)**: point estimate in this pipeline.
- A robust extension is repeated runs with bootstrap confidence intervals.
