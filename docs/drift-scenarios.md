# Drift Scenario Experiments

This document defines synthetic drift regimes used to benchmark adaptive detectors.

## Scenarios

1. **Sudden drift**
   - distribution changes abruptly at `drift_start`
   - expected behavior: low detection delay if detector is sensitive

2. **Gradual drift**
   - distribution moves slowly over transition window
   - expected behavior: fewer false alarms, smooth recovery

3. **Recurring drift**
   - concept alternates between two regimes over cycles
   - expected behavior: resilient re-detection and stable long-run accuracy

## Metrics

- `drift_events`
- `first_detection_delay`
- `recovery_steps`
- `false_alarms_before_drift`
- `final_accuracy`
- `final_f1`

## Execution

```bash
make run-drift-scenarios DATA=data/processed/heart_disease_processed.csv
```

Outputs:

- `experiments/tracking/drift_scenarios_report.csv`
- `experiments/tracking/drift_event_log.csv`
