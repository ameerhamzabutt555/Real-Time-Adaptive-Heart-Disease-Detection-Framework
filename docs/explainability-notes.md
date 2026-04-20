# Explainability Notes

This project uses coefficient-based explainability for the baseline logistic regression model.

## Why this approach

- Logistic regression exposes direct coefficients for each feature.
- This makes it easy to justify feature influence in clinical review.
- It provides transparent global and local interpretation without heavy dependencies.

## Artifacts generated

- `experiments/tracking/feature_importance.csv`
  - Global absolute and normalized coefficient importance per feature.
- `experiments/tracking/local_explanations.csv`
  - Top contributing features for sample patients.

## Clinical usage

- Global importance highlights the most influential risk factors overall.
- Local explanations clarify why a specific patient received a given risk score.

## Future extension

- Add SHAP values with tree-based baseline as secondary explainability benchmark.
