# Literature vs Our Results (Accuracy Context)

## 1) Important Clarification

High 90%+ accuracies in heart-disease papers often come from:
- larger or curated datasets,
- different splitting/validation choices,
- potential leakage in some studies,
- or reporting only best-case runs.

Directly comparing one single split from a very small dataset against such papers is not fair.

---

## 2) What dataset are we using right now?

Current repository now contains both:
- `data/raw/heart.csv` (small local sample used in early runs)
- `data/raw/heart_uci_303.csv` (**official UCI id=45 download, 303 rows**)

This file follows **UCI Heart Disease-style schema**:
- columns like `cp`, `trestbps`, `chol`, `fbs`, `thalach`, `exang`, `target`
- transformed to canonical schema during preprocessing.

If you run benchmarks on `heart_uci_303.csv`, you get a more representative estimate for literature comparison.

---

## 3) Repeated CV benchmark (more reliable than one split)

Source:
- `experiments/tracking/cv_benchmark.csv` (latest run on `heart_uci_303.csv`)

### Summary

- Logistic Regression:
  - mean accuracy: **0.8313**
  - std: **0.0456**
  - mean F1: **0.8139**
  - mean ROC-AUC: **0.9014**

- Random Forest:
  - mean accuracy: **0.8254**
  - std: **0.0457**
  - mean F1: **0.8015**
  - mean ROC-AUC: **0.9019**

Interpretation:
- Repeated CV gives a fairer estimate than single split.
- On official UCI-303, realistic and reproducible performance is around **0.83 mean accuracy** for this pipeline.
- This is strong, but still below some 90%+ papers, which can be due to architecture differences, feature engineering choices, or evaluation leakage in some reports.

---

## 4) Why our current accuracy can be lower than some papers

1. **Dataset size effect**  
   Smaller datasets produce high variance and unstable estimates.

2. **Evaluation protocol differences**  
   Single split vs repeated CV / nested CV can shift reported accuracy noticeably.

3. **Leakage risk in literature**  
   Some studies preprocess/select features before split, inflating metrics.

4. **Metric emphasis mismatch**  
   Some papers focus on accuracy only; clinical settings require recall/FNR balance too.

---

## 5) How to make comparison thesis-strong

Use this reporting strategy:
- Report single-split result (for reproducibility in pipeline),
- Report repeated CV mean ± std (primary fairness metric),
- Report F1/Recall/FNR alongside accuracy,
- Add this caveat explicitly in thesis text:
  - "Due to limited local sample size, repeated stratified CV was used to produce stable comparative estimates."

