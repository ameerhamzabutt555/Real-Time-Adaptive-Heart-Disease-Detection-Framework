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

Current file in this repository:
- `data/raw/heart.csv`

This file follows **UCI Heart Disease-style schema**:
- columns like `cp`, `trestbps`, `chol`, `fbs`, `thalach`, `exang`, `target`
- transformed to canonical schema during preprocessing.

However, this specific local sample currently has only about **61 records**, which is much smaller than commonly reported benchmark setups.

---

## 3) Repeated CV benchmark (more reliable than one split)

Source:
- `experiments/tracking/cv_benchmark.csv`

### Summary

- Logistic Regression:
  - mean accuracy: **0.7413**
  - std: **0.1284**
  - mean F1: **0.7436**
  - mean ROC-AUC: **0.7927**

- Random Forest:
  - mean accuracy: **0.7027**
  - std: **0.1138**
  - mean F1: **0.7217**
  - mean ROC-AUC: **0.7893**

Interpretation:
- Repeated CV gives a fairer estimate than single split.
- On this small sample, a realistic range is around 0.70–0.75 for accuracy, not stable 0.90+.

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

