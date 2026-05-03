# Chapter 4: Results and Discussion

## 4.1 Experimental Objectives Recap

This chapter evaluates whether the proposed framework addresses the thesis problem:

1. Static heart-disease models become outdated over time.
2. Concept drift reduces long-term reliability.
3. Real-time and adaptive behavior is required for practical deployment.

The implemented system was tested through:
- baseline static model evaluation,
- adaptive online learning with multiple drift detectors,
- synthetic drift scenario benchmarking,
- clinical error analysis,
- explainability outputs,
- and API latency measurement.

---

## 4.2 Baseline vs Adaptive Performance

Table 4.1 summarizes the primary comparison between static baseline and the best adaptive run.

### Table 4.1: Baseline vs Adaptive Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Drift Events |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Static Logistic Regression) | 0.8689 | 0.8125 | 0.9286 | 0.8667 | 0.9502 | 0 |
| Adaptive (Best Detector: ADWIN) | 0.8185 | N/A | N/A | 0.7955 | N/A | 0 |

**Interpretation:**  
On the UCI 303-record run, the **static baseline** achieved higher holdout accuracy (**0.8689**) than the adaptive online model (**0.8185**). This indicates that, when the evaluation distribution is stable and closely matches training conditions, a well-trained static classifier can outperform an online learner that is optimized for continual updates.  

However, the adaptive result remains strong given that it operates **in-memory**, learns **incrementally**, and is designed for non-stationary settings. Drift detection events were **0** for this run, suggesting no strong drift signal (under the current detector settings and stream order).

---

## 4.2A Repeated Cross-Validation Benchmark (Fairer estimate on 303 records)

Because single-split metrics can be sensitive to the specific holdout split, repeated stratified cross-validation was executed for a fairer estimate (5 folds × 20 repeats; 100 total folds per model).

### Table 4.1A: Repeated CV Summary (5-fold × 20 repeats)

| Model | Folds | Mean Accuracy | Accuracy Std | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 100 | 0.8313 | 0.0456 | 0.8267 | 0.8058 | 0.8139 | 0.9014 |
| Random Forest | 100 | 0.8254 | 0.0457 | 0.8401 | 0.7721 | 0.8015 | 0.9019 |

**Interpretation:**  
The repeated-CV estimate for logistic regression is **0.8313 ± 0.0456**, indicating stable performance on the UCI (303) dataset. The CV means are slightly lower than the best single holdout result (0.8689), which is expected: repeated-CV averages across many different splits rather than reporting a single potentially favorable split.

### Why literature often reports 90%+ while results can differ

Differences against literature are commonly due to:
1. **Split protocol differences:** single split vs repeated CV, and different stratification choices.
2. **Dataset curation differences:** removed rows, imputation rules, label mapping, and feature engineering vary across studies.
3. **Possible data leakage in some studies:** particularly when preprocessing is fit on the full dataset before evaluation.
4. **Reporting only best runs:** some papers report peak accuracy rather than mean ± std across multiple runs.

## 4.3 Detector-Level Adaptive Comparison

Detector comparison results are shown in Table 4.2.

### Table 4.2: Detector Comparison (Online Adaptive Run)

| Detector | Steps | Accuracy | F1 | Drift Events |
|---|---:|---:|---:|---:|
| ADWIN | 303 | 0.8185 | 0.7955 | 0 |
| DDM | 303 | 0.8185 | 0.7955 | 0 |
| Page-Hinkley | 303 | 0.8185 | 0.7955 | 0 |

**Interpretation:**  
All three detectors achieved the same final accuracy and F1 on the current UCI run. None of the detectors reported drift events in this configuration. In such cases, detector selection can be based on operational preferences (e.g., conservativeness vs sensitivity) because predictive performance is identical for this run.

---

## 4.4 Drift Scenario Analysis (Sudden, Gradual, Recurring)

To test robustness beyond a single dataset stream, synthetic scenarios were generated.

### Table 4.3: Drift Scenario Benchmark Summary

| Scenario | Detector | Drift Events | First Detection Delay | Recovery Steps | False Alarms Before Drift | Final Accuracy | Final F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Sudden | ADWIN | 0 | -1 | 0 | 0 | 0.6400 | 0.7198 |
| Sudden | DDM | 0 | -1 | 0 | 0 | 0.6400 | 0.7198 |
| Sudden | Page-Hinkley | 0 | -1 | 0 | 0 | 0.6400 | 0.7198 |
| Gradual | ADWIN | 0 | -1 | 0 | 0 | 0.6888 | 0.7547 |
| Gradual | DDM | 0 | -1 | 0 | 0 | 0.6888 | 0.7547 |
| Gradual | Page-Hinkley | 0 | -1 | 0 | 0 | 0.6888 | 0.7547 |
| Recurring | ADWIN | 0 | -1 | 0 | 0 | 0.6888 | 0.7556 |
| Recurring | DDM | 0 | -1 | 0 | 0 | 0.6888 | 0.7556 |
| Recurring | Page-Hinkley | 0 | -1 | 0 | 0 | 0.6888 | 0.7556 |

**Interpretation:**  
The scenario benchmark currently shows stable performance but no positive detections (`-1` delay indicates no confirmed drift trigger after ground-truth transition). This means:

- adaptive learning remains stable under generated shifts,
- but detector sensitivity in this synthetic setup may be conservative,
- and detector hyperparameter tuning should be considered for stronger explicit drift-trigger evidence.

---

## 4.5 Clinical Error Analysis (False Negatives)

Clinical reliability was evaluated using false-negative behavior.

- False Negatives (FN): **2**
- True Positives (TP): **26**
- False Negative Rate (FNR): **0.0714**

**Interpretation:**  
An FNR of **7.14%** indicates that relatively few positive cases were missed in this evaluation. For clinical deployment, it is still important to control missed-risk cases, which motivates either:
- threshold adjustment toward higher recall,
- or cost-sensitive optimization in future versions.

This analysis directly addresses the thesis requirement for medically meaningful performance beyond raw accuracy.

---

## 4.6 Statistical Comparison

The current statistical output reports:

- Metric: `accuracy_difference(adaptive-baseline)`
- Mean Difference: **-0.05037**
- 95% CI: `[-0.05037, -0.05037]` (point-estimate style in this run)

**Interpretation:**  
On this run, adaptive underperforms the static baseline by approximately **5.0 percentage points** in accuracy.  
For stronger inferential claims, repeated-seed or repeated-split experiments can be added in future work.

---

## 4.7 Repeated Cross-Validation Benchmark (Summary)

Repeated CV results are presented in Table 4.1A and used as the primary research-style estimate (mean ± std), while Table 4.1 serves as the deployment-style holdout comparison.

---

## 4.8 Explainability Findings

### 4.7.1 Global Importance (Top features)
From coefficient-based explainability, most influential features include:
- ca,
- thal,
- sex,
- chest_pain_type,
- exercise_angina.

This aligns with clinically relevant cardiovascular risk dimensions and supports interpretability goals.

### 4.7.2 Local Explanations
Patient-level explanations show top contributing features and sign of contribution per case, enabling transparent case-by-case interpretation for clinicians.

---

## 4.9 Real-Time API Latency

Latency report from local benchmarking:

- Requests: **50**
- Mean latency: **2.10 ms**
- P50 latency: **1.86 ms**
- P95 latency: **2.49 ms**
- Min latency: **1.73 ms**
- Max latency: **7.84 ms**

**Interpretation:**  
Typical response latency (P50/P95) is low and suitable for real-time use in prototype settings. These results were collected on a local machine and should be re-measured under production-like conditions (warm-up, steady-state load, and representative hardware/network).

---

## 4.10 Discussion Against Thesis Objectives

### Objective 1: Adaptive diagnosis model that continuously learns
**Status: Achieved (prototype level).**  
Online adaptive training, detector comparison, and scenario-level analysis are implemented and reproducible.

### Objective 2: Real-time diagnostic framework
**Status: Achieved (prototype level).**  
FastAPI serving (`/predict`, `/predict/threshold`, `/model-info`) and Streamlit dashboard are integrated, with measured low typical latency.

### Concept drift management claim
**Status: Partially achieved with stable behavior; detector sensitivity can be further tuned.**  
Drift benchmark pipeline exists and outputs required metrics, but explicit positive detections are limited in current synthetic parameterization.

### Dataset clarification for this study
The current project uses a **UCI-format heart disease tabular schema** (`age, sex, cp, trestbps, chol, ... , target`) and then maps it into canonical project columns (`chest_pain_type, resting_bp, cholesterol, ... , label`).  
So yes, the workflow is aligned with UCI-style heart disease data structure.

---

## 4.11 Extended Results on a Larger UCI-Format Dataset (1025 records)

In addition to the 303-record Cleveland subset, an extended UCI-format heart dataset (`data/raw/heart.csv`, 1025 rows) was evaluated using the same preprocessing and evaluation workflow.

### Table 4.5: Baseline vs Adaptive (1025-record dataset)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Drift Events |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (Static Logistic Regression) | 0.8732 | 0.8264 | 0.9524 | 0.8850 | 0.9466 | 0 |
| Adaptive (Online Logistic Regression + scaler) | 0.8263 | N/A | N/A | 0.8405 | N/A | 0 |
| Adaptive (Online Adaptive Random Forest) | 0.8673 | N/A | N/A | 0.8707 | N/A | 0 |
| Baseline (HistGradientBoosting, tuned threshold) | 0.9805 | 0.9633 | 1.0000 | 0.9813 | 0.9973 | 0 |

**Interpretation:**  
On the larger dataset, the baseline logistic regression retains a performance advantage over the simplest adaptive learner (accuracy **0.8732** vs **0.8263**). This gap is expected because the adaptive model is evaluated in a strict prequential manner and begins with limited prior information.  

After upgrading the adaptive learner to an online **Adaptive Random Forest** ensemble, the adaptive accuracy improves to **0.8673**, substantially narrowing the gap to the static baseline while retaining the ability to update continuously over time.

The boosted static baseline (HistGradientBoosting) reaches **0.9805** accuracy on this particular holdout split. This highlights an important methodological point: powerful batch learners can achieve very high holdout accuracy when the data distribution is stable and the model can be trained offline on a large labeled dataset. However, this is not the same as a streaming adaptive guarantee—online models prioritize continual updates, robustness to drift, and fast incremental learning.

### Clinical error behavior (1025-record dataset)

- False Negatives (FN): **5**
- True Positives (TP): **100**
- False Negative Rate (FNR): **0.0476**

**Interpretation:**  
The false negative rate of **4.76%** indicates fewer missed positive cases, which is desirable for clinical safety. This improvement is expected when evaluation is performed on a larger sample that provides a more stable estimate.

### Statistical difference summary (1025-record dataset)

- Metric: `accuracy_difference(adaptive-baseline)`
- Mean Difference: **-0.04683**

This indicates the adaptive model is approximately **4.68 percentage points** below the baseline accuracy on this dataset under the current streaming and detector configuration.

---

## 4.12 Chapter Conclusion

The implemented framework demonstrates that:

1. The baseline static model achieves strong performance on the UCI dataset, while the adaptive online model remains competitive and provides a path to continual learning in non-stationary settings.
2. End-to-end tooling for drift analysis, explainability, and monitoring is operational.
3. Real-time inference is feasible at low typical latency in local conditions.

Therefore, the thesis problem is addressed at a strong prototype/research level, with clear pathways for final tuning and clinical-grade validation.

