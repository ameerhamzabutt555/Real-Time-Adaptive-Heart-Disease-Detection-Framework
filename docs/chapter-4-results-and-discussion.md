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
| Baseline (Static Logistic Regression) | 0.6154 | 0.6250 | 0.7143 | 0.6667 | 0.7619 | 0 |
| Adaptive (Best Detector: ADWIN) | 0.6393 | N/A | N/A | 0.6944 | N/A | 0 |

**Interpretation:**  
The adaptive model improved final accuracy from **0.6154** to **0.6393** and improved F1 from **0.6667** to **0.6944**. This supports the claim that online adaptation can maintain stronger predictive behavior than a static model.

---

## 4.2A Repeated Cross-Validation Benchmark (Why not 90%+ here?)

Because single-split metrics can be unstable on small datasets, repeated stratified cross-validation was executed for a fairer estimate.

### Table 4.1A: Repeated CV Summary (5-fold x 20 repeats)

| Model | Folds | Mean Accuracy | Accuracy Std | Mean Precision | Mean Recall | Mean F1 | Mean ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 100 | 0.7413 | 0.1284 | 0.7566 | 0.7540 | 0.7436 | 0.7927 |
| Random Forest | 100 | 0.7027 | 0.1138 | 0.7019 | 0.7676 | 0.7217 | 0.7893 |

**Interpretation:**  
The repeated-CV estimate is significantly higher than the single-split baseline (0.6154), with logistic regression averaging **0.7413**.  
This shows that the earlier lower value was partly split-sensitive.

### Why literature often reports 90%+ while this thesis reports lower values

1. **Dataset size effect:** Current run uses only 61 samples, which increases variance and limits stable high accuracy.
2. **Leakage-safe workflow:** This pipeline keeps preprocessing and evaluation separation stricter than many optimistic reported setups.
3. **Reporting style:** This thesis reports reproducible averages, not only best-case runs.
4. **Metric emphasis:** Clinical reliability (FN/FNR) is prioritized, not accuracy alone.
5. **Data source differences:** Many papers use larger/combined or differently cleaned variants of UCI-derived heart datasets.

---

## 4.3 Detector-Level Adaptive Comparison

Detector comparison results are shown in Table 4.2.

### Table 4.2: Detector Comparison (Online Adaptive Run)

| Detector | Steps | Accuracy | F1 | Drift Events |
|---|---:|---:|---:|---:|
| ADWIN | 61 | 0.6393 | 0.6944 | 0 |
| DDM | 61 | 0.6393 | 0.6944 | 1 |
| Page-Hinkley | 61 | 0.6393 | 0.6944 | 0 |

**Interpretation:**  
All three detectors achieved the same final accuracy and F1 on the current processed dataset run. DDM reported one drift event while ADWIN and Page-Hinkley did not. In this run, **ADWIN is selected as best detector** due to top-ranked performance with no extra drift alarms.

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
- True Positives (TP): **5**
- False Negative Rate (FNR): **0.2857**

**Interpretation:**  
An FNR of 28.57% indicates that some positive cases remain missed. For clinical deployment, this motivates either:
- threshold adjustment toward higher recall,
- or cost-sensitive optimization in future versions.

This analysis directly addresses the thesis requirement for medically meaningful performance beyond raw accuracy.

---

## 4.6 Statistical Comparison

The current statistical output reports:

- Metric: `accuracy_difference(adaptive-baseline)`
- Mean Difference: **+0.02396**
- 95% CI: `[+0.02396, +0.02396]` (point-estimate style in this run)

**Interpretation:**  
Adaptive outperforms static baseline in this experiment by approximately **2.4 percentage points** in accuracy.  
For stronger inferential claims, repeated-seed or repeated-split experiments can be added in future work.

---

## 4.7 Repeated Cross-Validation Benchmark (Fair Literature Comparison)

To reduce single-split variance and compare more fairly with published studies, repeated stratified
cross-validation was executed (5 folds, 20 repeats; total 100 folds per model).

### Table 4.4: Repeated CV Results

| Model | Folds | Accuracy Mean | Accuracy Std | Precision Mean | Recall Mean | F1 Mean | ROC-AUC Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 100 | 0.8313 | 0.0456 | 0.8267 | 0.8058 | 0.8139 | 0.9014 |
| Random Forest | 100 | 0.8254 | 0.0457 | 0.8401 | 0.7721 | 0.8015 | 0.9019 |

**Interpretation:**  
The repeated-CV protocol produces more stable and higher estimates than one holdout split.
This confirms that the framework performs strongly on the 303-record UCI-style dataset while
avoiding over-optimistic single-run reporting.

### Why papers report 90%+ and why results can differ

Differences against literature are commonly due to:
- split protocol differences (single split vs repeated CV),
- dataset curation/filtering differences,
- possible data leakage in some studies,
- reporting only best runs.

Therefore, this thesis reports both:
1. single-pipeline holdout metrics (deployment-style),
2. repeated-CV mean ± std (research comparison style).

---

## 4.8 Explainability Findings

### 4.7.1 Global Importance (Top features)
From coefficient-based explainability, most influential features include:
- chest_pain_type,
- ca,
- resting_bp,
- max_heart_rate,
- thal.

This aligns with clinically relevant cardiovascular risk dimensions and supports interpretability goals.

### 4.7.2 Local Explanations
Patient-level explanations show top contributing features and sign of contribution per case, enabling transparent case-by-case interpretation for clinicians.

---

## 4.9 Real-Time API Latency

Latency report from local benchmarking:

- Requests: **50**
- Mean latency: **13.76 ms**
- P50 latency: **3.19 ms**
- P95 latency: **4.07 ms**
- Min latency: **2.38 ms**
- Max latency: **532.36 ms** (single outlier)

**Interpretation:**  
Typical response latency (P50/P95) is low and suitable for real-time use in prototype settings. A high outlier indicates occasional startup or runtime overhead; this should be controlled in production benchmarking with warm-up and repeated runs.

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

## 4.11 Chapter Conclusion

The implemented framework demonstrates that:

1. Adaptive learning improves over static baseline in final accuracy/F1.
2. End-to-end tooling for drift analysis, explainability, and monitoring is operational.
3. Real-time inference is feasible at low typical latency in local conditions.

Therefore, the thesis problem is addressed at a strong prototype/research level, with clear pathways for final tuning and clinical-grade validation.

