# Metrics Reference & Mathematical Definitions

TrustBench implements a comprehensive battery of evaluation metrics across classification, calibration, uncertainty estimation, selective prediction, regression, information retrieval, semantic segmentation, and algorithmic fairness.

Each metric is implemented according to published academic literature, validated against gold-standard reference implementations (such as `scikit-learn`, `Fairlearn`, and independent reference solvers), and unit-tested for edge-case safety.

---

## 1. Classification Metrics

Classification evaluation assesses discrete decision accuracy and error distribution across classes.

### Definitions & Formulations

- **Accuracy**:
  $$\text{Accuracy} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$

- **Balanced Accuracy**:
  Unweighted macro-average of recall across all classes, protecting against evaluation distortions caused by class imbalance:
  $$\text{Balanced Accuracy} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$

- **Macro Precision, Recall, and F1**:
  $$\text{Macro Precision} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c}, \quad \text{Macro Recall} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{\text{TP}_c + \text{FN}_c}$$
  $$\text{Macro F1} = \frac{1}{C} \sum_{c=1}^C \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$
  *(Classes with zero true positives and predictions default to $0.0$ to prevent division by zero).*

- **Confusion Matrix**:
  $C \times C$ cross-tabulation where row $i$ represents true class $i$ and column $j$ represents predicted class $j$.

### Verification & Reference Standard
- **Reference**: `sklearn.metrics.accuracy_score`, `balanced_accuracy_score`, `precision_recall_fscore_support(average="macro")`, and `confusion_matrix`.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestClassificationReferenceBenchmarks`.

---

## 2. Calibration Metrics

Calibration quantifies the alignment between a model's predicted confidence probabilities and its true empirical correctness.

### Definitions & Mathematical Conventions

- **Expected Calibration Error (ECE)**:
  Measures the expected absolute difference between confidence and accuracy across $M$ equal-width bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - **Default Bin Count**: $M = 15$ (`n_bins=15`).
  - **Bin Partitioning**: $M$ intervals over $[0.0, 1.0]$ generated via `np.linspace(0.0, 1.0, M + 1)`.
  - **Boundary Inclusivity**: Bins $m \in \{0, \dots, M-2\}$ are left-closed and right-open $[b_m, b_{m+1})$. The final bin ($m = M-1$) is closed on both sides $[b_{M-1}, 1.0]$ to include predictions with exact confidence $1.0$.

- **Adaptive Expected Calibration Error (Adaptive ECE)**:
  Addresses empty-bin and confidence-clustering artifacts by partitioning samples into $M$ quantile bins of equal sample count:
  $$\text{AdaECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  - **Default Bin Count**: $M = 15$ (`n_bins=15`).
  - **Binning Scheme**: Quantile split via `np.array_split` after sorting by confidence.

- **Maximum Calibration Error (MCE)**:
  Worst-case calibration gap observed across all non-empty bins:
  $$\text{MCE} = \max_{m: |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **Multi-Class Brier Score**:
  Mean squared error between the full predicted probability vector and one-hot ground truth:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$
  *(Note on convention: In binary classification, this multi-class formulation yields $2 \times$ the single-column binary Brier score $\frac{1}{N} \sum (p_1 - y_1)^2$, accounting for both positive and negative class residuals).*

- **Negative Log-Likelihood (NLL / Log-Loss)**:
  Cross-entropy loss with numerical stabilization ($\epsilon = 10^{-15}$):
  $$\text{NLL} = -\frac{1}{N} \sum_{i=1}^N \ln(p_{i, y_i})$$

### Verification & Reference Standard
- **References**:
  - Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). *On calibration of modern neural networks*. ICML.
  - Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015). *Obtaining well calibrated probabilities using Bayesian binning into quantiles*. AAAI.
  - Ding, Y., Liu, J., Xiong, H., & Huan, J. (2020). *Revisiting the evaluation of uncertainty estimation and its application to out-of-distribution detection*.
  - `sklearn.metrics.brier_score_loss`, `sklearn.metrics.log_loss`.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestCalibrationReferenceBenchmarks`.

---

## 3. Uncertainty & Error Detection

Uncertainty quantification tests whether a model's internal uncertainty signals can reliably identify incorrect predictions prior to downstream failure.

### Definitions & Conventions

- **Predictive Shannon Entropy**:
  Per-sample Shannon entropy:
  $$H(p) = -\sum_{c=1}^C p_c \ln(p_c)$$
  - **Logarithm Base**: Natural logarithm (**nats**), computed via `np.log` with a minimum probability floor $\epsilon = 10^{-12}$.
  - **Units**: nats. To convert to information bits (shannons), multiply by $\log_2(e) = \frac{1}{\ln(2)} \approx 1.4427$.
  - **Range**: $[0, \ln(C)]$, where $\ln(C)$ indicates uniform uncertainty over $C$ classes.

- **Free Energy Score**:
  Temperature-scaled negative free energy:
  $$E(x; T) = -T \cdot \ln \sum_{c=1}^C \exp\left(\frac{f_c(x)}{T}\right)$$
  - Positive free energy $-E(x; T)$ acts as an uncertainty score that scales with low confidence and out-of-distribution distance. Default temperature $T = 1.0$.

- **Error Detection AUROC**:
  Area Under the Receiver Operating Characteristic curve where prediction failure ($\hat{y}_i \neq y_i$) is the positive target class ($1 = \text{error}, 0 = \text{correct}$) and uncertainty is the decision score.
  - $\text{AUROC} = 0.50$ represents random guessing.
  - $\text{AUROC} = 1.0$ represents an oracle uncertainty score that perfectly flags every mistake.

- **Error Detection AUPRC**:
  Area Under the Precision-Recall curve for error detection, sensitive to class imbalance when the base error rate is low.

- **FPR at 95% TPR**:
  The False Positive Rate (fraction of correct predictions incorrectly flagged as uncertain) when the rejection threshold is set to capture at least 95% of all actual model errors:
  $$\text{FPR}_{95} = \min \left\{ \text{FPR} \mid \text{TPR} \ge 0.95 \right\}$$

### Verification & Reference Standard
- **References**:
  - Hendrycks, D., & Gimpel, K. (2017). *A baseline for detecting misclassified and out-of-distribution examples in neural networks*. ICLR.
  - Liu, W., Wang, X., Owens, J., & Li, Y. (2020). *Energy-based out-of-distribution detection*. NeurIPS.
  - `sklearn.metrics.roc_auc_score`, `sklearn.metrics.precision_recall_curve`, `sklearn.metrics.roc_curve`.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestErrorDetectionReferenceBenchmarks`.

---

## 4. Selective Prediction & Risk-Coverage (AURC)

Selective classification models have the option to abstain or defer low-confidence inferences to human experts or secondary systems.

### Coverage Range, Curve, and AURC Normalization

- **Empirical Coverage Spectrum for AURC**:
  The scalar metric **AURC** integrates across the **full empirical coverage spectrum** from $k = 1$ to $N$ ($c \in [1/N, 1.0]$) following the standard formulation by Geifman & El-Yaniv (2017):
  $$\text{AURC} = \frac{1}{N} \sum_{k=1}^N \mathcal{R}(S_k)$$
  where $S_k$ is the subset of the $k$ most confident samples, and $\mathcal{R}(S_k) = \frac{1}{k} \sum_{i \in S_k} \mathbb{I}(y_i \neq \hat{y}_i)$ is the empirical selective risk at sample threshold $k$.

- **Optimal AURC ($\text{AURC}^*$)**:
  The theoretical lower bound achieved by an oracle that ranks all erroneous predictions after all correct predictions across the full dataset:
  $$\text{AURC}^* = \frac{1}{N} \sum_{k=N-E+1}^N \left( 1 - \frac{N-E}{k} \right)$$
  where $E = \sum_{i=1}^N \mathbb{I}(y_i \neq \hat{y}_i)$ is the total error count, and $N$ is total sample count.

- **Excess AURC (E-AURC)**:
  The ranking penalty of the model's confidence scores beyond optimal ranking:
  $$\text{Excess AURC} = \max(0.0, \, \text{AURC} - \text{AURC}^*)$$
  Excess AURC is lower-bounded by $0.0$ (perfect confidence ranking) and isolates confidence ranking quality from the model's baseline error rate.

- **Discrete Risk-Coverage Table / Curve**:
  For tabular summaries and reports, TrustBench evaluates discrete checkpoints across coverage fractions:
  - **Default Checkpoints**: `coverages = (1.0, 0.9, 0.8, 0.7, 0.5)`.
  - **Customization**: Any custom sequence in $(0, 1]$ can be provided (e.g. `np.linspace(0.1, 1.0, 20)`).
  - *Note*: While the discrete curve summary inspects these specific coverage fractions, the scalar AURC and Excess AURC metrics always integrate over the full spectrum $c \in [1/N, 1.0]$.

### Verification & Reference Standard
- **Reference**: Geifman, Y., & El-Yaniv, R. (2017). *Selective classification for deep neural networks*. NeurIPS.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestSelectivePredictionReferenceBenchmarks`.

---

## 5. Regression Metrics

Regression evaluation covers error magnitudes, goodness-of-fit, and predictive uncertainty intervals.

### Definitions & Conventions

- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$.
- **Mean Squared Error (MSE)**: $\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$.
- **Root Mean Squared Error (RMSE)**: $\sqrt{\text{MSE}}$.
- **$R^2$ Score**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
- **Median Absolute Error (MedAE)**: $\text{median}(|y_1 - \hat{y}_1|, \dots, |y_N - \hat{y}_N|)$.
- **Maximum Error**: $\max_i |y_i - \hat{y}_i|$.
- **Mean Absolute Percentage Error (MAPE)**:
  $$\text{MAPE} = \frac{100}{N_{\text{nonzero}}} \sum_{i: |y_i| > 10^{-8}} \frac{|y_i - \hat{y}_i|}{|y_i|}$$
  - **Zero Denominator Handling**: Samples where $|y_i| \le 10^{-8}$ are masked out to avoid division by zero. If all targets are within $10^{-8}$ of zero, MAPE safely returns $0.0$.
  - **Units**: Expressed as a percentage ($0 - 100\%+$).

- **Prediction Interval Coverage Probability (PICP) & Mean Prediction Interval Width (MPIW)**:
  - **Requirement**: Evaluated when the regression model produces sample-level predictive standard deviations $\sigma_i$ (e.g., via `predict_uncertainty`, Gaussian Processes, or ensembles). Standard point regressors like `Ridge` do not output standard deviations; in such cases, PICP and MPIW are safely omitted.
  - **95% Confidence Interval**: Formed as $[\hat{y}_i - 1.96 \sigma_i, \, \hat{y}_i + 1.96 \sigma_i]$.
  - **PICP**: Empirical coverage fraction $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i - 1.96\sigma_i \le y_i \le \hat{y}_i + 1.96\sigma_i)$.
  - **MPIW**: Average interval breadth $\frac{1}{N} \sum_{i=1}^N (2 \times 1.96\sigma_i) \approx \frac{3.92}{N} \sum \sigma_i$.

### Verification & Reference Standard
- **Reference**: `sklearn.metrics.mean_absolute_error`, `mean_squared_error`, `r2_score`, `median_absolute_error`, `max_error`.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestRegressionReferenceBenchmarks`.

---

## 6. Information Retrieval Metrics

Information retrieval metrics evaluate the ranking quality of retrieved documents or items for a set of queries.

### Definitions & Conventions

- **Mean Reciprocal Rank (MRR)**:
  $$\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}$$
  where $\text{rank}_q$ is the position of the first relevant document for query $q$ (or $0$ if no relevant document was retrieved).

- **Mean Average Precision (MAP)**:
  Mean of Average Precision scores over all queries:
  $$\text{AP} = \frac{1}{|R_q|} \sum_{k=1}^K P@k \cdot \mathbb{I}(\text{doc}_k \in R_q)$$

- **Recall@K & Precision@K**:
  Fraction of relevant items retrieved within top $K$, and fraction of top $K$ items that are relevant.

- **Normalized Discounted Cumulative Gain (nDCG@K)**:
  - **Gain Scheme**: TrustBench uses **linear binary gain** $\text{rel}_i = \mathbb{I}(\text{doc}_i \in R_q)$.
  - **Discount**: Logarithmic discount $\log_2(i + 1)$ with 1-based rank $i$:
    $$\text{DCG}@K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}$$
    $$\text{IDCG}@K = \sum_{i=1}^{\min(K, |R_q|)} \frac{1}{\log_2(i + 1)}$$
    $$\text{nDCG}@K = \begin{cases} \frac{\text{DCG}@K}{\text{IDCG}@K} & \text{if } \text{IDCG}@K > 0 \\ 0.0 & \text{otherwise} \end{cases}$$

---

## 7. Semantic Segmentation Metrics

Segmentation evaluation measures pixel-level classification overlap and boundary accuracy.

### Definitions & Conventions

- **Mean Intersection-over-Union (mIoU)**:
  $$\text{IoU}_c = \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$$
  $$\text{mIoU} = \frac{1}{|C_{\text{present}}|} \sum_{c \in C_{\text{present}}} \text{IoU}_c$$
  - **Absent Class Handling**: Any class absent from both ground truth and prediction ($\text{TP}_c + \text{FP}_c + \text{FN}_c = 0$) results in $\text{NaN}$ and is **excluded** from the macro-average. If all classes are absent, mIoU returns $0.0$.
  - **Void / Boundary Filtering**: An optional `ignore_index` parameter (e.g., `255` in Cityscapes/Pascal VOC) filters ignored pixels from evaluation prior to metric computation.

- **Dice Score (Macro F1)**:
  $$\text{Dice}_c = \frac{2 \cdot \text{TP}_c}{2 \cdot \text{TP}_c + \text{FP}_c + \text{FN}_c}$$
  - **Aggregation**: TrustBench reports the **unweighted macro-average** across present classes (where $2\text{TP}_c + \text{FP}_c + \text{FN}_c > 0$).
  - *"Mean harmonic score"* denotes the macro-average of per-class harmonic means between pixel precision and recall.

- **Pixel Accuracy**: Overall ratio of correctly classified pixels $\frac{\sum \text{TP}_c}{N_{\text{valid\_pixels}}}$.

- **Frequency-Weighted IoU (FWIoU)**:
  $$\text{FWIoU} = \sum_{c \in C_{\text{present}}} \frac{\sum_{i} \mathbb{I}(y_i = c)}{N_{\text{valid\_pixels}}} \cdot \text{IoU}_c$$

---

## 8. Algorithmic Fairness Metrics

Fairness metrics measure disparities in model predictions and error rates across sensitive demographic or operational attributes (e.g., age, gender, geographic region).

### Definitions & Zero-Denominator Safety

- **Demographic Parity Difference (DPD)**:
  Maximum absolute difference in positive selection rate across all protected groups:
  $$\text{DPD} = \max_a P(\hat{Y}=1 \mid A=a) - \min_a P(\hat{Y}=1 \mid A=a)$$

- **Disparate Impact Ratio (DIR)**:
  Ratio between the minimum and maximum positive selection rates across groups:
  $$\text{DIR} = \frac{\min_a P(\hat{Y}=1 \mid A=a)}{\max_a P(\hat{Y}=1 \mid A=a)}$$
  - **Zero-Denominator Handling**: If the maximum selection rate across all groups is zero ($\max_a \text{selection\_rate}_a \le 10^{-12}$), TrustBench safely returns `1.0` (reflecting zero disparity, avoiding division by zero).

- **Equalized Odds Difference (EOD)**:
  Maximum disparity in True Positive Rate (TPR) or False Positive Rate (FPR) across groups:
  $$\text{EOD} = \max \left( \max_{a, b} |\text{TPR}_a - \text{TPR}_b|, \, \max_{a, b} |\text{FPR}_a - \text{FPR}_b| \right)$$

### Verification & Reference Standard
- **References**:
  - Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning*. NeurIPS.
  - Feldman, M., Friedler, S. A., Moeller, J., Scheidegger, C., & Venkatasubramanian, S. (2015). *Certifying and removing disparate impact*. KDD.
  - Matches the formal specification used by `Fairlearn`.
- **Test Suite**: Verified in `tests/unit/test_metric_reference_benchmarks.py::TestFairnessReferenceBenchmarks`.

---

## 9. Interpretation Guide & Diagnostic Playbook

Understanding what constitutes a "good" metric value and how metrics interact is critical for diagnosing model weaknesses.

### Diagnostic Rules of Thumb

| Metric | Ideal | Acceptable / Good | Needs Mitigation | Diagnostic Significance |
| :--- | :---: | :---: | :---: | :--- |
| **ECE** | $0.00$ | $< 0.05$ | $> 0.15$ | Low ECE means probability matches actual empirical accuracy. High ECE indicates over- or under-confidence. |
| **Brier Score (Multi-Class)** | $0.00$ | Task-dependent | Elevated | Combines accuracy and calibration in a strictly proper scoring rule. |
| **Error Detection AUROC** | $1.00$ | $> 0.75$ | $< 0.60$ | $> 0.85$ allows high-confidence filtering of model failures before reaching users. |
| **FPR at 95% TPR** | $0.00$ | $< 0.30$ | $> 0.50$ | Percentage of correct inferences unnecessarily rejected to catch 95% of mistakes. |
| **Excess AURC** | $0.00$ | $< 0.02$ | $> 0.08$ | Measures gap between model confidence ranking and optimal error ranking. |
| **Demographic Parity Diff** | $0.00$ | $< 0.10$ | $> 0.20$ | Difference in positive rate across groups. Regulatory compliance often targets $< 0.10$. |
| **Disparate Impact Ratio** | $1.00$ | $0.80 - 1.25$ | $< 0.80$ | Complies with the EEOC "four-fifths rule" when $\ge 0.80$. |
| **nDCG@K** | $1.00$ | $> 0.70$ | $< 0.50$ | Evaluates ranking discount; higher indicates relevant items placed near top. |
| **PICP (Nominal 95%)** | $0.95$ | $\ge 0.95$ | $< 0.90$ | Empirical coverage below nominal indicates overly narrow intervals (underestimated risk). |

### Metric Dynamics: Which Metrics Move Together?

1. **Temperature Scaling**:
   - Reduces **ECE** and **NLL** without modifying logits rank order.
   - Leaves **Accuracy**, **Error Detection AUROC**, and **AURC** completely unchanged because monotonic probability rescaling preserves ranking.
2. **Overconfident Neural Networks**:
   - Modern deep networks frequently exhibit **high accuracy** alongside **poor (high) ECE** and **high NLL** due to uncalibrated overconfidence on edge samples.
3. **Fairness vs Accuracy Trade-off**:
   - Enforcing strict demographic parity ($\text{DPD} \to 0$) in settings with differing base rates often slightly degrades overall accuracy or balanced accuracy.

### Worked Examples per Category

#### Example A: Classification & Calibration
```python
import numpy as np
from trustbench.evaluation.calibration import compute_calibration_metrics
from trustbench.evaluation.classification import compute_classification_metrics

# Ground truth and predicted class probabilities
y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
y_prob = np.array([
    [0.90, 0.10],
    [0.20, 0.80],
    [0.45, 0.55],
    [0.70, 0.30],
    [0.30, 0.70],
    [0.85, 0.15],
    [0.40, 0.60],  # Misclassification
    [0.10, 0.90],
])
y_pred = np.argmax(y_prob, axis=1)

clf_metrics = compute_classification_metrics(y_true, y_pred)
cal_metrics = compute_calibration_metrics(y_true, y_prob, n_bins=5)

print(f"Accuracy:    {clf_metrics['accuracy']:.2f}")     # Expected: 0.88
print(f"ECE (M=5):   {cal_metrics['ece']:.3f}")           # Calibration error
print(f"Brier Score: {cal_metrics['brier_score']:.3f}")   # Mean squared error
```

#### Example B: Uncertainty & Error Detection
```python
import numpy as np
from trustbench.evaluation.uncertainty import compute_predictive_entropy
from trustbench.evaluation.error_detection import compute_binary_detection_curve

# Compute entropy in nats from class probabilities
probs = np.array([[0.99, 0.01], [0.55, 0.45], [0.10, 0.90]])
entropy = compute_predictive_entropy(probs)  # [0.056, 0.688, 0.325] nats

# Evaluate whether entropy identifies prediction mistakes
errors = np.array([0, 1, 0])
curve = compute_binary_detection_curve(errors, entropy)
print(f"Error Detection AUROC: {curve['auroc']:.2f}")     # Expected: 1.00 (perfect separation)
```

#### Example C: Selective Prediction (AURC)
```python
import numpy as np
from trustbench.evaluation.selective import compute_risk_coverage

labels = np.array([0, 0, 0, 1, 1])
predictions = np.array([0, 0, 0, 0, 0])  # 2 errors
confidences = np.array([0.95, 0.90, 0.85, 0.30, 0.20])

res = compute_risk_coverage(labels, predictions, confidences)
print(f"AURC:        {res['aurc']:.4f}")
print(f"Optimal AURC:{res['optimal_aurc']:.4f}")
print(f"Excess AURC: {res['excess_aurc']:.4f}")          # 0.0 indicates perfect ranking
```

#### Example D: Algorithmic Fairness
```python
import numpy as np
from trustbench.evaluation.fairness import compute_fairness_metrics

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

fair = compute_fairness_metrics(y_true, y_pred, groups)
print(f"Demographic Parity Diff: {fair['demographic_parity_difference']:.2f}")
print(f"Disparate Impact Ratio:  {fair['disparate_impact_ratio']:.2f}")
print(f"Equalized Odds Diff:     {fair['equalized_odds_difference']:.2f}")
```
