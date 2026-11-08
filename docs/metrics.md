# Metrics Reference & Mathematical Definitions

TrustBench implements a comprehensive battery of evaluation metrics across classification, calibration, uncertainty, selective prediction, regression, retrieval, segmentation, and fairness.

---

## 1. Classification Metrics
- **Accuracy**: $\frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$.
- **Balanced Accuracy**: Macro-average of recall across all classes, protecting against class imbalance.
- **Precision, Recall, Macro-F1**: Per-class and unweighted macro-averaged harmonic mean.
- **Confusion Matrix**: Full $C \times C$ cross-tabulation of true vs predicted labels.

---

## 2. Calibration Metrics
- **Expected Calibration Error (ECE)**:
  Equal-width partitioning into $M$ confidence bins:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Adaptive ECE**:
  Partitions samples into $M$ quantile bins of equal sample size, eliminating empty-bin artifacts.
- **Maximum Calibration Error (MCE)**:
  Worst-case calibration gap $\max_m |\text{acc}(B_m) - \text{conf}(B_m)|$.
- **Brier Score**:
  Mean squared error between predicted probabilities and one-hot ground truth:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C (p_{ic} - y_{ic})^2$$
- **Negative Log-Likelihood (NLL)**:
  Cross-entropy loss $-\frac{1}{N} \sum_{i=1}^N \log(p_{i, y_i})$.

---

## 3. Uncertainty & Error Detection
- **Predictive Shannon Entropy**:
  $$H(p) = -\sum_{c=1}^C p_c \log(p_c)$$
- **Free Energy Score**:
  $$E(x; T) = -T \cdot \text{logsumexp}\left(\frac{f(x)}{T}\right)$$
- **Error Detection AUROC**:
  Area Under the ROC curve treating prediction error ($\hat{y} \neq y$) as the positive binary class.
- **FPR at 95% TPR**:
  False Positive Rate when detection catches $\ge 95\%$ of actual errors.

---

## 4. Selective Prediction (Risk-Coverage)
- **Risk-Coverage Curve**:
  Evaluates model error rate when the lowest-confidence samples are rejected across coverage levels $c \in [0.5, 1.0]$.
- **Area Under Risk-Coverage Curve (AURC)**:
  Cumulative risk across all retention thresholds.
- **Excess AURC**:
  $$\text{Excess AURC} = \text{AURC} - \text{AURC}^*$$
  where $\text{AURC}^*$ is the theoretical optimal curve achieved by a perfect confidence ranker.

---

## 5. Regression Metrics
- **MAE**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$.
- **MSE & RMSE**: $\frac{1}{N} \sum (y_i - \hat{y}_i)^2$ and its square root.
- **$R^2$**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$.
- **MedAE & Max Error**: Median absolute error and maximum residual.
- **MAPE**: Mean absolute percentage error.
- **PICP & MPIW**: Prediction Interval Coverage Probability and Mean Prediction Interval Width at 95%.

---

## 6. Information Retrieval Metrics
- **Mean Reciprocal Rank (MRR)**: Average reciprocal rank of the first relevant document.
- **Mean Average Precision (MAP)**: Mean precision across all relevant rank positions.
- **Recall@K & Precision@K**: Fraction of relevant items retrieved and precision at threshold $K$.
- **nDCG@K**: Normalized Discounted Cumulative Gain at threshold $K$.

---

## 7. Semantic Segmentation Metrics
- **mIoU**: Mean Intersection-over-Union across classes $\frac{1}{C} \sum_c \frac{\text{TP}_c}{\text{TP}_c + \text{FP}_c + \text{FN}_c}$.
- **Dice**: Mean harmonic score $\frac{2 \cdot \text{TP}}{2 \cdot \text{TP} + \text{FP} + \text{FN}}$.
- **Pixel Accuracy**: Ratio of correctly classified pixels.
- **FWIoU**: Frequency-Weighted IoU weighted by class pixel prevalence.

---

## 8. Algorithmic Fairness Metrics
- **Demographic Parity Difference**: $\max_a P(\hat{Y}=1|A=a) - \min_a P(\hat{Y}=1|A=a)$.
- **Disparate Impact Ratio**: $\frac{\min_a P(\hat{Y}=1|A=a)}{\max_a P(\hat{Y}=1|A=a)}$.
- **Equalized Odds Difference**: Maximum disparity in TPR and FPR across protected groups.
