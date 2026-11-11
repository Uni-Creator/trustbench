# Quickstart Guide

Get up and running with TrustBench in 5 minutes using either the Command Line Interface (CLI) or the Python API.

---

## 1. Quickstart via CLI

### Step 1: Initialize Configuration
Generate a starter `trustbench.yaml` configuration in your project root:

```bash
trustbench init
```

### Step 2: Validate Configuration
Check that your configuration paths and parameters are valid:

```bash
trustbench validate trustbench.yaml
```

### Step 3: Run Evaluation
Execute evaluation and generate structured artifacts:

```bash
trustbench evaluate trustbench.yaml
```

This will run inference, compute classification, calibration, and uncertainty metrics, and output interactive HTML and machine-readable JSON reports into `artifacts/`.

---

## 2. Quickstart via Python API

### Basic Classification Evaluation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from trustbench import TrustBench, InMemoryDatasetAdapter

# 1. Train or load your model
X_train = np.random.randn(100, 4)
y_train = np.random.randint(0, 2, size=100)
clf = LogisticRegression().fit(X_train, y_train)

# 2. Wrap evaluation dataset
X_test = np.random.randn(50, 4)
y_test = np.random.randint(0, 2, size=50)
dataset = InMemoryDatasetAdapter(inputs=X_test, labels=y_test)

# 3. Evaluate with TrustBench
bench = TrustBench(model=clf, dataset=dataset)
result = bench.evaluate()

print(f"Accuracy: {result.metrics['classification']['accuracy']:.2f}")
print(f"Brier Score: {result.metrics['calibration']['brier_score']:.4f}")
```

### Basic Regression Evaluation

```python
import numpy as np
from sklearn.linear_model import Ridge
from trustbench import evaluate_regression, InMemoryDatasetAdapter

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2.5 * X.squeeze() + np.random.normal(0, 0.5, size=100)
reg = Ridge().fit(X, y)

dataset = InMemoryDatasetAdapter(inputs=X, labels=y)
result = evaluate_regression(reg, dataset)

print(f"MAE: {result.metrics['mae']:.3f}")
print(f"R²:  {result.metrics['r2']:.3f}")
```

> [!NOTE]
> **Prediction Intervals & Uncertainty (PICP / MPIW)**:
> Standard point regressors like `Ridge` produce scalar predictions without variance estimates, so interval metrics (`picp_95`, `mpiw_95`) are omitted. When evaluating probabilistic models (such as Gaussian Processes, deep ensembles, or custom adapters exposing `predict_uncertainty(inputs)`), TrustBench automatically extracts predictive standard deviations $\sigma$ and computes 95% coverage (`picp_95`) and interval width (`mpiw_95`).

