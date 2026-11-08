<div align="center">

# TrustBench

**ML Reliability, Robustness, Uncertainty & Fairness Evaluation Framework**

[![PyPI version](https://img.shields.io/badge/pypi-v0.2.0-blue.svg)](https://pypi.org/project/trustbench/)
[![Python Versions](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://pypi.org/project/trustbench/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![CI Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](.github/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-194%20passed-success.svg)](tests/)

</div>

---

TrustBench is a framework-independent machine learning evaluation library designed to measure how reliable models remain under distribution shifts, input corruptions, calibration failures, and algorithmic bias.

Traditional model evaluation focuses almost exclusively on accuracy. TrustBench investigates whether models remain reliable in deployment:
- **Calibration**: Does predicted confidence reflect the true probability of correctness?
- **Robustness**: How gracefully does performance degrade under physical, digital, and environmental corruptions?
- **Uncertainty & Error Detection**: Can predictive entropy and free energy identify incorrect predictions before downstream failures occur?
- **Selective Prediction**: What error rate is achieved when models defer low-confidence decisions?
- **Multi-Framework**: Evaluate PyTorch, Scikit-learn, ONNX Runtime, TensorFlow/Keras, and JAX models using identical contracts.
- **Multi-Task**: Native evaluators for Classification, Regression, Information Retrieval, and Semantic Segmentation.

---

## Architecture Overview

```mermaid
flowchart TD
    UserConfig["trustbench.yaml / Python API"] --> TB["TrustBench Orchestrator"]
    
    subgraph FrameworkAdapters ["Framework Adapters (Optional)"]
        Torch["PyTorch (nn.Module)"]
        SKLearn["Scikit-learn (Estimators)"]
        ONNX["ONNX Runtime (.onnx)"]
        Keras["TensorFlow / Keras"]
        JAX["JAX / Flax"]
    end
    
    subgraph TaskPipelines ["Task-Specific Evaluators"]
        Cls["Classification\n(Accuracy, Macro-F1, Confusion Matrix)"]
        Calib["Calibration & Reliability\n(ECE, AdaECE, Brier, Temperature)"]
        Unc["Uncertainty & Detection\n(Entropy, Energy, AUROC, AUPRC)"]
        Sel["Selective Prediction\n(Risk-Coverage, AURC)"]
        Rob["Robustness Suite\n(Noise, Blur, Brightness 1-5)"]
        Reg["Regression Suite\n(MAE, RMSE, R², 95% PICP)"]
        Ret["Retrieval Suite\n(Recall@K, MRR, nDCG@K)"]
        Seg["Segmentation Suite\n(mIoU, Dice, Pixel Acc)"]
        Fair["Fairness Suite\n(Demographic Parity, Disparate Impact)"]
    end
    
    subgraph StorageReporting ["Storage & Reporting"]
        FSStore["Filesystem Store\n(Atomic Writes & Checksums)"]
        SQLiteStore["SQLite & SQLAlchemy\n(Queryable Registry)"]
        HTMLRep["Interactive HTML Report\n(Zero-Dependency Plotly)"]
        JSONRep["Versioned JSON Schema\n(v1.0 Lineage)"]
    end

    FrameworkAdapters --> TB
    TB --> TaskPipelines
    TaskPipelines --> StorageReporting
```

---

## Installation

TrustBench features a lightweight core with optional framework groups:

```bash
# Core installation (NumPy, SciPy, Scikit-learn, Click, Pydantic)
pip install trustbench

# For PyTorch model evaluation
pip install "trustbench[pytorch]"

# For ONNX Runtime inference
pip install "trustbench[onnx]"

# Full installation (all frameworks, storage, and visualization)
pip install "trustbench[all]"
```

See the [Installation Guide](docs/installation.md) for detailed platform requirements.

---

## 5-Minute Quickstart

### 1. Command Line Interface (CLI)

```bash
# Initialize a starter configuration
trustbench init

# Validate configuration paths and settings
trustbench validate trustbench.yaml

# Run evaluation and generate reports
trustbench evaluate trustbench.yaml
```

### 2. Python API

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from trustbench import TrustBench, InMemoryDatasetAdapter

# 1. Train or load your model
X_train = np.random.randn(200, 4)
y_train = np.random.randint(0, 2, size=200)
clf = LogisticRegression().fit(X_train, y_train)

# 2. Prepare test dataset
X_test = np.random.randn(50, 4)
y_test = np.random.randint(0, 2, size=50)
dataset = InMemoryDatasetAdapter(inputs=X_test, labels=y_test)

# 3. Evaluate reliability
bench = TrustBench(model=clf, dataset=dataset)
result = bench.evaluate()

print(f"Accuracy:    {result.metrics['classification']['accuracy']:.2%}")
print(f"Brier Score: {result.metrics['calibration']['brier_score']:.4f}")
print(f"ECE:         {result.metrics['calibration']['ece']:.4f}")
```

---

## Multi-Task Evaluation

### Continuous Regression Evaluation
```python
from trustbench import evaluate_regression, InMemoryDatasetAdapter
from sklearn.linear_model import Ridge
import numpy as np

model = Ridge().fit(X_train, y_train)
dataset = InMemoryDatasetAdapter(inputs=X_test, labels=y_test)

result = evaluate_regression(model, dataset)
print(f"MAE:  {result.metrics['mae']:.3f}")
print(f"RMSE: {result.metrics['rmse']:.3f}")
print(f"R²:   {result.metrics['r2']:.3f}")
```

### Information Retrieval Evaluation
```python
from trustbench import evaluate_retrieval

ground_truth = {"q1": ["docA", "docB"], "q2": ["docC"]}
retrieved = {"q1": ["docA", "docX", "docB"], "q2": ["docY", "docC"]}

res = evaluate_retrieval(ground_truth, retrieved, k_list=(1, 3))
print(f"MRR:      {res.metrics['mrr']:.3f}")
print(f"Recall@1: {res.metrics['recall@1']:.3f}")
print(f"nDCG@3:   {res.metrics['ndcg@3']:.3f}")
```

### Semantic Segmentation Evaluation
```python
from trustbench import evaluate_segmentation

result = evaluate_segmentation(segmentation_model, segmentation_dataset, num_classes=21)
print(f"Mean IoU:       {result.metrics['miou']:.3f}")
print(f"Dice (F1):      {result.metrics['dice']:.3f}")
print(f"Pixel Accuracy: {result.metrics['pixel_accuracy']:.2%}")
```

---

## Key Metrics Reference

| Category | Metrics Implemented |
|---|---|
| **Classification** | Accuracy, Balanced Accuracy, Precision, Recall, Macro-F1, Confusion Matrix |
| **Calibration** | Expected Calibration Error (ECE), Adaptive ECE, Maximum Calibration Error (MCE), Brier Score, NLL |
| **Uncertainty** | Predictive Shannon Entropy, Maximum Softmax Probability (MSP), Free Energy |
| **Error Detection** | AUROC, AUPRC, False Positive Rate at 95% TPR (FPR@95%TPR), Minimum Detection Error |
| **Selective Prediction** | Risk-Coverage Trade-off Curves, Area Under Risk-Coverage (AURC), Excess AURC |
| **Robustness** | Gaussian Noise, Blur, Brightness, Defocus, Contrast, Corrupted Accuracy Drop |
| **Regression** | MAE, MSE, RMSE, $R^2$, MedAE, MAPE, Max Error, 95% Interval Coverage (PICP, MPIW) |
| **Retrieval** | Mean Reciprocal Rank (MRR), Mean Average Precision (MAP), Recall@K, Precision@K, nDCG@K |
| **Segmentation** | Mean IoU (mIoU), Dice (F1), Pixel Accuracy, Frequency-Weighted IoU (FWIoU) |
| **Fairness** | Demographic Parity Difference, Disparate Impact Ratio, Equalized Odds Difference |

See [Metrics Documentation](docs/metrics.md) for mathematical derivations.

---

## Tutorials & Examples

Ready-to-run examples are available in the [`examples/`](examples/) directory:

1. **[01_evaluate_pytorch_model.py](examples/01_evaluate_pytorch_model.py)**: Evaluate custom PyTorch `nn.Module`.
2. **[02_evaluate_cifar10.py](examples/02_evaluate_cifar10.py)**: End-to-end CIFAR-10 ResNet evaluation via configuration.
3. **[03_robustness_benchmark.py](examples/03_robustness_benchmark.py)**: Evaluate accuracy degradation across noise & corruptions.
4. **[04_analyze_calibration.py](examples/04_analyze_calibration.py)**: Compute ECE and fit validation temperature scaling.
5. **[05_evaluate_uncertainty.py](examples/05_evaluate_uncertainty.py)**: Predict error detection AUROC and risk-coverage curves.
6. **[06_compare_checkpoints.py](examples/06_compare_checkpoints.py)**: Quantify statistical differentials between model checkpoints.
7. **[07_save_and_reproduce_experiment.py](examples/07_save_and_reproduce_experiment.py)**: Atomic experiment persistence and checksum verification.
8. **[08_create_custom_metric.py](examples/08_create_custom_metric.py)**: Register custom evaluation scoring functions.
9. **[09_create_custom_model_adapter.py](examples/09_create_custom_model_adapter.py)**: Implement the runtime `ModelAdapter` protocol.

---

## Documentation

Full documentation is available in the [`docs/`](docs/) directory:
- [Installation Guide](docs/installation.md)
- [Quickstart Guide](docs/quickstart.md)
- [Python API Reference](docs/api.md)
- [CLI Reference](docs/cli.md)
- [Configuration Guide](docs/configuration.md)
- [Model Adapters](docs/model-adapters.md)
- [Dataset Adapters](docs/dataset-adapters.md)
- [Metrics Documentation](docs/metrics.md)
- [Robustness Testing](docs/robustness.md)
- [Experiment Tracking](docs/experiments.md)
- [Reporting & Visualization](docs/reporting.md)
- [Plugin Development](docs/plugins.md)
- [Troubleshooting & FAQ](docs/troubleshooting.md)

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and our [Code of Conduct](CODE_OF_CONDUCT.md).

```bash
# Run tests
pytest

# Check code style
ruff check .

# Static type analysis
mypy src/trustbench/core src/trustbench/reporting src/trustbench/metrics
```

---

## License

TrustBench is open-source software licensed under the [Apache License, Version 2.0](LICENSE).
