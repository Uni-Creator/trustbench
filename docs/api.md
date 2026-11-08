# Python API Reference

The TrustBench Python API is organized into top-level evaluation orchestrators, protocols, task-specific pipelines, and metric modules.

---

## 1. Top-Level Evaluation Interface

### `TrustBench`
High-level orchestrator for model and dataset evaluation.

```python
from trustbench import TrustBench

tb = TrustBench(
    model,                      # ModelAdapter, nn.Module, sklearn estimator, or ONNX session
    dataset,                    # DatasetAdapter, torch Dataset, or DataLoader
    metrics=None,               # Sequence of metric groups (default: classification, calibration, uncertainty)
    batch_size=100,             # Batch size for evaluation
    class_names=None,           # Optional list of human-readable class names
    seed=None,                  # Random seed for reproducibility
    output_dir=None,            # Optional path to persist artifacts
)

result = tb.evaluate()          # or tb.run()
```

### `TrustBench.from_config(config_path)`
Instantiates and configures a `TrustBench` instance directly from a `trustbench.yaml` file.

```python
bench = TrustBench.from_config("trustbench.yaml")
result = bench.evaluate()
```

---

## 2. Core Evaluators

### `evaluate(model, dataset, ...)`
Persistence-free core evaluation function for classification models.

```python
from trustbench import evaluate

result = evaluate(
    model,
    dataset,
    metrics=("classification", "calibration", "uncertainty"),
    batch_size=100,
    class_names=None,
    seed=42,
)
```

### `evaluate_regression(model, dataset, ...)`
Evaluates regression models on continuous targets.

```python
from trustbench import evaluate_regression

result = evaluate_regression(
    model,
    dataset,
    batch_size=100,
    seed=42,
)
# Returns RegressionResult with MAE, MSE, RMSE, R², MedAE, MAPE, and interval metrics
```

### `evaluate_retrieval(ground_truth, ranked_predictions, ...)`
Evaluates search, ranking, and retrieval pipelines.

```python
from trustbench import evaluate_retrieval

result = evaluate_retrieval(
    ground_truth,              # Mapping or list of relevant doc IDs
    ranked_predictions,        # Mapping or list of ranked retrieved doc IDs
    k_list=(1, 3, 5, 10),      # Cutoffs
)
# Returns RetrievalResult with MRR, MAP, Recall@K, Precision@K, nDCG@K
```

### `evaluate_segmentation(model, dataset, ...)`
Evaluates pixel-level semantic segmentation predictions.

```python
from trustbench import evaluate_segmentation

result = evaluate_segmentation(
    model,
    dataset,
    num_classes=21,
    ignore_index=255,
)
# Returns SegmentationResult with mIoU, Dice, Pixel Accuracy, and FWIoU
```

---

## 3. Results Schema

### `EvaluationResult`
- `metrics`: Dictionary of computed metrics categorized by group.
- `predictions`: NumPy 1D array of class predictions $(N,)$.
- `labels`: NumPy 1D array of ground truth labels $(N,)$.
- `sample_ids`: NumPy array of sample identifiers.
- `probabilities`: Optional 2D array of class probabilities $(N, C)$.
- `logits`: Optional 2D array of raw model outputs $(N, C)$.
- `summary`: Dictionary shortcut to primary summary metrics.
- `save(output_dir)`: Atomically saves `manifest.json`, `metadata.json`, `metrics.json`, and `predictions.npz`.

### `RegressionResult`
- `metrics`: Dictionary of regression metrics (`mae`, `mse`, `rmse`, `r2`, etc.).
- `predictions`: Continuous model predictions $(N,)$.
- `targets`: Continuous ground truth targets $(N,)$.
- `residuals`: Difference between targets and predictions $(y - \hat{y})$.
- `uncertainties`: Optional standard deviations / confidence intervals.

### `RetrievalResult`
- `metrics`: Ranking metrics (`mrr`, `map`, `recall@k`, `precision@k`, `ndcg@k`).
- `query_ids`: Identifiers for evaluated queries.
- `ground_truth`: Relevant document mapping.
- `ranked_predictions`: Retrieved document rankings.

### `SegmentationResult`
- `metrics`: Segmentation metrics (`miou`, `dice`, `pixel_accuracy`, `fwiou`, `per_class_iou`).
- `predictions`: Pixel class masks $(N, H, W)$.
- `ground_truth`: Ground truth masks $(N, H, W)$.
