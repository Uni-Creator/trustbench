"""Example 8: Create and Register a Custom Metric in TrustBench.

Demonstrates using @register_metric to register a custom evaluation metric
and using it in TrustBench evaluation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from trustbench import evaluate, register_metric
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.sklearn import SklearnModelAdapter


# 1. Register custom metric with decorator
@register_metric("geometric_mean_accuracy")
def compute_geometric_mean_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Compute geometric mean accuracy across binary classes."""
    preds = np.asarray(predictions)
    targs = np.asarray(targets)

    # Class 0 recall
    mask0 = targs == 0
    rec0 = float(np.mean(preds[mask0] == 0)) if np.any(mask0) else 1.0

    # Class 1 recall
    mask1 = targs == 1
    rec1 = float(np.mean(preds[mask1] == 1)) if np.any(mask1) else 1.0

    return float(np.sqrt(rec0 * rec1))


def main():
    print("--- Example 8: Custom Metric Registration ---")

    # 2. Fit small model
    x = np.array([[1.0], [2.0], [-1.0], [-2.0]])
    y = np.array([1, 1, 0, 0])
    clf = LogisticRegression().fit(x, y)
    model = SklearnModelAdapter(clf)
    dataset = InMemoryDatasetAdapter(inputs=x, labels=y, sample_ids=["1", "2", "3", "4"])

    # 3. Evaluate with standard metrics + registered custom metric
    result = evaluate(
        model=model,
        dataset=dataset,
        metrics=["classification", "geometric_mean_accuracy"],
    )

    acc = result.metrics["classification"]["accuracy"]
    geom_acc = result.metrics["geometric_mean_accuracy"]
    print(f"Standard Accuracy:        {acc:.4f}")
    print(f"Geometric Mean Accuracy:  {geom_acc:.4f}")
    assert 0.0 <= acc <= 1.0, f"Standard accuracy out of bounds: {acc}"
    assert 0.0 <= geom_acc <= 1.0, f"Geometric mean accuracy out of bounds: {geom_acc}"
    print("✓ Custom metric registered and evaluated successfully.")


if __name__ == "__main__":
    main()
