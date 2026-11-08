"""Example 9: Create and Register a Custom Model Adapter.

Demonstrates implementing the ModelAdapter protocol for a proprietary or
heuristic model and evaluating it with TrustBench.
"""

from typing import Any

import numpy as np

from trustbench import ModelAdapter, evaluate
from trustbench.datasets.memory import InMemoryDatasetAdapter


# 1. Custom Model Adapter implementing the ModelAdapter protocol
class HeuristicThresholdModelAdapter:
    """Classifies inputs based on a simple linear threshold rule."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold

    @property
    def num_classes(self) -> int:
        return 2

    @property
    def supports_predict_proba(self) -> bool:
        return True

    @property
    def supports_predict_logits(self) -> bool:
        return True

    def predict(self, inputs: Any) -> np.ndarray:
        arr = np.asarray(inputs).squeeze()
        return (arr > self.threshold).astype(int)

    def predict_logits(self, inputs: Any) -> np.ndarray:
        arr = np.asarray(inputs).squeeze()
        # Distance from threshold as pseudo-logit
        diff = arr - self.threshold
        return np.column_stack([-diff, diff])

    def predict_proba(self, inputs: Any) -> np.ndarray:
        logits = self.predict_logits(inputs)
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)


def main():
    print("--- Example 9: Custom Model Adapter ---")

    # Verify adapter conforms to ModelAdapter runtime protocol
    model = HeuristicThresholdModelAdapter(threshold=0.5)
    assert isinstance(model, ModelAdapter), "Model must satisfy ModelAdapter protocol."

    # Define test dataset
    x = np.array([0.1, 0.4, 0.6, 0.9])
    y = np.array([0, 0, 1, 1])
    dataset = InMemoryDatasetAdapter(inputs=x, labels=y, sample_ids=["s1", "s2", "s3", "s4"])

    # Evaluate
    result = evaluate(
        model=model,
        dataset=dataset,
        metrics=["classification", "calibration"],
    )

    acc = result.metrics["classification"]["accuracy"]
    brier = result.metrics["calibration"]["brier_score"]
    print(f"Classification Accuracy: {acc:.2f}")
    print(f"Brier Score:             {brier:.4f}")
    assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
    assert brier >= 0.0, f"Brier score must be non-negative: {brier}"
    print("✓ Custom model adapter verified and evaluated successfully.")


if __name__ == "__main__":
    main()
