"""Example 1: Evaluate a Custom PyTorch Model using TrustBench.

Demonstrates wrapping an nn.Module with PyTorchModelAdapter, defining a custom
dataset adapter, and computing classification, calibration, and uncertainty metrics.
"""

import numpy as np
from torch import nn

from trustbench import TrustBench
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.models.pytorch import PyTorchModelAdapter


def main():
    print("--- Example 1: Custom PyTorch Model Evaluation ---")

    # 1. Define a synthetic PyTorch neural network
    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 3),  # 3-class classification
    )
    model.eval()

    # 2. Wrap model with PyTorchModelAdapter
    adapter = PyTorchModelAdapter(model, num_classes=3, device="cpu")

    # 3. Create synthetic test dataset
    np.random.seed(42)
    x_test = np.random.randn(100, 8).astype(np.float32)
    y_test = np.random.randint(0, 3, size=(100,))
    sample_ids = [f"sample_{i}" for i in range(100)]

    dataset = InMemoryDatasetAdapter(inputs=x_test, labels=y_test, sample_ids=sample_ids)

    # 4. Evaluate with TrustBench high-level interface
    bench = TrustBench(
        model=adapter,
        dataset=dataset,
        metrics=["classification", "calibration", "uncertainty"],
        class_names=["Class A", "Class B", "Class C"],
    )

    result = bench.evaluate()

    # 5. Inspect structured evaluation output
    acc = result.metrics["classification"]["accuracy"]
    f1 = result.metrics["classification"]["macro_f1"]
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {f1:.4f}")
    assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
    assert 0.0 <= f1 <= 1.0, f"Macro-F1 out of bounds: {f1}"

    if "calibration" in result.metrics and not result.metrics["calibration"].get("skipped"):
        ece = result.metrics["calibration"]["ece"]
        brier = result.metrics["calibration"]["brier_score"]
        print(f"Expected Calibration Error (ECE): {ece:.4f}")
        print(f"Brier Score: {brier:.4f}")
        assert 0.0 <= ece <= 1.0, f"ECE out of bounds: {ece}"
        assert brier >= 0.0, f"Brier score must be non-negative: {brier}"

    print("✓ PyTorch evaluation completed successfully.")


if __name__ == "__main__":
    main()
