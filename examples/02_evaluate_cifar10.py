"""Example 2: Evaluate CIFAR-10 Model with TrustBench Configuration.

Demonstrates loading a trained ResNet architecture checkpoint and evaluating
it against local CIFAR-10 test data using TrustBench.from_config().
"""

from pathlib import Path

from trustbench import TrustBench


def main():
    print("--- Example 2: CIFAR-10 Model Evaluation ---")
    config_path = Path("trustbench.yaml")

    if not config_path.exists():
        print("Creating default trustbench.yaml configuration...")
        from click.testing import CliRunner

        from trustbench.cli.main import app

        runner = CliRunner()
        runner.invoke(app, ["init", "--output", "trustbench.yaml"])

    print(f"Loading configuration from {config_path}...")
    bench = TrustBench.from_config(str(config_path))

    print("Running evaluation on CIFAR-10...")
    result = bench.evaluate()

    acc = result.metrics["classification"]["accuracy"]
    f1 = result.metrics["classification"]["macro_f1"]
    print(f"Classification Accuracy: {acc * 100:.2f}%")
    print(f"Macro F1 Score:          {f1:.4f}")
    assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"
    assert 0.0 <= f1 <= 1.0, f"Macro-F1 out of bounds: {f1}"

    if "calibration" in result.metrics:
        print(f"ECE:                     {result.metrics['calibration'].get('ece', 'N/A')}")
    print("✓ CIFAR-10 evaluation complete.")


if __name__ == "__main__":
    main()
