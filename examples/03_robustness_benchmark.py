"""Example 3: Run Robustness Benchmarks across Progressive Corruptions.

Demonstrates evaluating model degradation under systematic corruptions
(Gaussian noise, blur, brightness) across multiple severity levels (1 to 5).
"""

import tempfile
from pathlib import Path

from trustbench.robustness.evaluator import RobustnessEvaluator


def main():
    print("--- Example 3: Robustness & Corruption Benchmark ---")

    ckpt_path = Path("checkpoint/ckpt.pth")
    if not ckpt_path.exists():
        print(f"Checkpoint {ckpt_path} not found. Skipping live inference run.")
        return

    with tempfile.TemporaryDirectory() as tmp_dir:
        evaluator = RobustnessEvaluator(
            checkpoint_path=str(ckpt_path),
            model_name="ResNet18",
            data_dir="dataset",
            output_dir=tmp_dir,
            batch_size=128,
        )

        # Evaluate Gaussian noise at severity 1 and severity 3
        print("Evaluating Gaussian noise (severity 1)...")
        metrics_sev1, _ = evaluator.evaluate_condition(
            corruption_type="gaussian_noise",
            severity=1,
        )
        acc1 = metrics_sev1["classification"]["accuracy"] * 100.0
        print(f"  Accuracy at Severity 1: {acc1:.2f}%")
        assert 0.0 <= acc1 <= 100.0, f"Accuracy out of bounds: {acc1}"

        print("Evaluating Gaussian noise (severity 3)...")
        metrics_sev3, _ = evaluator.evaluate_condition(
            corruption_type="gaussian_noise",
            severity=3,
        )
        acc3 = metrics_sev3["classification"]["accuracy"] * 100.0
        print(f"  Accuracy at Severity 3: {acc3:.2f}%")
        assert 0.0 <= acc3 <= 100.0, f"Accuracy out of bounds: {acc3}"

        print(f"\nObserved accuracy drop: {acc1 - acc3:.2f} percentage points.")
        print("✓ Robustness benchmark completed successfully.")


if __name__ == "__main__":
    main()
