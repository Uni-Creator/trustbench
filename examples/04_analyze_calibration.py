"""Example 4: Analyze Model Calibration and Temperature Scaling.

Demonstrates computing Expected Calibration Error (ECE), Brier Score,
reliability diagram data, and fitting temperature scaling to uncalibrated logits.

NOTE on temperature scaling with synthetic data
-----------------------------------------------
Temperature scaling minimises negative log-likelihood (NLL) on a *held-out*
validation set.  The optimiser searches in the range (0.01, 10.0].  When
logits are drawn from N(0, sigma²) with sigma=3 the resulting softmax
distributions are very *overconfident* (peaky).  Flattening such distributions
requires a large temperature.  Reaching the upper bound (T=10.0) is therefore
expected for high-variance synthetic logits and does **not** indicate a bug
in the implementation.

For this reason the example deliberately uses a modest sigma (sigma=1.5) for
the *main* demonstration and shows the boundary-reaching behaviour separately
in its own section.
"""

import numpy as np

from trustbench.evaluation.calibration import (
    compute_adaptive_ece,
    compute_calibration_metrics,
    compute_reliability_diagram_data,
)
from trustbench.evaluation.uncertainty import apply_temperature_scaling, fit_temperature


def main():
    print("--- Example 4: Model Calibration & Reliability Analysis ---")
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # 1. Simulate uncalibrated logits (moderate variance, sigma=1.5)
    # ------------------------------------------------------------------ #
    n_total = 300
    n_classes = 5
    sigma = 1.5
    logits_all = rng.standard_normal((n_total, n_classes)) * sigma
    labels_all = rng.integers(0, n_classes, size=(n_total,))

    # Split into calibration (fit) set and evaluation set
    split = n_total // 2  # 150 / 150
    logits_cal, logits_eval = logits_all[:split], logits_all[split:]
    labels_cal, labels_eval = labels_all[:split], labels_all[split:]

    # Softmax probabilities for the evaluation set (before calibration)
    exp_logits_eval = np.exp(logits_eval - np.max(logits_eval, axis=1, keepdims=True))
    probs_eval = exp_logits_eval / np.sum(exp_logits_eval, axis=1, keepdims=True)

    # ------------------------------------------------------------------ #
    # 2. Compute initial calibration metrics on the evaluation set
    # ------------------------------------------------------------------ #
    calib = compute_calibration_metrics(labels_eval, probs_eval, n_bins=10)
    adapt_ece = compute_adaptive_ece(labels_eval, probs_eval, n_bins=10)

    uncal_ece = calib["ece"]
    uncal_brier = calib["brier_score"]
    print(f"Uncalibrated ECE (equal-width):  {uncal_ece:.4f}")
    print(f"Adaptive ECE (equal-frequency):  {adapt_ece:.4f}")
    print(f"Brier Score:                     {uncal_brier:.4f}")

    # Validity assertions
    assert 0.0 <= uncal_ece <= 1.0, f"ECE out of range: {uncal_ece}"
    assert 0.0 <= adapt_ece <= 1.0, f"Adaptive ECE out of range: {adapt_ece}"
    assert uncal_brier >= 0.0, f"Brier score must be non-negative: {uncal_brier}"

    # ------------------------------------------------------------------ #
    # 3. Fit optimal temperature on the *calibration* split (not eval set)
    # ------------------------------------------------------------------ #
    optimal_temp = fit_temperature(logits_cal, labels_cal)
    print(f"\nFitted Optimal Temperature:      {optimal_temp:.3f}")
    assert optimal_temp > 0.0, "Temperature must be positive."

    # ------------------------------------------------------------------ #
    # 4. Apply temperature scaling to the *evaluation* split
    # ------------------------------------------------------------------ #
    calibrated_probs, _ = apply_temperature_scaling(logits_eval, optimal_temp)
    post_calib = compute_calibration_metrics(labels_eval, calibrated_probs, n_bins=10)
    cal_ece = post_calib["ece"]
    cal_brier = post_calib["brier_score"]
    print(f"Calibrated ECE:                  {cal_ece:.4f}")
    print(f"Calibrated Brier Score:          {cal_brier:.4f}")

    # With a proper calibration/eval split, temperature scaling should improve ECE
    # on the held-out set.  For random labels this improvement may be modest.
    assert 0.0 <= cal_ece <= 1.0, f"Calibrated ECE out of range: {cal_ece}"
    assert cal_brier >= 0.0, "Calibrated Brier score must be non-negative."

    # ------------------------------------------------------------------ #
    # 5. Reliability diagram bin data (after calibration)
    # ------------------------------------------------------------------ #
    diagram_data = compute_reliability_diagram_data(labels_eval, calibrated_probs, n_bins=5)
    print(f"\nReliability Diagram Bins: {len(diagram_data['bin_accuracies'])} bins.")
    assert len(diagram_data["bin_accuracies"]) == 5

    # ------------------------------------------------------------------ #
    # 6. Illustrate boundary-reaching behaviour with sigma=3 logits
    # ------------------------------------------------------------------ #
    print(
        "\n[Boundary behaviour] High-variance logits (sigma=3) → temperature may hit upper bound:"
    )
    logits_hv = rng.standard_normal((200, 5)) * 3.0
    labels_hv = rng.integers(0, 5, size=(200,))
    temp_hv = fit_temperature(logits_hv, labels_hv)
    print(
        f"  Fitted temperature: {temp_hv:.3f}"
        "  (reaching 10.0 is expected for very overconfident logits)"
    )

    print("✓ Calibration analysis completed successfully.")


if __name__ == "__main__":
    main()
