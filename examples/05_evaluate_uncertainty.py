"""Example 5: Evaluate Uncertainty & Error Detection.

Demonstrates extracting Maximum Softmax Probability (MSP), predictive entropy,
free energy scores, AUROC for error detection, and selective prediction risk-coverage.

NOTE on AUROC with random predictions
--------------------------------------
When both the model predictions and the ground-truth labels are random there
is *no signal* linking uncertainty scores to actual errors.  The expected
AUROC under these conditions is 0.5 ± sampling noise.  Observing a value
close to 0.5 (e.g. 0.47-0.53) is correct and expected behaviour; it is NOT
an indication that the score direction is inverted or that the implementation
is wrong.

The example includes a second, deterministic scenario that creates clear signal
(high-entropy samples are all wrong; low-entropy samples are all correct) and
asserts AUROC ≥ 0.9 to validate the implementation direction.
"""

import numpy as np

from trustbench.evaluation.error_detection import (
    compute_binary_detection_curve,
)
from trustbench.evaluation.selective import compute_selective_prediction_metrics
from trustbench.evaluation.uncertainty import (
    compute_energy_scores,
    compute_predictive_entropy,
    compute_uncertainty_metrics,
)


def main():
    print("--- Example 5: Uncertainty & Error Detection ---")
    rng = np.random.default_rng(42)

    # ------------------------------------------------------------------ #
    # 1. Random-data scenario (no signal expected)
    # ------------------------------------------------------------------ #
    n_samples = 300
    logits = rng.standard_normal((n_samples, 4))
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    labels = rng.integers(0, 4, size=(n_samples,))
    predictions = np.argmax(probs, axis=1)

    entropy = compute_predictive_entropy(probs)
    compute_energy_scores(logits)
    uncertainty = compute_uncertainty_metrics(probs, logits=logits)

    print(f"Mean Predictive Entropy: {uncertainty['mean_entropy']:.4f}")
    print(f"Mean Max Probability:   {uncertainty['mean_max_probability']:.4f}")
    print(f"Mean Free Energy:        {uncertainty['mean_energy']:.4f}")

    assert uncertainty["mean_entropy"] >= 0.0, "Entropy must be non-negative."
    assert 0.0 <= uncertainty["mean_max_probability"] <= 1.0

    # Error detection on random data: AUROC ≈ 0.5 ± noise is expected
    errors = (predictions != labels).astype(int)
    detection = compute_binary_detection_curve(errors, uncertainty_scores=entropy)
    print(f"\nError Detection AUROC (Entropy, random data): {detection['auroc']:.4f}")
    print("  (Expected ≈ 0.5 ± noise for random predictions – this is correct.)")
    print(f"FPR at 95% TPR:                               {detection['fpr_at_95_tpr']:.4f}")

    assert 0.0 <= detection["auroc"] <= 1.0, "AUROC must be in [0, 1]."
    assert 0.0 <= detection["fpr_at_95_tpr"] <= 1.0

    # ------------------------------------------------------------------ #
    # 2. Deterministic scenario: validate score-direction with known signal
    # ------------------------------------------------------------------ #
    print("\n[Signal validation] 50 certain-correct + 50 uncertain-wrong samples:")
    n_half = 50
    n_classes = 4

    # Group A: correct predictions, low entropy (confident)
    # Create near-one-hot probabilities (class 0 is correct)
    probs_correct = np.zeros((n_half, n_classes))
    probs_correct[:, 0] = 0.97
    probs_correct[:, 1:] = 0.01
    labels_correct = np.zeros(n_half, dtype=int)      # true label = 0
    preds_correct = np.zeros(n_half, dtype=int)        # predicted = 0 ✓

    # Group B: wrong predictions, high entropy (uncertain)
    # Create near-uniform probabilities (class 1 predicted, label is 0 → wrong)
    probs_wrong = np.full((n_half, n_classes), 1.0 / n_classes)
    labels_wrong = np.zeros(n_half, dtype=int)         # true label = 0
    preds_wrong = np.ones(n_half, dtype=int)           # predicted = 1 ✗

    all_probs = np.vstack([probs_correct, probs_wrong])
    all_labels = np.concatenate([labels_correct, labels_wrong])
    all_preds = np.concatenate([preds_correct, preds_wrong])

    all_entropy = compute_predictive_entropy(all_probs)
    all_errors = (all_preds != all_labels).astype(int)
    signal_detection = compute_binary_detection_curve(all_errors, uncertainty_scores=all_entropy)

    print(f"  AUROC (should be ≥ 0.9): {signal_detection['auroc']:.4f}")
    assert signal_detection["auroc"] >= 0.9, (
        f"Expected AUROC ≥ 0.9 for a perfect entropy signal, "
        f"got {signal_detection['auroc']:.4f}"
    )

    # ------------------------------------------------------------------ #
    # 3. Selective prediction (risk-coverage tradeoff)
    # ------------------------------------------------------------------ #
    sel = compute_selective_prediction_metrics(labels, predictions, probabilities=probs)
    msp_stats = sel["max_softmax_probability"]
    print(f"\nArea Under Risk-Coverage (AURC): {msp_stats['aurc']:.4f}")
    print(f"Optimal AURC:                    {msp_stats['optimal_aurc']:.4f}")
    print(f"Excess Risk (AURC - Optimal):    {msp_stats['excess_aurc']:.4f}")

    assert msp_stats["aurc"] >= 0.0
    assert msp_stats["optimal_aurc"] >= 0.0
    assert msp_stats["excess_aurc"] >= 0.0
    assert msp_stats["aurc"] >= msp_stats["optimal_aurc"] - 1e-9, (
        "AURC cannot be less than Optimal AURC."
    )

    print("✓ Uncertainty evaluation completed successfully.")


if __name__ == "__main__":
    main()
