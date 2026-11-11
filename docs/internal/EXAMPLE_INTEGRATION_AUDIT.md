# TrustBench Example Integration Audit

**Date:** 2026-11-09  
**Version:** 0.2.0  
**Scope:** Numerical correctness, API adherence, and regression verification across all 9 tutorial examples (`examples/01_*.py` through `examples/09_*.py`).

---

## Executive Summary

An exhaustive numerical and structural audit was conducted on TrustBench's 9 tutorial examples and their underlying evaluation modules. While all scripts were syntactically executable, the audit investigated three key numerical findings:

1. **Calibration Analysis (`examples/04_analyze_calibration.py`)**: Reached the temperature optimization boundary ($T = 10.000$) and previously lacked train/validation separation.
2. **Uncertainty & Error Detection (`examples/05_evaluate_uncertainty.py`)**: Observed an AUROC of 0.4835, raising questions regarding score direction and random signal behavior.
3. **Checkpoint Comparison (`examples/06_compare_checkpoints.py`)**: Printed raw dictionary keys rather than formatted metric deltas due to an argument and schema mismatch with `ExperimentComparator.compare()`.

All root causes were diagnosed, fixed, and verified without breaking public API contracts or weakening tests. Structural and numerical assertions were added across all 9 example scripts, and 10 new regression tests were added to the test suite (`tests/unit/test_example_integration_audit.py`). All 204 tests pass and Ruff checks pass with zero errors.

---

## Finding 1: Calibration Analysis (`examples/04_analyze_calibration.py`)

### 1.1 Root Cause Analysis

- **Temperature Scaling Objective:**  
  `fit_temperature(val_logits, val_labels)` in `src/trustbench/evaluation/uncertainty.py` minimizes negative log-likelihood (NLL) using `scipy.optimize.minimize(..., method="L-BFGS-B", bounds=[(0.01, 10.0)])`.
- **Upper Bound Reached ($T = 10.000$):**  
  In the original example, logits were simulated as $z \sim \mathcal{N}(0, 3^2)$ with uniform random labels. High-variance logits create extreme overconfidence (softmax probabilities approaching 1.0 even though the accuracy on random labels is only $\approx 20\%$). To flatten these peaky distributions and minimize NLL toward a uniform distribution, the optimizer drives $T \to \infty$. The upper bound of 10.0 is deliberately hit as a safety ceiling. This is mathematically correct behavior for overconfident logits and is not an implementation bug.
- **Methodological Issue in Example:**  
  The original example used the exact same 200 samples to fit the temperature and compute post-calibration metrics, violating the fundamental principle of held-out calibration.

### 1.2 Fixes & Improvements

1. **Held-Out Calibration Split:**  
   Split data into distinct calibration (150 samples) and evaluation (150 samples) subsets. Temperature is fitted strictly on the calibration set and evaluated on the evaluation set.
2. **Moderate Variance Logits + Boundary Demonstration:**  
   The primary demonstration uses realistic logits ($\sigma = 1.5$), demonstrating expected calibration behavior. A secondary section explicitly illustrates the high-variance boundary behavior ($\sigma = 3.0$), documenting why $T = 10.0$ is reached.
3. **Assertions:**  
   Added range assertions on ECE ($[0, 1]$), Adaptive ECE ($[0, 1]$), Brier score ($\ge 0$), and temperature ($> 0$).

---

## Finding 2: Uncertainty & Error Detection (`examples/05_evaluate_uncertainty.py`)

### 1.1 Root Cause Analysis

- **AUROC Value of 0.4835:**  
  The example simulated both logits and labels independently and uniformly at random. In this setting, the model predictions are uncorrelated with true labels (pure noise), and the entropy of the random softmax output provides zero predictive mutual information regarding correctness. The theoretical expected AUROC is $0.50 \pm \text{sampling noise}$. An empirical AUROC of 0.4835 over 300 samples is within normal random deviation ($\sigma \approx 0.035$) and does not indicate an inverted convention.
- **Score Direction Verification:**  
  `compute_binary_detection_curve(y_true_binary, uncertainty_scores)` defines $y = 1$ as errors and $y = 0$ as correct. Higher uncertainty scores indicate higher likelihood of error. Predictive entropy $H(p) \ge 0$ increases as probabilities become uniform (high uncertainty), which correctly aligns with the positive class ($y = 1$).
- **Selective Prediction & Optimal AURC:**  
  `compute_optimal_aurc(n_samples, n_errors)` implements the theoretical minimum risk-coverage area under perfect uncertainty ranking:
  $$\text{AURC}^* = \frac{1}{N} \sum_{k=N-E+1}^N \left(1 - \frac{N-E}{k}\right)$$
  Verified to be monotonically increasing in error count and bounded in $[0, 1]$.

### 1.2 Fixes & Improvements

1. **Documented Noise Behavior:**  
   Clarified in docstrings and console output that an AUROC $\approx 0.5$ on random data is mathematically expected.
2. **Deterministic Signal Validation:**  
   Added a deterministic verification block containing 50 confident-correct samples ($p_0 = 0.97, y = 0, \hat{y} = 0$) and 50 uncertain-wrong samples ($p = 0.25, y = 0, \hat{y} = 1$). Verified that AUROC achieves $1.0000$ and FPR@95%TPR achieves $0.0000$.
3. **Assertions:**  
   Added assertions for AUROC bounds, non-negativity of entropy and energy, and that $\text{AURC} \ge \text{Optimal AURC}$ ($\text{Excess AURC} \ge 0$).

---

## Finding 3: Checkpoint Comparison (`examples/06_compare_checkpoints.py`)

### 3.1 Root Cause Analysis

- **API Calling Convention:**  
  `ExperimentComparator.compare()` signature is:
  `compare(experiments: list[...], conditions: list[str] | None = None, ...)`
  The example originally invoked `comparator.compare(exp_a, exp_b)`, passing `exp_b` as the `conditions` argument rather than passing a list `[exp_a, exp_b]`.
- **Output Schema Mismatch:**  
  `compare()` returns a structured dictionary:
  - `experiment_ids`: list of experiment identifiers
  - `experiment_names`: mapping of id to display name
  - `checkpoints`: mapping of id to checkpoint hash
  - `conditions`: list of evaluated conditions (e.g., `["Clean"]`)
  - `metrics`: list of metric names
  - `comparison_table`: nested dictionary `{condition: {metric: {exp_id: value}}}`
  - `deltas`: nested dictionary `{condition: {metric: {absolute_difference, relative_percent_change, preferred}}}`
  The example was iterating over the top-level keys (`diff.items()`), treating `experiment_names`, `checkpoints`, and `deltas` as metric groups, resulting in empty or nonsensical output.
- **Metric Grouping Convention:**  
  Scalar summary metrics must be placed under `"summary"` within the experiment record's `metrics` dictionary to be evaluated under the `"Clean"` condition.

### 3.2 Fixes & Improvements

1. Passed `[exp_a, exp_b]` as a list to `comparator.compare()`.
2. Formatted experiment inputs with `metrics: {"summary": {...}}`.
3. Rewrote the display logic to cleanly format:
   - Per-condition absolute metric values across both models.
   - Differentials (absolute difference, relative percentage change, preferred candidate).
4. Added assertions verifying delta computations and candidate preference direction.

---

## Finding 4: Assertions Added Across All 9 Examples

Every tutorial script in `examples/` now contains runtime assertions ensuring numerical bounds and structural invariants:

| Script | Added Assertions |
|---|---|
| `examples/01_evaluate_pytorch_model.py` | `0.0 <= accuracy <= 1.0`, `0.0 <= macro_f1 <= 1.0`, `0.0 <= ece <= 1.0`, `brier_score >= 0.0` |
| `examples/02_evaluate_cifar10.py` | `0.0 <= accuracy <= 1.0`, `0.0 <= macro_f1 <= 1.0` |
| `examples/03_robustness_benchmark.py` | `0.0 <= acc1 <= 100.0`, `0.0 <= acc3 <= 100.0` |
| `examples/04_analyze_calibration.py` | `0.0 <= uncal_ece <= 1.0`, `0.0 <= adapt_ece <= 1.0`, `brier >= 0.0`, `optimal_temp > 0`, `0.0 <= cal_ece <= 1.0`, `len(diagram_bins) == 5` |
| `examples/05_evaluate_uncertainty.py` | `entropy >= 0.0`, `0.0 <= max_prob <= 1.0`, `0.0 <= auroc <= 1.0`, `signal_auroc >= 0.9`, `aurc >= optimal_aurc`, `excess_aurc >= 0.0` |
| `examples/06_compare_checkpoints.py` | `"comparison_table" in result`, `len(result["deltas"]) > 0`, `accuracy diff > 0.0`, `ece diff < 0.0` |
| `examples/07_save_and_reproduce_experiment.py` | `retrieved is not None`, artifact checksum integrity `assert is_valid`, artifact record count |
| `examples/08_create_custom_metric.py` | `0.0 <= accuracy <= 1.0`, `0.0 <= geometric_mean_accuracy <= 1.0` |
| `examples/09_create_custom_model_adapter.py` | `isinstance(model, ModelAdapter)`, `0.0 <= accuracy <= 1.0`, `brier_score >= 0.0` |

---

## New Regression Test Suite (`tests/unit/test_example_integration_audit.py`)

10 dedicated unit tests were implemented:

1. `test_hand_computed_calibration_metrics`: Hand-verified Brier score ($0.6500$) and Adaptive ECE ($0.2500$) on a 4-sample synthetic dataset.
2. `test_temperature_scaling_converges_for_well_calibrated_logits`: Verified $T \approx 1.0$ ($0.7 < T < 1.4$) on logits generated by a true logistic model.
3. `test_temperature_scaling_boundary_behavior`: Verified safe optimization bounded at $T \le 10.0$ without NaNs or divergence on extreme overconfident logits.
4. `test_held_out_calibration_separation`: Validated calibration fitting on split A and evaluation on split B.
5. `test_perfect_error_detection_separation`: Validated $\text{AUROC} = 1.0$, $\text{FPR@95\%TPR} = 0.0$, and $\text{Detection Error} = 0.0$ for perfect signal.
6. `test_error_detection_against_sklearn`: Verified exact parity between `compute_binary_detection_curve` AUROC and `sklearn.metrics.roc_auc_score`.
7. `test_optimal_aurc_theoretical_properties`: Validated boundary cases ($E=0 \to 0, E=N \to 1$), monotonicity, and hand calculation ($N=4, E=2 \to 5/24 \approx 0.208333$).
8. `test_selective_prediction_excess_risk`: Verified $\text{Excess AURC} \approx 0.0$ and $\text{AURC} = \text{Optimal AURC}$ under optimal uncertainty ranking.
9. `test_checkpoint_comparison_schema_and_deltas`: Validated schema, relative percentage change, and metric optimization direction (`accuracy` $\to$ higher preferred; `ece` $\to$ lower preferred).
10. `test_checkpoint_comparison_empty_and_single`: Verified graceful handling of empty lists and single-experiment comparisons.

---

## Verification Results

### Test Execution
```bash
$ uv run pytest --tb=short -q
204 passed, 3 warnings in 8.70s
```

### Static Analysis
```bash
$ uv run ruff check .
All checks passed!
```

### Tutorial Script Execution
All 9 examples run to completion with status code 0:
```bash
$ uv run python examples/01_evaluate_pytorch_model.py
$ uv run python examples/02_evaluate_cifar10.py
$ uv run python examples/03_robustness_benchmark.py
$ uv run python examples/04_analyze_calibration.py
$ uv run python examples/05_evaluate_uncertainty.py
$ uv run python examples/06_compare_checkpoints.py
$ uv run python examples/07_save_and_reproduce_experiment.py
$ uv run python examples/08_create_custom_metric.py
$ uv run python examples/09_create_custom_model_adapter.py
```

---

## Known Limitations & Methodological Recommendations

1. **Synthetic vs. Real-World Calibration:**  
   In synthetic experiments with random targets, temperature scaling will typically drive toward the maximum bound to minimize cross-entropy loss against high-entropy labels. When evaluating real models (e.g., ResNet on CIFAR-10), temperatures typically calibrate in the range $T \in [1.1, 2.5]$.
2. **Error Detection on Uninformative Representations:**  
   AUROC measures the ranking capability of uncertainty scores relative to misclassifications. If a model's predictive entropy is uninformative regarding its failure modes, AUROC will be near $0.50$. Users should evaluate multiple uncertainty metrics (MSP, Entropy, Energy, MC Dropout) to identify the best discriminator for their specific task.
3. **Experiment Comparison Input Format:**  
   `ExperimentComparator.compare()` expects a `list` of experiment dictionaries or `ExperimentRecord` objects. Scalar metrics should be nested under `"summary"` for baseline comparison or under condition keys (e.g. corruption names) for multi-condition comparison.
