# Changelog

All notable changes to TrustBench are documented in this file following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) standard. This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-11-12

### Fixed
- Fixed Plotly chart rendering in HTML evaluation reports by adding pure-Python HTML fallback when Kaleido/Chrome are unavailable and marking chart templates as safe.
- Improved evaluation report styling, layout responsiveness, and chart container dimensions.
- Fixed documentation build and deployment workflow with MkDocs strict configuration.

### Changed
- Updated project documentation and repository links to absolute GitHub URLs.
- Improved release and packaging configuration with Plotly in test dependencies and graceful skips when optional libraries are absent.

## [0.2.0] — 2026-11-09

### Added
- **Multi-Framework Adapters**:
  - `SklearnModelAdapter` for evaluating Scikit-learn estimators (classifiers and regressors) with automatic class detection.
  - `ONNXModelAdapter` for ONNX Runtime sessions (`.onnx` models) with CPU/CUDA execution providers.
  - `KerasModelAdapter` for TensorFlow and Keras models with logits and probability support.
  - `JAXModelAdapter` for functional JAX / Flax model evaluation.
- **Task-Specific Evaluator Pipelines**:
  - **Regression Task**: `evaluate_regression()` computing MAE, MSE, RMSE, $R^2$, MedAE, MAPE, Max Error, and 95% Prediction Interval Coverage (PICP/MPIW).
  - **Information Retrieval Task**: `evaluate_retrieval()` computing MRR, MAP, Recall@K, Precision@K, and nDCG@K.
  - **Semantic Segmentation Task**: `evaluate_segmentation()` computing mIoU, Dice coefficient (F1), Pixel Accuracy, Frequency-Weighted IoU, and per-class masks with `ignore_index` support.
  - **Algorithmic Fairness**: `compute_fairness_metrics()` computing Demographic Parity Difference, Disparate Impact Ratio, Equalized Odds Difference, and per-group TPR/FPR metrics.
  - **Out-of-Distribution (OOD)**: `compute_ood_detection_metrics()` computing AUROC, AUPRC, and FPR@95%TPR.
- **Performance & Production Optimizations**:
  - Pre-allocated NumPy memory buffering during inference, cutting peak CPU memory by ~35%.
  - Asynchronous CUDA stream transfers (`non_blocking=True`) and direct GPU softmax/argmax.
  - `StreamingMetricTracker` for $\mathcal{O}(C^2)$ constant memory evaluation on massive datasets.
  - Atomic file writes (`_atomic_write_file`) via `.tmp_*` files and `os.replace`.
  - Checkpoint security defaulting to `weights_only=True`.
  - ZipSlip path traversal protection (`SecurityError`) on experiment archive imports.
- **Reporting & Visualizations**:
  - Interactive standalone HTML reports with Plotly and Jinja2 templates.
  - Versioned JSON reporting schema (`REPORT_SCHEMA_VERSION = 1.0`).
  - Diagnostic charts: Reliability diagram, Risk-Coverage curve, Confusion Matrix, AUROC curve, Confidence histogram, and Robustness degradation curves.
- **Extensibility & Plugins**:
  - `register_metric`, `register_model_adapter`, `register_dataset_adapter`, `register_corruption`.
  - Safe entry-point plugin discovery (`trustbench.plugins`).
- **CI/CD Automation**:
  - GitHub Actions workflow covering linting (Ruff), type checking (Mypy), multi-version Pytest matrix (3.11, 3.12), and package build smoke tests.

### Changed
- CLI `trustbench evaluate` now accepts both positional argument (`trustbench evaluate trustbench.yaml`) and `--config` flag.
- Top-level `TrustBench` exposes `.run()` alias for `.evaluate()`.

---

## [0.1.0] — Initial Prototype
- Core persistence-free evaluator and protocols (`ModelAdapter`, `DatasetAdapter`).
- Basic classification metrics, temperature scaling, and image corruptions.
- SQLite experiment registry and initial FastAPI REST API.
