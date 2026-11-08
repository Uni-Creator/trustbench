# TrustBench Documentation

Welcome to the documentation for **TrustBench**, a modern, framework-independent machine learning evaluation library for model reliability, robustness, uncertainty, and fairness.

Traditional model evaluation focuses almost exclusively on test accuracy. TrustBench extends evaluation to answer critical operational questions:
- **Calibration**: Does the model's confidence accurately match its empirical probability of being correct?
- **Robustness**: How severely does performance degrade when inputs suffer from physical, digital, or environmental corruptions?
- **Uncertainty & Error Detection**: Can uncertainty metrics (predictive entropy, free energy) catch incorrect predictions before they cause failures?
- **Selective Prediction**: What risk-coverage trade-off is achievable if low-confidence predictions are deferred?
- **Multi-Task & Multi-Framework**: Does evaluation work seamlessly across Classification, Regression, Information Retrieval, and Semantic Segmentation across PyTorch, Scikit-learn, ONNX, and Keras?

---

## Documentation Contents

- **[Installation](installation.md)**: Package installation, optional dependencies, and hardware setup.
- **[Quickstart](quickstart.md)**: 5-minute walk-through from CLI to Python API.
- **[Python API Reference](api.md)**: Public interfaces, evaluators, and results.
- **[Command Line Interface (CLI)](cli.md)**: `trustbench init`, `validate`, `evaluate`, and options.
- **[Configuration Guide](configuration.md)**: `trustbench.yaml` specification, schemas, and resolution.
- **[Model Adapters](model-adapters.md)**: PyTorch, Scikit-learn, ONNX, Keras, JAX, and custom adapters.
- **[Dataset Adapters](dataset-adapters.md)**: PyTorch Datasets, NumPy memory datasets, and custom loaders.
- **[Metrics Reference](metrics.md)**: Classification, calibration, uncertainty, regression, retrieval, segmentation, and fairness metrics.
- **[Robustness & Corruptions](robustness.md)**: Image corruptions, severity progression, and degradation curves.
- **[Experiment Management](experiments.md)**: Storage backends, lineage tracking, and reproducibility.
- **[Reporting & Visualization](reporting.md)**: Standalone HTML reports, charts, and versioned JSON schemas.
- **[Plugin Development](plugins.md)**: Extending TrustBench with custom metrics, adapters, and entry-points.
- **[Troubleshooting](troubleshooting.md)**: Common errors, GPU configurations, and FAQ.
- **[Contributing](contributing.md)**: Development workflow, coding style, and pull request guidelines.
