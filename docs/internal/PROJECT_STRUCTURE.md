# TrustBench Project Structure

This document outlines the final directory structure of TrustBench and describes the responsibility of each retained package, module, and directory.

```
trustbench/
├── .github/                        # CI/CD and open-source governance
│   ├── ISSUE_TEMPLATE/             # Bug report and feature request templates
│   ├── workflows/ci.yml            # Multi-OS, multi-version GitHub Actions workflow
│   └── pull_request_template.md    # Pull request submission checklist
├── benchmarks/                     # Performance and throughput measurement
│   ├── benchmark_performance.py    # Benchmark harness measuring throughput and RAM
│   └── results_baseline.json       # Reference benchmark numbers
├── configs/                        # Example and reference configuration files
│   └── cifar10_resnet18.yaml       # Sample CIFAR-10 evaluation configuration
├── docs/                           # Developer and user documentation
│   ├── api.md                      # Python API reference
│   ├── cli.md                      # Command-line interface guide
│   ├── configuration.md            # trustbench.yaml schema and syntax
│   ├── contributing.md             # Developer guidelines
│   ├── dataset-adapters.md         # Custom and built-in dataset adapter guide
│   ├── experiments.md              # Experiment tracking and comparison
│   ├── index.md                    # Documentation index
│   ├── installation.md             # Installation and optional extras guide
│   ├── metrics.md                  # Metric catalog and mathematical definitions
│   ├── model-adapters.md           # Model adapter guide across ML frameworks
│   ├── plugins.md                  # Plugin creation and discovery
│   ├── quickstart.md               # 5-minute quickstart tutorial
│   ├── reporting.md                # HTML and JSON reporting guide
│   ├── robustness.md               # Corruption and distribution-shift benchmark
│   └── troubleshooting.md          # Common issues and solutions
├── examples/                       # Executable tutorial examples
│   ├── 01_evaluate_pytorch_model.py
│   ├── 02_evaluate_cifar10.py
│   ├── 03_robustness_benchmark.py
│   ├── 04_analyze_calibration.py
│   ├── 05_evaluate_uncertainty.py
│   ├── 06_compare_checkpoints.py
│   ├── 07_save_and_reproduce_experiment.py
│   ├── 08_create_custom_metric.py
│   └── 09_create_custom_model_adapter.py
├── scripts/                        # Utility scripts
│   ├── train_baseline.py           # Optional script for training CIFAR-10 baseline
│   └── utils.py                    # Terminal progress bar utility
├── src/trustbench/                 # Main library source code
│   ├── __init__.py                 # Root public API exports and lazy attribute loader
│   ├── benchmark.py                # High-level TrustBench interface (scikit-learn style)
│   ├── api/                        # Optional FastAPI REST service
│   │   ├── app.py                  # FastAPI application factory
│   │   └── routes/                 # REST endpoints for experiment querying
│   ├── cli/                        # Click CLI implementation
│   │   └── main.py                 # Commands: init, validate, evaluate, inspect, compare, version
│   ├── config/                     # Configuration loading and schema validation
│   │   ├── loader.py               # YAML file loader and schema validator
│   │   └── schema.py               # Pydantic v2 configuration models
│   ├── core/                       # Framework-independent evaluation engine
│   │   ├── evaluator.py            # evaluate() entry point
│   │   ├── exceptions.py           # TrustBench exception hierarchy
│   │   ├── inference.py            # Batched model inference loop
│   │   ├── protocols.py            # ModelAdapter and DatasetAdapter runtime protocols
│   │   ├── reproducibility.py      # Seed setting and system environment collection
│   │   └── results.py              # EvaluationResult data class and artifact export
│   ├── datasets/                   # Dataset adapters
│   │   ├── base.py                 # DatasetAdapter protocol
│   │   ├── cifar10.py              # CIFAR-10 dataset adapter and loader utilities
│   │   ├── memory.py               # Pure NumPy in-memory dataset adapter
│   │   └── torch_dataset.py        # Adapter wrapping torch.utils.data.Dataset (lazy)
│   ├── evaluation/                 # Task evaluators & structured metric calculation
│   │   ├── artifacts.py            # Compatibility shim for artifact I/O
│   │   ├── calibration.py          # ECE, Adaptive ECE, Brier score, Reliability diagram
│   │   ├── classification.py       # Accuracy, Balanced Accuracy, Precision, Recall, F1
│   │   ├── distribution_shift.py   # OOD detection (AUROC, FPR95) and label shift
│   │   ├── error_detection.py      # Error detection curves across uncertainty scores
│   │   ├── fairness.py             # Demographic parity, equal opportunity, disparate impact
│   │   ├── inference.py            # Compatibility shim for model inference
│   │   ├── regression.py           # Regression evaluator and RegressionResult
│   │   ├── reproducibility.py      # Compatibility shim for reproducibility
│   │   ├── retrieval.py            # Retrieval evaluator and RetrievalResult
│   │   ├── runner.py               # End-to-end checkpoint evaluation runner
│   │   ├── segmentation.py         # Segmentation evaluator and SegmentationResult
│   │   ├── selective.py            # Selective classification risk-coverage and AURC
│   │   ├── streaming.py            # Chunked and streaming evaluation for large data
│   │   └── uncertainty.py          # Entropy, Energy scores, Temperature scaling, Ensembles
│   ├── experiments/                # Experiment tracking and comparison
│   │   ├── comparison.py           # Multi-experiment metric comparator and differential engine
│   │   ├── lineage.py              # Lineage tracking, dependency hashing, config diffing
│   │   ├── manager.py              # High-level manager coordinating runs and store
│   │   ├── registry.py             # In-memory and database experiment registry
│   │   └── schemas.py              # Experiment metadata Pydantic schemas
│   ├── metrics/                    # Canonical metric functions & backward compatibility
│   │   ├── calibration.py          # Calibration metric shim
│   │   ├── classification.py       # Classification metric shim
│   │   ├── error_detection.py      # Error detection metric shim
│   │   ├── regression.py           # Pure regression metrics (MAE, MSE, RMSE, R², MAPE)
│   │   ├── retrieval.py            # Pure retrieval metrics (MRR, MAP, Recall@K, nDCG)
│   │   ├── segmentation.py         # Pure segmentation metrics (IoU, mIoU, Dice, Pixel Acc)
│   │   ├── selective.py            # Selective prediction metric shim
│   │   └── uncertainty.py          # Uncertainty metric shim
│   ├── models/                     # Multi-framework model adapters and architectures
│   │   ├── base.py                 # ModelAdapter runtime protocol
│   │   ├── checkpoint.py           # Checkpoint loading and architecture resolution
│   │   ├── jax.py                  # JAX / Flax model adapter
│   │   ├── keras.py                # TensorFlow / Keras model adapter
│   │   ├── onnx.py                 # ONNX Runtime model adapter
│   │   ├── pytorch.py              # PyTorch nn.Module model adapter
│   │   ├── sklearn.py              # Scikit-learn estimator model adapter
│   │   └── [cifar_architectures]   # PyTorch reference architectures (ResNet, VGG, etc., lazy)
│   ├── registry/                   # Extensibility and plugin discovery
│   │   ├── corruptions.py          # CorruptionRegistry for custom image perturbations
│   │   ├── datasets.py             # GenericDatasetAdapter and dataset registry
│   │   ├── metrics.py              # @register_metric decorator and metric registry
│   │   ├── models.py               # GenericModelAdapter and model registry
│   │   └── plugins.py              # Filesystem and package plugin discovery
│   ├── reporting/                  # Standalone reports, plots, and tables
│   │   ├── artifacts.py            # Atomic JSON, NPZ, and metadata serialization
│   │   ├── hashing.py              # SHA-256 artifact verification
│   │   ├── html_report.py          # Standalone HTML report generator with embedded styles
│   │   ├── plots.py                # Pure SVG and Plotly reliability, ROC, and risk curves
│   │   ├── result_io.py            # EvaluationResult save and load helpers
│   │   ├── schema.py               # JSON schema generator and validator for reports
│   │   └── tables.py               # Markdown and ASCII table formatters
│   ├── robustness/                 # Systematic robustness benchmark
│   │   ├── aggregation.py          # Mean Corruption Error (mCE) and Relative mCE
│   │   ├── corruptions.py          # 15 standard image corruptions (noise, blur, weather)
│   │   ├── evaluator.py            # Multi-severity corruption evaluator
│   │   ├── report.py               # Compatibility shim for robustness tables
│   │   └── severity.py             # Severity parameter configurations (levels 1-5)
│   └── storage/                    # Persistence engine
│       ├── base.py                 # ExperimentStore interface, errors, and records
│       ├── database.py             # SQLite engine and session factory (SQLAlchemy, lazy)
│       ├── filesystem.py           # Atomic JSON/NPZ filesystem store (pure Python)
│       ├── models.py               # SQLAlchemy ORM models (lazy)
│       ├── repositories.py         # SQLAlchemy repositories (lazy)
│       └── sqlite.py               # SQLiteExperimentStore implementation (lazy)
├── tests/                          # Test suite (204 tests)
│   ├── api/                        # FastAPI REST endpoint integration tests
│   ├── integration/                # End-to-end evaluation pipeline tests
│   └── unit/                       # Comprehensive unit tests for all components
├── CHANGELOG.md                    # Release history and version changes
├── CODE_OF_CONDUCT.md              # Contributor Code of Conduct
├── CONTRIBUTING.md                 # Contribution guidelines
├── EXAMPLE_INTEGRATION_AUDIT.md    # Pre-release audit of numerical correctness
├── LICENSE                         # Apache 2.0 Open Source License
├── pyproject.toml                  # Package definition, entrypoints, and dependency extras
├── README.md                       # Main project overview and usage guide
├── RELEASE_AUDIT_INITIAL.md        # Initial audit findings prior to cleanup
├── RELEASE_READINESS_REPORT.md     # Production release assessment report
├── REMOVAL_LOG.md                  # Log of removed files, functions, and dependencies
├── SECURITY.md                     # Vulnerability reporting instructions
├── trustbench.yaml                 # Sample configuration file
└── uv.lock                         # Exact reproducible dependency lockfile
```
