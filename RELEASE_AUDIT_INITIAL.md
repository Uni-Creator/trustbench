# TrustBench Release Audit — Initial Architecture & Codebase Assessment

**Date:** 2026-11-09  
**Version Evaluated:** 0.2.0  
**Status:** Mandatory Pre-Cleanup Assessment (No modifications performed prior to this inventory)

---

## 1. Executive Summary

TrustBench is designed as a lightweight, scikit-learn-like library and CLI for machine learning model evaluation (reliability, robustness, calibration, uncertainty, and performance). 

This initial audit evaluated all source code files, configurations, dependencies, documentation, examples, and packaging outputs. The audit identified key areas where the codebase can be cleaned up, simplified, and hardened for production release:
1. **Dependency Inconsistencies & Phantom Extras:** The `pydantic-settings` dependency is declared in core dependencies but never imported. The `dashboard` optional dependency extra declares `streamlit>=1.66.1`, but there is zero Streamlit or dashboard code in the entire repository. Core dependencies declare excessively strict minimum versions matching the development environment (e.g. `numpy>=2.4.6`, `scikit-learn>=1.9.1`, `torch>=2.14.1`).
2. **Optional Dependency Leakage:** Eager `import torch` in `trustbench/datasets/__init__.py` (via `torch_dataset.py`) and `trustbench/models/__init__.py` (via 17 PyTorch CIFAR-10 model architecture files) causes non-PyTorch imports (e.g., `from trustbench.datasets import InMemoryDatasetAdapter` or `from trustbench.models import SklearnModelAdapter`) to fail when PyTorch is not installed.
3. **Compatibility Shims & Duplication:** Five redundant 10–15 line compatibility shims exist (`trustbench.data.cifar10`, `trustbench.evaluation.artifacts`, `trustbench.evaluation.reproducibility`, `trustbench.evaluation.inference`, `trustbench.robustness.report`).
4. **Legacy PyTorch Model Clutter:** 17 CIFAR-10 model architecture files in `src/trustbench/models/` contain copy-pasted `if __name__ == '__main__': test()` blocks and eager torch imports.
5. **Storage / API Coupling:** `fastapi` and `uvicorn` are bundled under `[project.optional-dependencies].storage`, coupling database persistence to web server packages.

---

## 2. Current Architecture & Package Inventory

The repository is structured into 14 packages under `src/trustbench/`:

| Package | Purpose & Responsibility | Assessment |
|---|---|---|
| `trustbench` | Root module, public API exports, high-level `TrustBench` class. | **Required** (core) |
| `trustbench.core` | Core evaluator, protocols, inference loop, exceptions, reproducibility. | **Required** (core) |
| `trustbench.evaluation` | Task evaluators (classification, calibration, uncertainty, selective, distribution shift, fairness, regression, retrieval, segmentation, streaming). | **Required** (core) |
| `trustbench.metrics` | Canonical metric calculations (regression, retrieval, segmentation) and metric shims. | **Required** (core) |
| `trustbench.models` | Model adapters (`ModelAdapter`, PyTorch, Sklearn, ONNX, Keras, JAX), checkpoint loader, and CIFAR architectures. | **Required** (core adapters) + **Review** (CIFAR architectures) |
| `trustbench.datasets` | Dataset adapters (`DatasetAdapter`, `InMemoryDatasetAdapter`, `TorchDatasetAdapter`, `CIFAR10DatasetAdapter`). | **Required** (core) |
| `trustbench.robustness` | Robustness benchmarking, image corruptions, severity scaling, mCE aggregation. | **Required** (core) |
| `trustbench.registry` | Extensibility registries for metrics, model adapters, dataset adapters, corruptions, plugins. | **Required** (core) |
| `trustbench.experiments` | Experiment tracking, lineage, comparison engine, metadata schemas. | **Required** (core) |
| `trustbench.storage` | Filesystem atomic store (`FilesystemExperimentStore`) and optional SQLite/SQLAlchemy store. | **Required** (core filesystem, optional SQL) |
| `trustbench.reporting` | JSON schema, artifact serialization, HTML reporting, Plotly/SVG plots, Markdown tables. | **Required** (core) |
| `trustbench.config` | Pydantic v2 YAML configuration parsing and validation. | **Required** (core) |
| `trustbench.cli` | Click CLI interface (`init`, `validate`, `evaluate`, `inspect`, `compare`, `version`). | **Required** (core) |
| `trustbench.api` | Optional FastAPI REST service for experiment retrieval and directory indexing. | **Optional** (unnecessary for core library) |
| `trustbench.data` | Legacy 1-file package (`cifar10.py`) re-exporting `trustbench.datasets.cifar10`. | **Redundant** |

---

## 3. Public API Inventory

### Python Public API (`from trustbench import ...`)
- **Protocols & Adapters:** `ModelAdapter`, `DatasetAdapter`, `GenericModelAdapter`, `GenericDatasetAdapter`, `PyTorchModelAdapter`, `SklearnModelAdapter`, `ONNXModelAdapter`, `KerasModelAdapter`, `JAXModelAdapter`.
- **Evaluators:** `TrustBench`, `evaluate`, `evaluate_regression`, `evaluate_retrieval`, `evaluate_segmentation`, `evaluate_distribution_shift`.
- **Result Containers:** `EvaluationResult`, `RegressionResult`, `RetrievalResult`, `SegmentationResult`, `ShiftType`.
- **Metrics:** `compute_fairness_metrics`, `compute_label_distribution_shift`, `compute_ood_detection_metrics`, `compute_regression_metrics`, `compute_retrieval_metrics`, `compute_segmentation_metrics`.
- **Registries:** `register_metric`, `register_model_adapter`, `register_dataset_adapter`, `register_corruption`, `list_registered_metrics`, `list_registered_corruptions`, `validate_model_adapter`, `validate_dataset_adapter`, `discover_plugins`, `load_plugin`.
- **Exceptions:** `TrustBenchError`, `ShapeError`, `SecurityError`, `UnsupportedCapabilityError`.
- **Lazy Loaders:** `ExperimentManager`, `FilesystemExperimentStore`, `SQLiteExperimentStore`.

### CLI API (`trustbench <subcommand>`)
- `trustbench init [--output FILE] [--force]`
- `trustbench validate [CONFIG]`
- `trustbench evaluate [CONFIG] [-o OUTPUT_DIR] [-q] [-v]`
- `trustbench inspect RUN_DIR`
- `trustbench compare RUN_A RUN_B`
- `trustbench version`

---

## 4. Dependency Inventory & Issues

### Current `pyproject.toml`
```toml
dependencies = [
    "numpy>=2.4.6",
    "scipy>=1.16.0",
    "scikit-learn>=1.9.1",
    "pydantic>=2.14.1",
    "pydantic-settings>=2.16.1",
    "pyyaml>=6.0",
    "click>=8.1.0",
    "jinja2>=3.1.6",
    "pillow>=10.0.0",
]

[project.optional-dependencies]
torch = ["torch>=2.14.1", "torchvision>=0.29.1", "pillow>=13.0.0"]
pytorch = ["torch>=2.14.1", "torchvision>=0.29.1", "pillow>=13.0.0"]
storage = ["sqlalchemy>=2.1.6", "fastapi>=0.161.1", "uvicorn[standard]>=0.54.0"]
reporting = ["plotly>=7.1.0", "jinja2>=3.1.6"]
dashboard = ["streamlit>=1.66.1"]
onnx = ["onnxruntime>=1.16.0"]
tensorflow = ["tensorflow>=2.15.0"]
jax = ["jax>=0.4.20"]
all = ["trustbench[torch,storage,reporting,dashboard,onnx]"]
```

### Identified Dependency Problems:
1. **Unused Core Dependency:** `pydantic-settings` is never imported. TrustBench uses standard `pydantic.BaseModel`.
2. **Phantom Dependency Extra:** `dashboard` declares `streamlit>=1.66.1`. No Streamlit code exists in the repository.
3. **Inappropriate Minimum Version Pins:**
   - `numpy>=2.4.6`: Rejects normal Python 3.11/3.12 installs with NumPy 1.26 or 2.0. Recommended: `numpy>=1.24.0`.
   - `scipy>=1.16.0`: Rejects SciPy 1.10–1.15. Recommended: `scipy>=1.10.0`.
   - `scikit-learn>=1.9.1`: Scikit-learn 1.9.1 is a bleeding-edge release. Recommended: `scikit-learn>=1.3.0`.
   - `torch>=2.14.1`: Rejects PyTorch 2.0–2.3. Recommended: `torch>=2.0.0`.
   - `torchvision>=0.29.1`: Rejects TorchVision 0.15–0.18. Recommended: `torchvision>=0.15.0`.
4. **Redundant Duplicate Dependencies:**
   - `jinja2>=3.1.6` is declared in both `dependencies` and `reporting`.
   - `pillow` is declared in `dependencies` (`>=10.0.0`) and in `torch`/`pytorch` (`>=13.0.0`).
5. **Coupled Storage Extra:** `storage` bundles `fastapi` and `uvicorn` with `sqlalchemy`. Database persistence should not force web server installation. If the REST API is retained, it should be under `api`.

---

## 5. Identified Duplicate & Redundant Code

### Compatibility Shims
| File | Canonical Target | Action |
|---|---|---|
| `src/trustbench/data/cifar10.py` | `src/trustbench/datasets/cifar10.py` | Update `scripts/train_baseline.py` to import from canonical; safely deprecate/remove `trustbench/data`. |
| `src/trustbench/evaluation/artifacts.py` | `src/trustbench/reporting/artifacts.py` | Keep thin shim for backward compatibility; update internal calls to canonical. |
| `src/trustbench/evaluation/reproducibility.py` | `src/trustbench/core/reproducibility.py` | Keep thin shim; update internal calls to canonical. |
| `src/trustbench/evaluation/inference.py` | `src/trustbench/models/checkpoint.py`, `src/trustbench/models/pytorch.py` | Keep thin shim; update internal calls to canonical. |
| `src/trustbench/robustness/report.py` | `src/trustbench/reporting/tables.py` | Keep thin shim; update internal calls to canonical. |

### Dead Code in Model Definitions
- 12 files in `src/trustbench/models/` (`dla.py`, `dla_simple.py`, `dpn.py`, `pnasnet.py`, `preact_resnet.py`, `regnet.py`, `resnext.py`, `senet.py`, `shufflenet.py`, `shufflenetv2.py`, etc.) contain dead `def test(): ... if __name__ == '__main__': test()` routines.
- `src/trustbench/models/__init__.py` imports all 17 model files eagerly with wildcard imports (`from .densenet import *`), leaking torch dependency to any caller of `trustbench.models`.

---

## 6. Import & Isolation Problems

### Problem 1: `from trustbench.datasets import InMemoryDatasetAdapter`
- **Cause:** `src/trustbench/datasets/__init__.py` imports `from trustbench.datasets.torch_dataset import TorchDatasetAdapter`.
- **Effect:** Fails with `ModuleNotFoundError: No module named 'torch'` if PyTorch is not installed, even though `InMemoryDatasetAdapter` is pure NumPy.
- **Fix:** Lazy-load or guarded import for `TorchDatasetAdapter`.

### Problem 2: `from trustbench.models import SklearnModelAdapter`
- **Cause:** `src/trustbench/models/__init__.py` imports all PyTorch CIFAR models eagerly.
- **Effect:** Fails with `ModuleNotFoundError: No module named 'torch'` if PyTorch is not installed, even though `SklearnModelAdapter` is framework-independent.
- **Fix:** In `src/trustbench/models/__init__.py`, use `__getattr__` or conditional imports for PyTorch models so that `ModelAdapter`, `SklearnModelAdapter`, `ONNXModelAdapter`, `KerasModelAdapter`, and `JAXModelAdapter` can be imported without PyTorch.

---

## 7. Packaging & Distribution Problems

1. **Wheel & Sdist Verification:**
   - Both wheel (`trustbench-0.2.0-py3-none-any.whl`) and sdist (`trustbench-0.2.0.tar.gz`) build successfully via `uv build`.
   - However, `pyproject.toml.orig` was packaged inside the source distribution.
   - Pinned minimum versions in `pyproject.toml` will prevent clean installation in standard environments with slightly older (but fully compatible) package versions.
2. **Phantom Extras in Wheel Metadata:**
   - `dist-info/METADATA` lists `Provides-Extra: dashboard` with `Requires-Dist: streamlit>=1.66.1; extra == 'dashboard'`. This must be removed.

---

## 8. Proposed Deletions & Modifications with Reasons

| Target | Proposed Change | Reason | Risk |
|---|---|---|---|
| `pyproject.toml` | Remove `pydantic-settings>=2.16.1` | Completely unused dependency. | None |
| `pyproject.toml` | Remove `dashboard` extra | No Streamlit dashboard exists in repo. | None |
| `pyproject.toml` | Relax strict pinned versions (`numpy>=1.24.0`, `scipy>=1.10.0`, `scikit-learn>=1.3.0`, `torch>=2.0.0`) | Avoid rejecting valid user environments. | Very low |
| `pyproject.toml` | Remove duplicate `jinja2` from `reporting` extra; remove duplicate `pillow` from `torch` extra | Avoid redundant declarations with mismatched version constraints. | None |
| `pyproject.toml` | Separate `storage` (`sqlalchemy`) from `api` (`fastapi`, `uvicorn`) | Storage should not mandate web server packages. | Low |
| `src/trustbench/datasets/__init__.py` | Guard/lazy-load `TorchDatasetAdapter` | Prevent eager torch dependency for pure NumPy dataset adapters. | Low |
| `src/trustbench/models/__init__.py` | Lazy-load CIFAR PyTorch models via `__getattr__` or conditional import | Prevent eager torch dependency for pure Sklearn/ONNX/JAX adapters. | Low |
| `src/trustbench/data/` | Deprecate and update `scripts/train_baseline.py` to `datasets.cifar10` | Redundant 1-file package. Keep thin shim or remove safely. | Very low |
| `src/trustbench/models/*.py` | Remove dead `if __name__ == '__main__': test()` blocks from CIFAR architectures | Library hygiene; removes dead execution code. | None |
| `docs/installation.md` | Remove mention of `trustbench[dashboard]` | Documentation accuracy. | None |

---

## 9. Next Steps

With this initial audit completed:
1. Implement justified non-breaking changes in reviewable batches.
2. Run full test suite and Ruff verification after each batch.
3. Test clean installation of the built wheel and sdist in an isolated virtual environment.
4. Produce final documentation and readiness deliverables.
