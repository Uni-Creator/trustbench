# TrustBench Removal & Cleanup Log

This document records all deleted files, classes, methods, functions, and dependencies removed during the final production release audit, along with the justification, evidence, and safety verification.

---

## 1. Removed Files & Directories

| Item Removed | Type | Reason for Removal | Evidence & Safety Verification |
|---|---|---|---|
| `src/trustbench/data/cifar10.py` | Python file | Redundant compatibility shim. It duplicated `src/trustbench/datasets/cifar10.py` without adding functionality. | Only referenced by `scripts/train_baseline.py`, which was updated to use canonical `trustbench.datasets.cifar10`. No internal, external, or test code imports `trustbench.data`. Package directory `src/trustbench/data/` removed. |

---

## 2. Removed Dead Code, Functions & Test Routines

| File | Item Removed | Reason for Removal | Evidence & Safety Verification |
|---|---|---|---|
| `src/trustbench/models/densenet.py` | `def test()` | Dead test routine and script runner in library module. | Never called in tests or production code; called internal `print()`. |
| `src/trustbench/models/dla.py` | `def test()`, `if __name__ == '__main__':` | Dead test routine and script runner in library module. | Standalone script code left over from upstream reference; not part of public API. |
| `src/trustbench/models/dla_simple.py` | `def test()`, `if __name__ == '__main__':` | Dead test routine and script runner in library module. | Standalone script code left over from upstream reference; not part of public API. |
| `src/trustbench/models/dpn.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/efficientnet.py` | `def test()`, `if __name__ == '__main__':` | Dead test routine and script runner in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/googlenet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/mobilenet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/mobilenetv2.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/pnasnet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/preact_resnet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/regnet.py` | `def test()`, `if __name__ == '__main__':` | Dead test routine and script runner in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/resnet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/senet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/shufflenet.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/shufflenetv2.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |
| `src/trustbench/models/vgg.py` | `def test()` | Dead test routine in library module. | Unused; called internal `print()`. |

---

## 3. Removed & Cleaned Dependencies

| Package / Extra | Location | Reason for Removal | Evidence & Safety Verification |
|---|---|---|---|
| `pydantic-settings>=2.16.1` | `dependencies` | Declared in core dependencies but never imported or used. | `git grep pydantic_settings` showed 0 matches. TrustBench uses `pydantic.BaseModel` directly. Removed without breaking any configuration logic. |
| `dashboard = ["streamlit>=1.66.1"]` | `project.optional-dependencies` | Phantom dependency extra. Zero Streamlit or dashboard code exists in the repository. | `git grep streamlit` showed only test exclusion sets and `pyproject.toml`. Removing eliminated 16 unused transitive dependencies from `uv.lock`. |
| `pillow>=13.0.0` (in `torch`/`pytorch`) | `project.optional-dependencies` | Redundant duplication with mismatched version constraints. | `pillow>=10.0.0` is already in core dependencies. PyTorch extras do not require a separate pinned Pillow version. |
| `jinja2>=3.1.6` (in `reporting`) | `project.optional-dependencies` | Redundant duplicate declaration. | `jinja2>=3.1.6` is already declared in core `dependencies`. |
| `fastapi` and `uvicorn` (in `storage`) | `project.optional-dependencies` | Storage was tightly coupled with web server packages. | Separated into `storage = ["sqlalchemy>=2.0.0"]` and `api = ["fastapi>=0.100.0", "uvicorn[standard]>=0.20.0"]`. |

---

## 4. Removed Eager Import Bottlenecks (Architectural Decoupling)

| File | Change | Reason |
|---|---|---|
| `src/trustbench/models/__init__.py` | Removed 17 eager wildcard imports (`from .densenet import *`, `from .resnet import *`, etc.). Replaced with `_ARCH_MODULES` dictionary and `__getattr__`. | Eager imports previously forced PyTorch to load whenever `trustbench.models` (or `SklearnModelAdapter`, `ONNXModelAdapter`) was accessed, breaking non-PyTorch environments. |
| `src/trustbench/datasets/__init__.py` | Made `TorchDatasetAdapter` and `CIFAR10DatasetAdapter` lazy-loaded via `__getattr__`. | Previously, importing `trustbench.datasets` forced `import torch`, breaking pure NumPy `InMemoryDatasetAdapter` usage in environments without PyTorch. |
