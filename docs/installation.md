# Installation Guide

TrustBench is designed with a lightweight, framework-independent core. You only install the dependencies required for your specific frameworks and workloads.

## Prerequisites

- **Python**: `>= 3.11`
- **Package Manager**: `pip` or [`uv`](https://github.com/astral-sh/uv)
- **OS**: Linux, macOS (tested in continuous integration); Windows (supported, not tested in CI)

---

## 1. Core Installation

To install TrustBench with base dependencies (NumPy, SciPy, Scikit-learn, Pydantic, PyYAML, Click, Jinja2, Pillow):

```bash
pip install trustbench
```

The core installation includes complete tabular evaluation (classification, regression), evaluation engine, and CLI tools.

---

## 2. Optional Dependency Groups

Install framework and feature bundles depending on your requirements:

| Extra | Command | Included Functionality |
|---|---|---|
| **PyTorch** | `pip install "trustbench[pytorch]"` | PyTorch model adapter, torchvision datasets, GPU device transfers |
| **ONNX Runtime** | `pip install "trustbench[onnx]"` | ONNX session execution (`onnxruntime`) on CPU or CUDA |
| **Reporting** | `pip install "trustbench[reporting]"` | Interactive Plotly HTML reporting and diagnostic charts |
| **Storage** | `pip install "trustbench[storage]"` | SQLite persistence via SQLAlchemy (tested); PostgreSQL (experimental) |
| **API** *(Experimental)* | `pip install "trustbench[api]"` | FastAPI REST server and uvicorn runner |
| **TensorFlow** *(Experimental)* | `pip install "trustbench[tensorflow]"` | Keras and TensorFlow model adapters |
| **JAX** *(Experimental)* | `pip install "trustbench[jax]"` | JAX/Flax functional inference adapters |
| **Everything** | `pip install "trustbench[all]"` | Installs all core and tested framework extras |

---

## 3. Verifying Installation

Verify the CLI is installed and check the version:

```bash
trustbench version
```
Expected output:
```text
trustbench x.y.z
```
*(e.g., `trustbench 0.2.3`)*

Verify Python imports:
```python
import trustbench
print(trustbench.__version__)
```
