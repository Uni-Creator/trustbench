# Model Adapters Reference

Model adapters decouple the evaluation core from specific machine learning frameworks. TrustBench supports PyTorch, Scikit-learn, ONNX Runtime, Keras, JAX, and custom adapters via runtime protocols.

---

## The `ModelAdapter` Protocol

Any model adapter must implement the structural protocol defined in `trustbench.core.protocols.ModelAdapter`:

```python
class ModelAdapter(Protocol):
    @property
    def num_classes(self) -> int:
        """Total number of output classes."""
        ...

    @property
    def supports_predict_proba(self) -> bool:
        """Whether predict_proba is implemented."""
        ...

    @property
    def supports_predict_logits(self) -> bool:
        """Whether predict_logits is implemented."""
        ...

    def predict(self, inputs: Any) -> np.ndarray:
        """Return predicted class indices (N,)."""
        ...

    def predict_proba(self, inputs: Any) -> np.ndarray:
        """Return class probabilities (N, C)."""
        ...

    def predict_logits(self, inputs: Any) -> np.ndarray:
        """Return raw unnormalized logits (N, C)."""
        ...
```

---

## Built-In Adapters

### 1. PyTorch (`PyTorchModelAdapter`)
Wraps any `torch.nn.Module`.
- Automatically moves inputs to device (`cpu`, `cuda`).
- Employs non-blocking asynchronous CUDA streaming (`tensor.to(device, non_blocking=True)`).
- Executes under `torch.inference_mode()`.

```python
from trustbench.models.pytorch import PyTorchModelAdapter
adapter = PyTorchModelAdapter(model, num_classes=10, device="cuda")
```

### 2. Scikit-learn (`SklearnModelAdapter`)
Wraps any estimator supporting `predict()` and `fit()`.
- Automatically discovers classes from `estimator.classes_`.
- Handles binary, multi-class, and continuous regression estimators.
- Converts `decision_function` to probabilities when `predict_proba` is absent.

```python
from trustbench.models.sklearn import SklearnModelAdapter
adapter = SklearnModelAdapter(logistic_regression_model)
```

### 3. ONNX Runtime (`ONNXModelAdapter`)
Wraps an ONNX model file or pre-created `onnxruntime.InferenceSession`.
- Works on CPU or CUDA execution providers.
- Automatically discovers session input/output bindings.

```python
from trustbench.models.onnx import ONNXModelAdapter
adapter = ONNXModelAdapter("model.onnx", num_classes=10)
```

### 4. Keras / TensorFlow (`KerasModelAdapter`)
Wraps `tf.keras.Model` or `keras.Model` instances.

```python
from trustbench.models.keras import KerasModelAdapter
adapter = KerasModelAdapter(keras_model, num_classes=10)
```

### 5. JAX / Flax (`JAXModelAdapter`)
Wraps functional models via `apply_fn(params, inputs)`.

```python
from trustbench.models.jax import JAXModelAdapter
adapter = JAXModelAdapter(apply_fn=model.apply, params=params, num_classes=10)
```

---

## Creating a Custom Adapter

You can implement a custom adapter by implementing the `ModelAdapter` protocol or subclassing `trustbench.models.base.ModelAdapter`:

```python
import numpy as np
from trustbench import ModelAdapter

class ConstantModelAdapter(ModelAdapter):
    def __init__(self, constant_class: int = 0, num_classes: int = 2):
        self._class = constant_class
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def predict(self, inputs):
        return np.full(len(inputs), self._class, dtype=int)
```
