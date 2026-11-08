# Plugin Development & Extensibility

TrustBench allows external developers and teams to extend the evaluation engine without modifying TrustBench source code.

---

## 1. Extension Points

You can register custom:
1. **Metrics**: Custom scoring functions (`@register_metric`).
2. **Model Adapters**: Custom model wrappers (`@register_model_adapter`).
3. **Dataset Adapters**: Custom data loaders (`@register_dataset_adapter`).
4. **Input Corruptions**: Domain-specific corruption functions (`@register_corruption`).

---

## 2. In-Code Custom Registration

### Custom Metric
```python
from trustbench import register_metric
import numpy as np

@register_metric("top2_accuracy")
def compute_top2_accuracy(probabilities: np.ndarray, targets: np.ndarray) -> float:
    top2 = np.argsort(probabilities, axis=1)[:, -2:]
    hits = [t in row for t, row in zip(targets, top2)]
    return float(np.mean(hits))
```

### Custom Model Adapter
```python
from trustbench import register_model_adapter, ModelAdapter

@register_model_adapter("custom_framework")
class MyCustomFrameworkAdapter(ModelAdapter):
    def __init__(self, model):
        self.model = model

    @property
    def num_classes(self) -> int:
        return 10

    def predict(self, inputs):
        return self.model.infer(inputs)
```

---

## 3. Package Entry-Point Discovery

To package and distribute your plugin as an independent Python library:

In your plugin's `pyproject.toml`:
```toml
[project.entry-points."trustbench.plugins"]
my_custom_plugin = "my_plugin_package:register"
```

In `my_plugin_package/__init__.py`:
```python
from trustbench import register_metric, register_corruption

def register():
    """TrustBench plugin entry-point hook."""
    register_metric("my_metric", my_metric_func)
    register_corruption("rain_effect", rain_corruption_func)
```

When TrustBench initializes, it discovers and registers installed plugins safely using Python entry points, without executing arbitrary directory scripts.
