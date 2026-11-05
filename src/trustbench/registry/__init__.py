"""Extensibility and registration system for TrustBench."""

from trustbench.registry.corruptions import (
    CORRUPTION_REGISTRY,
    RegisteredCorruption,
    get_registered_corruption,
    list_registered_corruptions,
    register_corruption,
    unregister_corruption,
)
from trustbench.registry.datasets import (
    DATASET_ADAPTER_REGISTRY,
    GenericDatasetAdapter,
    get_dataset_adapter,
    register_dataset_adapter,
    validate_dataset_adapter,
)
from trustbench.registry.metrics import (
    METRIC_REGISTRY,
    MetricMetadata,
    MetricRegistrationError,
    RegisteredMetric,
    get_registered_metric,
    list_registered_metrics,
    register_metric,
    unregister_metric,
)
from trustbench.registry.models import (
    MODEL_ADAPTER_REGISTRY,
    GenericModelAdapter,
    get_model_adapter,
    register_model_adapter,
    validate_model_adapter,
)
from trustbench.registry.plugins import (
    PLUGIN_ENTRY_POINT_GROUP,
    PLUGIN_MANAGER,
    PluginError,
    PluginInfo,
    PluginVersionMismatchError,
    discover_plugins,
    load_plugin,
)

__all__ = [
    "CORRUPTION_REGISTRY",
    "DATASET_ADAPTER_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_ADAPTER_REGISTRY",
    "PLUGIN_ENTRY_POINT_GROUP",
    "PLUGIN_MANAGER",
    "GenericDatasetAdapter",
    "GenericModelAdapter",
    "MetricMetadata",
    "MetricRegistrationError",
    "PluginError",
    "PluginInfo",
    "PluginVersionMismatchError",
    "RegisteredCorruption",
    "RegisteredMetric",
    "discover_plugins",
    "get_dataset_adapter",
    "get_model_adapter",
    "get_registered_corruption",
    "get_registered_metric",
    "list_registered_corruptions",
    "list_registered_metrics",
    "load_plugin",
    "register_corruption",
    "register_dataset_adapter",
    "register_metric",
    "register_model_adapter",
    "unregister_corruption",
    "unregister_metric",
    "validate_dataset_adapter",
    "validate_model_adapter",
]
