"""Unit tests for Phase 4: Extensibility, custom registration, and plugin architecture."""

import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from trustbench import (
    GenericDatasetAdapter,
    GenericModelAdapter,
    TrustBench,
    discover_plugins,
    evaluate,
    list_registered_corruptions,
    list_registered_metrics,
    load_plugin,
    register_corruption,
    register_dataset_adapter,
    register_metric,
    register_model_adapter,
    validate_dataset_adapter,
    validate_model_adapter,
)
from trustbench.datasets.memory import InMemoryDatasetAdapter
from trustbench.registry.corruptions import CORRUPTION_REGISTRY, unregister_corruption
from trustbench.registry.metrics import (
    METRIC_REGISTRY,
    MetricRegistrationError,
    unregister_metric,
)
from trustbench.registry.plugins import (
    PLUGIN_MANAGER,
    PluginError,
    PluginVersionMismatchError,
)

# ---------------------------------------------------------------------------
# Custom Metric Registration
# ---------------------------------------------------------------------------


def test_custom_metric_registration_and_evaluation():
    @register_metric("custom_error_count", overwrite=True)
    def custom_error_count(predictions: np.ndarray, labels: np.ndarray) -> int:
        return int(np.sum(predictions != labels))

    assert "custom_error_count" in list_registered_metrics()

    preds = np.array([0, 1, 1, 1])
    labels = np.array([0, 1, 0, 1])
    dataset = InMemoryDatasetAdapter(inputs=preds, labels=labels)

    model = GenericModelAdapter(
        predict_fn=lambda x: preds,
        num_classes=2,
    )

    result = evaluate(model, dataset, metrics=["classification", "custom_error_count"])
    assert "custom_error_count" in result.metrics
    assert result.metrics["custom_error_count"] == 1
    assert result.summary["custom_error_count"] == 1.0

    unregister_metric("custom_error_count")


def test_custom_metric_duplicate_registration():
    @register_metric("duplicate_metric", overwrite=True)
    def metric_v1(preds, targets):
        return 1.0

    try:
        # Re-registering without overwrite=True must raise
        with pytest.raises(MetricRegistrationError, match="already registered"):
            @register_metric("duplicate_metric", overwrite=False)
            def metric_v2(preds, targets):
                return 2.0

        # Re-registering with overwrite=True succeeds
        @register_metric("duplicate_metric", overwrite=True)
        def metric_v3(preds, targets):
            return 3.0

        m = METRIC_REGISTRY.get("duplicate_metric")
        assert m is not None
        assert m(np.array([0]), np.array([0])) == 3.0
    finally:
        unregister_metric("duplicate_metric")


def test_custom_metric_invalid_interface():
    with pytest.raises(MetricRegistrationError, match="must be a non-empty string"):
        register_metric("", fn=lambda p, t: 0.0)

    with pytest.raises(MetricRegistrationError, match="not callable"):
        register_metric("invalid_fn", fn="not_callable")  # type: ignore[arg-type]


def test_custom_metric_requirements_skipping():
    @register_metric("prob_requiring_metric", requires_probabilities=True, overwrite=True)
    def prob_metric(preds, targets, probabilities=None):
        return float(np.mean(probabilities))

    try:
        model = GenericModelAdapter(predict_fn=lambda x: np.array([0, 1]), num_classes=2)
        dataset = InMemoryDatasetAdapter(inputs=np.array([0, 1]), labels=np.array([0, 1]))

        result = evaluate(model, dataset, metrics=["prob_requiring_metric"])
        assert result.metrics["prob_requiring_metric"]["skipped"] is True
        assert "requires probabilities" in result.metrics["prob_requiring_metric"]["reason"]
    finally:
        unregister_metric("prob_requiring_metric")


# ---------------------------------------------------------------------------
# Custom Model Adapter
# ---------------------------------------------------------------------------


def test_generic_model_adapter_and_validation():
    # Valid adapter
    adapter = GenericModelAdapter(
        predict_fn=lambda x: np.array([0, 1, 0]),
        num_classes=2,
        predict_proba_fn=lambda x: np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2]]),
        metadata={"creator": "test"},
    )
    errors = validate_model_adapter(adapter)
    assert errors == []

    # Invalid adapter: missing methods and invalid num_classes
    class BrokenAdapter:
        num_classes = 0

    broken_errors = validate_model_adapter(BrokenAdapter())
    assert len(broken_errors) >= 3


def test_model_adapter_registry():
    @register_model_adapter("custom_tree_adapter", overwrite=True)
    class CustomTreeAdapter(GenericModelAdapter):
        pass

    assert register_model_adapter is not None


# ---------------------------------------------------------------------------
# Custom Dataset Adapter
# ---------------------------------------------------------------------------


def test_generic_dataset_adapter_and_validation():
    samples = [{"val": 10}, {"val": 20}, {"val": 30}]
    labels = [0, 1, 0]
    adapter = GenericDatasetAdapter(
        samples=samples,
        labels=labels,
        sample_ids=["s1", "s2", "s3"],
        class_names=["cat", "dog"],
    )

    errors = validate_dataset_adapter(adapter)
    assert errors == []
    assert len(adapter) == 3
    assert adapter.get_batch([0, 2]) == [{"val": 10}, {"val": 30}]
    assert np.array_equal(adapter.get_labels(), np.array([0, 1, 0]))

    # Broken dataset
    class BrokenDataset:
        def __len__(self):
            return -5  # noqa: PLE0303

    broken_errors = validate_dataset_adapter(BrokenDataset())
    assert len(broken_errors) >= 3


def test_dataset_adapter_registry():
    @register_dataset_adapter("custom_parquet_dataset", overwrite=True)
    class CustomParquetAdapter(GenericDatasetAdapter):
        pass

    assert register_dataset_adapter is not None


# ---------------------------------------------------------------------------
# Custom Corruption Functions
# ---------------------------------------------------------------------------


def test_custom_corruption_registration():
    @register_corruption(
        "custom_channel_drop",
        severities={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5},
        param_name="drop_rate",
        overwrite=True,
    )
    def apply_channel_drop(image: Image.Image, drop_rate: float, seed: int | None = None) -> Image.Image:
        arr = np.array(image).copy()
        arr[:, :, 0] = (arr[:, :, 0] * (1.0 - drop_rate)).astype(np.uint8)
        return Image.fromarray(arr)

    assert "custom_channel_drop" in list_registered_corruptions()

    img = Image.new("RGB", (32, 32), color=(200, 200, 200))
    from trustbench.robustness.corruptions import corrupt_pil_image
    from trustbench.robustness.severity import get_severity_config

    cfg = get_severity_config("custom_channel_drop", 3)
    assert cfg.param_value == 0.3

    corrupted = corrupt_pil_image(img, "custom_channel_drop", severity=3)
    assert corrupted.size == (32, 32)
    # Red channel scaled by 0.7: 200 * 0.7 = 140
    assert np.array(corrupted)[0, 0, 0] == 140

    unregister_corruption("custom_channel_drop")
    assert "custom_channel_drop" not in CORRUPTION_REGISTRY.list_corruptions()


# ---------------------------------------------------------------------------
# Plugin Discovery & Safety
# ---------------------------------------------------------------------------


def test_plugin_loading_and_setup_execution():
    fake_plugin = types.ModuleType("my_custom_plugin")
    setup_called = False

    def setup():
        nonlocal setup_called
        setup_called = True

    fake_plugin.setup = setup  # type: ignore[attr-defined]
    fake_plugin.__min_trustbench_version__ = "0.1.0"  # type: ignore[attr-defined]

    info = load_plugin(fake_plugin, name="fake_plugin", safe=True)
    assert info.status == "loaded"
    assert setup_called is True


def test_plugin_missing_dependencies_safe_failure():
    # Simulate a plugin whose import fails (e.g. missing optional heavy dep)
    info = load_plugin("non_existent_uninstalled_library_xyz", name="failing_dep", safe=True)
    assert info.status == "failed"
    assert info.error is not None
    assert "No module named" in info.error

    # When safe=False, raises PluginError
    with pytest.raises(PluginError):
        load_plugin("non_existent_uninstalled_library_xyz", safe=False)


def test_plugin_version_incompatibility():
    incompatible_plugin = types.ModuleType("future_plugin")
    incompatible_plugin.__min_trustbench_version__ = "99.0.0"  # type: ignore[attr-defined]

    info = load_plugin(incompatible_plugin, name="future_plugin", safe=True)
    assert info.status == "incompatible"
    assert "requires TrustBench >= 99.0.0" in str(info.error)

    # When safe=False, raises PluginVersionMismatchError
    with pytest.raises(PluginVersionMismatchError):
        load_plugin(incompatible_plugin, safe=False)


def test_plugin_discovery_mock():
    mock_ep = MagicMock()
    mock_ep.name = "mocked_plugin"
    mock_ep.value = "mock_pkg.plugin"

    plugin_mod = types.ModuleType("mocked_plugin")
    mock_ep.load.return_value = plugin_mod

    with patch("importlib.metadata.entry_points") as mock_entry_points:
        # Mock entry_points().select(group=...)
        mock_eps_obj = MagicMock()
        mock_eps_obj.select.return_value = [mock_ep]
        mock_entry_points.return_value = mock_eps_obj

        discovered = discover_plugins(safe=True)
        assert "mocked_plugin" in discovered
        assert discovered["mocked_plugin"].status == "loaded"

    PLUGIN_MANAGER.clear()


# ---------------------------------------------------------------------------
# Core Remains Usable Without Plugins
# ---------------------------------------------------------------------------


def test_core_usable_without_any_plugins():
    # Pure Python evaluation with no plugins, no external imports
    labels = np.array([0, 1, 0, 1])
    preds = np.array([0, 1, 0, 1])
    probs = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    model = GenericModelAdapter(
        predict_fn=lambda x: preds,
        num_classes=2,
        predict_proba_fn=lambda x: probs,
    )
    dataset = InMemoryDatasetAdapter(inputs=preds, labels=labels)

    tb = TrustBench(model, dataset, num_classes=2)
    res = tb.evaluate()

    assert res.summary["accuracy"] == 1.0
    assert res.summary["macro_f1"] == 1.0
    assert res.summary["brier_score"] is not None
