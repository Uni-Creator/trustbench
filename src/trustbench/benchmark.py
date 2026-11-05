"""TrustBench — public high-level API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from trustbench.core.evaluator import evaluate as _core_evaluate
from trustbench.core.protocols import DatasetAdapter, ModelAdapter
from trustbench.core.results import EvaluationResult


def _wrap_model(model: Any, **kwargs: Any) -> ModelAdapter:
    """Coerce *model* to a :class:`ModelAdapter`."""
    # Already a ModelAdapter — use directly
    if isinstance(model, ModelAdapter):
        return model

    # torch.nn.Module — wrap in PyTorchModelAdapter
    try:
        from torch import nn

        from trustbench.models.pytorch import PyTorchModelAdapter
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to auto-wrap an nn.Module. "
            "Install it with: pip install torch"
        ) from exc

    if isinstance(model, nn.Module):
        num_classes: int | None = kwargs.get("num_classes")
        if num_classes is None:
            raise ValueError(
                "Pass num_classes=<int> when providing a raw nn.Module to TrustBench."
            )
        device = kwargs.get("device")
        return PyTorchModelAdapter(model, num_classes=num_classes, device=device)

    raise TypeError(
        f"model must be a ModelAdapter or nn.Module, got {type(model).__name__}."
    )


def _wrap_dataset(dataset: Any) -> DatasetAdapter:
    """Coerce *dataset* to a :class:`DatasetAdapter`."""
    if isinstance(dataset, DatasetAdapter):
        return dataset

    try:
        from torch.utils.data import DataLoader, Dataset

        from trustbench.datasets.torch_dataset import TorchDatasetAdapter
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to auto-wrap a Dataset/DataLoader. "
            "Install it with: pip install torch"
        ) from exc

    if isinstance(dataset, DataLoader):
        # Extract the underlying Dataset from the DataLoader
        dataset = dataset.dataset  # type: ignore[assignment]

    if isinstance(dataset, Dataset):
        return TorchDatasetAdapter(dataset)

    raise TypeError(
        f"dataset must be a DatasetAdapter, torch Dataset, or DataLoader, "
        f"got {type(dataset).__name__}."
    )


class TrustBench:
    """High-level evaluation interface.

    Example::

        import torch.nn as nn
        from trustbench import TrustBench

        model = nn.Linear(4, 3)
        # dataset is any torch.utils.data.Dataset or DatasetAdapter
        tb = TrustBench(model, dataset, num_classes=3)
        result = tb.evaluate()

    Args:
        model: A :class:`~trustbench.core.protocols.ModelAdapter`, ``nn.Module``
               (requires *num_classes* kwarg), DataLoader, or Dataset.
        dataset: A :class:`~trustbench.core.protocols.DatasetAdapter`,
                 ``torch.utils.data.Dataset``, or ``torch.utils.data.DataLoader``.
        metrics: Metric group names to compute.  Defaults to
                 ``("classification", "calibration", "uncertainty")``.
        batch_size: Inference batch size.
        class_names: Optional list of human-readable class labels.
        seed: Global random seed.
        output_dir: If provided, artifacts are saved here after :meth:`evaluate`.
        num_classes: Required when *model* is a raw ``nn.Module``.
        device: Device string passed to :class:`PyTorchModelAdapter`.
    """

    def __init__(
        self,
        model: Any,
        dataset: Any,
        *,
        metrics: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 100,
        class_names: list[str] | tuple[str, ...] | None = None,
        seed: int | None = None,
        output_dir: str | Path | None = None,
        # forwarded to _wrap_model when model is nn.Module
        num_classes: int | None = None,
        device: str | None = None,
    ) -> None:
        self._model_adapter: ModelAdapter = _wrap_model(
            model, num_classes=num_classes, device=device
        )
        self._dataset_adapter: DatasetAdapter = _wrap_dataset(dataset)
        self._metrics = metrics
        self._batch_size = batch_size
        self._class_names = class_names
        self._seed = seed
        self._output_dir = Path(output_dir) if output_dir is not None else None

    # ------------------------------------------------------------------
    # Class-method constructor from config
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> TrustBench:
        """Construct a :class:`TrustBench` instance from a YAML config file.

        The *model* and *dataset* must be resolvable from the config (i.e. a
        checkpoint path and architecture name must be provided).

        Args:
            config_path: Path to a ``trustbench.yaml`` config file.

        Returns:
            Configured :class:`TrustBench` instance ready to call :meth:`evaluate`.
        """
        from trustbench.config.loader import load_config
        from trustbench.models.checkpoint import load_model_from_checkpoint
        from trustbench.models.pytorch import PyTorchModelAdapter

        cfg = load_config(config_path)
        mcfg = cfg.model
        dcfg = cfg.dataset
        rcfg = cfg.reporting

        # ----- Model -----
        if mcfg.checkpoint_path is None:
            raise ValueError(
                "config.model.checkpoint_path is required for TrustBench.from_config()."
            )
        if mcfg.arch_name is None:
            raise ValueError(
                "config.model.arch_name is required for TrustBench.from_config()."
            )

        ckpt_path = cfg.resolve_path(mcfg.checkpoint_path)
        nn_module, _ = load_model_from_checkpoint(
            ckpt_path, model_name=mcfg.arch_name, device=mcfg.device
        )
        model_adapter = PyTorchModelAdapter(
            nn_module, num_classes=mcfg.num_classes, device=mcfg.device
        )

        # ----- Dataset -----
        dataset_adapter = cls._build_dataset_from_config(cfg)

        output_dir: Path | None = Path(
            cfg.resolve_path(rcfg.output_dir)
        )

        return cls(
            model_adapter,
            dataset_adapter,
            metrics=cfg.metrics.groups,
            batch_size=dcfg.batch_size,
            seed=cfg.seed,
            output_dir=output_dir,
        )

    @staticmethod
    def _build_dataset_from_config(cfg: Any) -> DatasetAdapter:
        """Instantiate the correct DatasetAdapter from config."""
        dcfg = cfg.dataset
        dtype = dcfg.dataset_type.lower()

        if dtype == "cifar10":
            from trustbench.datasets.cifar10 import (
                CIFAR10DatasetAdapter,
                get_cifar10_datasets,
            )

            data_dir = (
                str(cfg.resolve_path(dcfg.data_dir))
                if dcfg.data_dir
                else "./data"
            )
            trainset, testset = get_cifar10_datasets(data_dir=data_dir, download=True)
            split_dataset = trainset if dcfg.split == "train" else testset
            return CIFAR10DatasetAdapter(split_dataset)

        raise ValueError(
            f"Unknown dataset_type {dcfg.dataset_type!r}. "
            "Supported types: 'cifar10'."
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> EvaluationResult:
        """Run the full evaluation pipeline and return an :class:`EvaluationResult`.

        If *output_dir* was provided at construction, artifacts are automatically
        saved via :func:`~trustbench.reporting.save_result`.

        Returns:
            :class:`EvaluationResult` containing metrics, predictions, and metadata.
        """
        result = _core_evaluate(
            self._model_adapter,
            self._dataset_adapter,
            metrics=self._metrics,
            batch_size=self._batch_size,
            class_names=self._class_names,
            seed=self._seed,
        )

        if self._output_dir is not None:
            from trustbench.reporting.result_io import save_result

            save_result(result, self._output_dir)

        return result
