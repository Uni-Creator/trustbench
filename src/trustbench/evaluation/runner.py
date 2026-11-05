"""Compatibility runner: checkpoint + CIFAR-10 evaluation with artifact saving."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from trustbench.core.evaluator import evaluate
from trustbench.core.reproducibility import get_environment_info, set_seed
from trustbench.datasets.cifar10 import (
    CIFAR10_CLASSES,
    CIFAR10DatasetAdapter,
    get_cifar10_datasets,
)
from trustbench.models.checkpoint import load_model_from_checkpoint
from trustbench.models.pytorch import PyTorchModelAdapter
from trustbench.reporting.artifacts import save_experiment_artifacts


def run_evaluation(
    checkpoint_path: str | Path = "./checkpoint/ckpt.pth",
    model_name: str = "ResNet18",
    data_dir: str = "./dataset",
    output_dir: str | Path = "./artifacts/experiments/baseline_001",
    batch_size: int = 100,
    seed: int = 42,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Execute end-to-end evaluation and record artifacts.

    Args:
        checkpoint_path: Path to model checkpoint file.
        model_name: Model architecture name.
        data_dir: Path to dataset directory.
        output_dir: Directory where artifacts are saved.
        batch_size: Batch size for test evaluation.
        seed: Random seed.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing all computed metrics and artifact path.
    """
    set_seed(seed)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    model, raw_checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
    )
    adapter = PyTorchModelAdapter(
        model,
        num_classes=len(CIFAR10_CLASSES),
        device=device,
    )

    _, testset = get_cifar10_datasets(data_dir=data_dir, download=True)
    dataset = CIFAR10DatasetAdapter(testset)

    result = evaluate(
        adapter,
        dataset,
        batch_size=batch_size,
        class_names=CIFAR10_CLASSES,
    )

    config = {
        "checkpoint_path": str(checkpoint_path),
        "model_name": model_name,
        "data_dir": str(data_dir),
        "batch_size": batch_size,
        "seed": seed,
        "classes": list(CIFAR10_CLASSES),
    }

    metadata = {
        "experiment_id": output_dir.name,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint_epoch": raw_checkpoint.get("epoch"),
        "checkpoint_train_acc": raw_checkpoint.get("acc"),
        "environment": get_environment_info(),
    }

    save_experiment_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        metrics=result.metrics,
        config=config,
        predictions=result.to_predictions_dict(),
    )

    return {
        "output_dir": str(output_dir),
        "metrics": result.metrics,
        "metadata": metadata,
        "config": config,
    }
