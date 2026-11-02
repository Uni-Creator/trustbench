"""Evaluation pipeline runner coordinating inference, metrics, and artifact saving."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from scipy.special import softmax

from trustbench.data.cifar10 import CIFAR10_CLASSES, get_cifar10_loaders
from trustbench.evaluation.artifacts import save_experiment_artifacts
from trustbench.evaluation.inference import load_model_from_checkpoint, run_inference
from trustbench.evaluation.reproducibility import get_environment_info, set_seed
from trustbench.metrics.calibration import compute_calibration_metrics
from trustbench.metrics.classification import compute_classification_metrics
from trustbench.metrics.uncertainty import compute_uncertainty_metrics


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
        batch_size: Batch size for test dataloader.
        seed: Random seed.
        device: Device to run evaluation on.

    Returns:
        Dictionary containing all computed metrics and artifact path.
    """
    set_seed(seed)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    # 1. Load model and checkpoint metadata
    model, raw_checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        device=device,
    )

    # 2. Prepare test data
    _, testloader = get_cifar10_loaders(
        data_dir=data_dir,
        batch_size_test=batch_size,
        num_workers=2,
        download=True,
    )

    # 3. Run inference
    inference_results = run_inference(model, testloader, device=device)
    logits = inference_results["logits"]
    predictions = inference_results["predictions"]
    labels = inference_results["labels"]

    # Compute probabilities via softmax
    probabilities = softmax(logits, axis=1)
    inference_results["probabilities"] = probabilities

    # 4. Compute metrics
    classification_metrics = compute_classification_metrics(
        labels=labels,
        predictions=predictions,
        class_names=CIFAR10_CLASSES,
    )

    calibration_metrics = compute_calibration_metrics(
        labels=labels,
        probabilities=probabilities,
    )

    uncertainty_metrics = compute_uncertainty_metrics(
        probabilities=probabilities,
    )

    combined_metrics = {
        "classification": classification_metrics,
        "calibration": calibration_metrics,
        "uncertainty": uncertainty_metrics,
        "summary": {
            "accuracy": classification_metrics["accuracy"],
            "macro_f1": classification_metrics["macro_f1"],
            "brier_score": calibration_metrics["brier_score"],
            "ece": calibration_metrics["ece"],
            "mean_entropy": uncertainty_metrics["mean_entropy"],
            "total_samples": len(labels),
        },
    }

    # 5. Metadata and Config
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

    # 6. Save artifacts
    save_experiment_artifacts(
        output_dir=output_dir,
        metadata=metadata,
        metrics=combined_metrics,
        config=config,
        predictions=inference_results,
    )

    return {
        "output_dir": str(output_dir),
        "metrics": combined_metrics,
        "metadata": metadata,
        "config": config,
    }
