"""Experiment artifact management and serialization."""

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_write_file(target: Path, write_fn: Any) -> None:
    """Write to a temporary file in the same directory and atomically rename."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix
    tmp_path = target.with_name(f".tmp_{target.stem}_{os.getpid()}_{time.time_ns()}{suffix}")
    try:
        write_fn(tmp_path)
        if not tmp_path.exists() and Path(f"{tmp_path}.npz").exists():
            tmp_path = Path(f"{tmp_path}.npz")
        tmp_path.replace(target)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


class NpEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy scalars and arrays."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def save_experiment_artifacts(
    output_dir: str | Path,
    metadata: dict[str, Any],
    metrics: dict[str, Any],
    config: dict[str, Any],
    predictions: dict[str, np.ndarray],
) -> Path:
    """Save standard TrustBench experiment artifacts.

    Structure:
    artifacts/experiments/<experiment_id>/
        ├── metadata.json
        ├── metrics.json
        ├── predictions.npz
        └── config.json

    Args:
        output_dir: Directory to store artifacts in.
        metadata: Experiment metadata (timestamp, model info, git commit, etc.).
        metrics: Computed evaluation metrics.
        config: Evaluation and model configuration.
        predictions: Dict containing logits, predictions, labels, sample_ids.

    Returns:
        Path to output directory.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Save metadata.json
    def _write_json(file_path: Path, data: Any) -> None:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, cls=NpEncoder)

    _atomic_write_file(out_path / "metadata.json", lambda p: _write_json(p, metadata))
    _atomic_write_file(out_path / "metrics.json", lambda p: _write_json(p, metrics))
    _atomic_write_file(out_path / "config.json", lambda p: _write_json(p, config))

    # 4. Save predictions.npz
    npz_data = {
        "logits": np.asarray(predictions.get("logits", [])),
        "predictions": np.asarray(predictions.get("predictions", [])),
        "labels": np.asarray(predictions.get("labels", [])),
        "sample_ids": np.asarray(predictions.get("sample_ids", [])),
    }
    if "probabilities" in predictions:
        npz_data["probabilities"] = np.asarray(predictions["probabilities"])

    _atomic_write_file(
        out_path / "predictions.npz",
        lambda p: np.savez_compressed(p, **npz_data),  # type: ignore[arg-type]
    )

    return out_path


def load_experiment_artifacts(artifact_dir: str | Path) -> dict[str, Any]:
    """Load all experiment artifacts from an experiment directory.

    Args:
        artifact_dir: Directory containing experiment files.

    Returns:
        Dict containing metadata, metrics, config, and predictions.
    """
    art_path = Path(artifact_dir)
    if not art_path.is_dir():
        raise FileNotFoundError(f"Artifact directory not found: {art_path}")

    with open(art_path / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    with open(art_path / "metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    with open(art_path / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    npz_file = np.load(art_path / "predictions.npz")
    predictions = {key: npz_file[key] for key in npz_file.files}

    return {
        "metadata": metadata,
        "metrics": metrics,
        "config": config,
        "predictions": predictions,
    }
