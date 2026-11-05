"""Metadata and lineage tracking for reproducible evaluations."""

from __future__ import annotations

import datetime
import hashlib
import logging
import os
import platform
import sys
from typing import Any

import numpy as np

from trustbench.reporting.hashing import compute_file_sha256_or_none

logger = logging.getLogger(__name__)

# Sentinels for distinguishing unavailable from missing metadata
UNAVAILABLE = "unavailable"
MISSING = "missing"


def compute_dataset_fingerprint(dataset: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint for a dataset.

    Inspects sample_ids, labels, or raw arrays when available.
    Returns UNAVAILABLE if the dataset cannot be fingerprinted.
    """
    hasher = hashlib.sha256()

    # 1. Check for explicit get_labels() or labels attribute
    labels = None
    if hasattr(dataset, "get_labels") and callable(dataset.get_labels):
        try:
            labels = dataset.get_labels()
        except Exception:  # noqa: BLE001
            labels = None
    elif hasattr(dataset, "labels"):
        labels = dataset.labels

    # 2. Check for explicit sample IDs
    sample_ids = None
    if hasattr(dataset, "get_sample_ids") and callable(dataset.get_sample_ids):
        try:
            sample_ids = dataset.get_sample_ids()
        except Exception:  # noqa: BLE001
            sample_ids = None
    elif hasattr(dataset, "sample_ids"):
        sample_ids = dataset.sample_ids

    fingerprinted = False

    if sample_ids is not None:
        try:
            arr_ids = np.asarray(sample_ids)
            hasher.update(arr_ids.tobytes())
            fingerprinted = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to hash sample IDs: %s", exc)

    if labels is not None:
        try:
            arr_labels = np.asarray(labels)
            hasher.update(arr_labels.tobytes())
            fingerprinted = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to hash labels: %s", exc)

    # If neither is available, try hashing length and dataset type
    if not fingerprinted and hasattr(dataset, "__len__"):
        try:
            hasher.update(f"{type(dataset).__name__}:{len(dataset)}".encode())
            fingerprinted = True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to hash dataset length: %s", exc)

    if not fingerprinted:
        return UNAVAILABLE

    return hasher.hexdigest()


def compute_checkpoint_sha256(checkpoint_path: str | os.PathLike[str] | None) -> str:
    """Compute SHA-256 for a model checkpoint file, distinguishing missing from unavailable."""
    if checkpoint_path is None:
        return UNAVAILABLE
    path_str = str(checkpoint_path)
    if not os.path.isfile(path_str):
        return MISSING
    h = compute_file_sha256_or_none(path_str)
    return h or MISSING


def collect_model_metadata(model: Any) -> dict[str, Any]:
    """Extract model metadata while cleanly handling unavailable properties."""
    meta: dict[str, Any] = {
        "model_name": getattr(model, "model_name", None) or type(model).__name__,
        "model_class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "num_classes": getattr(model, "num_classes", UNAVAILABLE),
        "architecture": getattr(model, "architecture", UNAVAILABLE),
        "parameter_count": UNAVAILABLE,
        "capabilities": {},
    }

    # Inspect capabilities
    for cap in ("supports_predict_proba", "supports_predict_logits"):
        if hasattr(model, cap):
            meta["capabilities"][cap] = bool(getattr(model, cap))

    # Inspect PyTorch module parameter count if applicable
    if "torch" in sys.modules:
        import torch

        if isinstance(model, torch.nn.Module):
            try:
                meta["parameter_count"] = sum(p.numel() for p in model.parameters())
                meta["architecture"] = model.__class__.__name__
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to count parameters for PyTorch module: %s", exc)
    elif hasattr(model, "parameters") and callable(model.parameters):
        try:
            params = list(model.parameters())
            meta["parameter_count"] = sum(p.numel() for p in params if hasattr(p, "numel"))
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to count parameters: %s", exc)

    return meta


def collect_dataset_metadata(dataset: Any) -> dict[str, Any]:
    """Extract dataset metadata and compute dataset fingerprint."""
    num_samples = len(dataset) if hasattr(dataset, "__len__") else UNAVAILABLE
    class_names = getattr(dataset, "class_names", UNAVAILABLE)
    if class_names is None:
        class_names = UNAVAILABLE

    meta: dict[str, Any] = {
        "dataset_name": getattr(dataset, "name", None) or type(dataset).__name__,
        "dataset_class": f"{dataset.__class__.__module__}.{dataset.__class__.__qualname__}",
        "num_samples": num_samples,
        "class_names": class_names,
        "dataset_fingerprint": compute_dataset_fingerprint(dataset),
    }
    return meta


def collect_hardware_metadata() -> dict[str, Any]:
    """Capture hardware specifications without forcing heavyweight framework loading."""
    hardware: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or UNAVAILABLE,
        "cpu_count": os.cpu_count() or UNAVAILABLE,
        "cuda_available": False,
        "cuda_device_name": UNAVAILABLE,
    }

    # Only inspect CUDA if torch is already imported in runtime
    if "torch" in sys.modules:
        import torch

        try:
            cuda_avail = torch.cuda.is_available()
            hardware["cuda_available"] = cuda_avail
            if cuda_avail:
                hardware["cuda_device_name"] = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to query CUDA hardware metadata: %s", exc)

    return hardware


def collect_framework_versions() -> dict[str, Any]:
    """Capture runtime and dependency framework versions."""
    versions: dict[str, Any] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
    }

    try:
        from trustbench import __version__ as tb_version

        versions["trustbench"] = tb_version
    except Exception:  # noqa: BLE001
        versions["trustbench"] = "0.2.0"

    # Only check heavy frameworks if installed without failing
    for pkg in ("torch", "torchvision", "scipy", "sklearn", "pydantic", "fastapi", "sqlalchemy"):
        if pkg in sys.modules:
            mod = sys.modules[pkg]
            versions[pkg] = getattr(mod, "__version__", UNAVAILABLE)
        else:
            try:
                import importlib.metadata

                versions[pkg] = importlib.metadata.version(pkg)
            except Exception:  # noqa: BLE001
                versions[pkg] = UNAVAILABLE

    return versions


def build_lineage_record(
    model: Any = None,
    dataset: Any = None,
    config: dict[str, Any] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    seed: int | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete, standardized metadata and lineage record."""
    now_iso = datetime.datetime.now(datetime.UTC).isoformat()

    record: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": now_iso,
        "execution_timestamp": now_iso,
        "checkpoint_sha256": compute_checkpoint_sha256(checkpoint_path),
        "random_seeds": {
            "seed": seed if seed is not None else UNAVAILABLE,
            "deterministic_mode": True if seed is not None else UNAVAILABLE,
        },
        "environment": collect_framework_versions(),
        "hardware": collect_hardware_metadata(),
        "model": collect_model_metadata(model) if model is not None else UNAVAILABLE,
        "dataset": collect_dataset_metadata(dataset) if dataset is not None else UNAVAILABLE,
        "config": config or {},
    }

    if extra_metadata:
        record["extra"] = extra_metadata

    return record
