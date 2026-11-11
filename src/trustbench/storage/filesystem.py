"""Filesystem-based experiment storage backend with atomic writes and checksum verification."""

from __future__ import annotations

import datetime
import json
import logging
import os
import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from trustbench.core.exceptions import SecurityError
from trustbench.reporting.artifacts import NpEncoder
from trustbench.reporting.hashing import compute_file_sha256
from trustbench.storage.base import (
    CorruptedArtifactError,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRecord,
    ExperimentState,
    ExperimentSummary,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from trustbench.core.results import EvaluationResult


def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically by creating a temporary file and renaming."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _atomic_write_json(path: Path, data: Any) -> None:
    """Serialize and write JSON atomically."""
    content = json.dumps(data, indent=2, cls=NpEncoder)
    _atomic_write_text(path, content)


def _atomic_write_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    """Save compressed NPZ arrays atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:8]}")
    with open(tmp_path, "wb") as f:
        np.savez_compressed(f, **data)  # type: ignore[arg-type]
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """Safely extract tar archive, rejecting path traversal attempts."""
    dest_dir = dest_dir.resolve()
    for member in tar.getmembers():
        target_path = (dest_dir / member.name).resolve()
        if target_path != dest_dir and dest_dir not in target_path.parents:
            raise SecurityError(
                f"Malicious archive entry '{member.name}' attempts path traversal outside '{dest_dir}'."
            )
    try:
        tar.extractall(path=dest_dir, filter="data")
    except TypeError:
        tar.extractall(path=dest_dir)


class FilesystemExperimentStore:
    """Filesystem experiment storage backend operating completely without a database.

    Directory Layout:
        base_dir/
            <experiment_id>/
                ├── manifest.json
                ├── metadata.json
                ├── metrics.json
                ├── config.json
                └── predictions.npz
    """

    def __init__(self, base_dir: str | Path = "./artifacts/experiments") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _exp_dir(self, experiment_id: str) -> Path:
        return self.base_dir / experiment_id

    def _read_manifest(self, exp_dir: Path) -> dict[str, Any]:
        manifest_file = exp_dir / "manifest.json"
        if not manifest_file.is_file():
            raise ExperimentNotFoundError(f"Manifest not found in {exp_dir}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_record_from_dir(self, exp_dir: Path) -> ExperimentRecord:
        manifest = self._read_manifest(exp_dir)
        exp_id = manifest.get("id", exp_dir.name)

        metrics_file = exp_dir / "metrics.json"
        metrics = {}
        if metrics_file.is_file():
            try:
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
            except Exception:  # noqa: BLE001
                metrics = {}

        config_file = exp_dir / "config.json"
        config = manifest.get("config", {})
        if config_file.is_file():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to read config.json: %s", exc)

        metadata_file = exp_dir / "metadata.json"
        metadata = manifest.get("metadata", {})
        if metadata_file.is_file():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Failed to read metadata.json: %s", exc)

        return ExperimentRecord(
            id=exp_id,
            name=manifest.get("name", exp_dir.name),
            experiment_type=manifest.get("experiment_type", "evaluation"),
            status=manifest.get("status", "completed"),
            created_at=manifest.get("created_at", ""),
            completed_at=manifest.get("completed_at"),
            checkpoint_sha256=manifest.get("checkpoint_sha256") or metadata.get("checkpoint_sha256"),
            dataset_fingerprint=manifest.get("dataset_fingerprint") or metadata.get("dataset_fingerprint"),
            config=config,
            metadata=metadata,
            metrics=metrics,
            artifacts=manifest.get("artifacts", []),
            error=manifest.get("error"),
        )

    def create(
        self,
        name: str,
        *,
        experiment_type: str = "evaluation",
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Create and initialize a new experiment record on disk."""
        exp_id = experiment_id or str(uuid.uuid4())
        exp_dir = self._exp_dir(exp_id)

        # Check for existing ID or Name
        if exp_dir.exists() and not overwrite:
            raise ExperimentAlreadyExistsError(
                f"Experiment with ID {exp_id!r} already exists at {exp_dir}."
            )

        existing = self.get(name)
        if existing and existing.id != exp_id and not overwrite:
            raise ExperimentAlreadyExistsError(
                f"Experiment with name {name!r} already exists (ID: {existing.id})."
            )

        exp_dir.mkdir(parents=True, exist_ok=True)
        now_iso = datetime.datetime.now(datetime.UTC).isoformat()

        cfg = config or {}
        meta = metadata or {}
        checkpoint_sha = meta.get("checkpoint_sha256") or cfg.get("checkpoint_sha256")
        dataset_fp = meta.get("dataset_fingerprint") or cfg.get("dataset_fingerprint")

        manifest: dict[str, Any] = {
            "id": exp_id,
            "name": name,
            "experiment_type": experiment_type,
            "status": ExperimentState.PENDING.value,
            "created_at": now_iso,
            "completed_at": None,
            "checkpoint_sha256": checkpoint_sha,
            "dataset_fingerprint": dataset_fp,
            "config": cfg,
            "metadata": meta,
            "artifacts": [],
            "checksums": {},
            "error": None,
        }

        _atomic_write_json(exp_dir / "manifest.json", manifest)
        if cfg:
            _atomic_write_json(exp_dir / "config.json", cfg)
        if meta:
            _atomic_write_json(exp_dir / "metadata.json", meta)

        return self._build_record_from_dir(exp_dir)

    def update_status(
        self,
        experiment_id: str,
        status: str | ExperimentState,
        *,
        error: str | None = None,
        completed: bool = False,
    ) -> ExperimentRecord:
        """Update experiment status and optional error details."""
        rec = self.get(experiment_id)
        if rec is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self._exp_dir(rec.id)
        manifest = self._read_manifest(exp_dir)

        status_str = status.value if isinstance(status, ExperimentState) else str(status)
        manifest["status"] = status_str
        if error:
            manifest["error"] = error
        if completed or status_str in (ExperimentState.COMPLETED.value, ExperimentState.FAILED.value):
            manifest["completed_at"] = datetime.datetime.now(datetime.UTC).isoformat()

        _atomic_write_json(exp_dir / "manifest.json", manifest)
        return self._build_record_from_dir(exp_dir)

    def save_result(
        self,
        experiment_id: str,
        result: EvaluationResult,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Persist evaluation results, artifacts, and computed checksums."""
        rec = self.get(experiment_id)
        if rec is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self._exp_dir(rec.id)
        manifest = self._read_manifest(exp_dir)

        # Merge metadata
        merged_meta = {**manifest.get("metadata", {}), **result.metadata}
        if extra_metadata:
            merged_meta.update(extra_metadata)

        config_data = manifest.get("config", {})

        # 1. Atomic write metrics.json
        metrics_file = exp_dir / "metrics.json"
        _atomic_write_json(metrics_file, result.metrics)

        # 2. Atomic write metadata.json
        meta_file = exp_dir / "metadata.json"
        _atomic_write_json(meta_file, merged_meta)

        # 3. Atomic write config.json
        config_file = exp_dir / "config.json"
        _atomic_write_json(config_file, config_data)

        # 4. Atomic write predictions.npz
        preds_file = exp_dir / "predictions.npz"
        _atomic_write_npz(preds_file, result.to_predictions_dict())

        # 5. Compute SHA-256 for all artifacts
        checksums = {
            "metrics.json": compute_file_sha256(metrics_file),
            "metadata.json": compute_file_sha256(meta_file),
            "config.json": compute_file_sha256(config_file),
            "predictions.npz": compute_file_sha256(preds_file),
        }

        artifacts = [
            {
                "artifact_type": "metrics",
                "file_path": str(metrics_file),
                "checksum": checksums["metrics.json"],
            },
            {
                "artifact_type": "metadata",
                "file_path": str(meta_file),
                "checksum": checksums["metadata.json"],
            },
            {
                "artifact_type": "config",
                "file_path": str(config_file),
                "checksum": checksums["config.json"],
            },
            {
                "artifact_type": "predictions",
                "file_path": str(preds_file),
                "checksum": checksums["predictions.npz"],
            },
        ]

        now_iso = datetime.datetime.now(datetime.UTC).isoformat()
        manifest["status"] = ExperimentState.COMPLETED.value
        manifest["completed_at"] = now_iso
        manifest["checksums"] = checksums
        manifest["artifacts"] = artifacts
        manifest["checkpoint_sha256"] = merged_meta.get("checkpoint_sha256")
        manifest["dataset_fingerprint"] = merged_meta.get("dataset_fingerprint")

        _atomic_write_json(exp_dir / "manifest.json", manifest)
        return self._build_record_from_dir(exp_dir)

    def get(self, experiment_id_or_name: str) -> ExperimentRecord | None:
        """Retrieve full experiment record by UUID or unique name."""
        direct_dir = self._exp_dir(experiment_id_or_name)
        if direct_dir.is_dir() and (direct_dir / "manifest.json").is_file():
            return self._build_record_from_dir(direct_dir)

        # Search across all directories by name or id
        if not self.base_dir.is_dir():
            return None

        for child in self.base_dir.iterdir():
            if child.is_dir() and (child / "manifest.json").is_file():
                try:
                    manifest = self._read_manifest(child)
                    if manifest.get("id") == experiment_id_or_name or manifest.get("name") == experiment_id_or_name:
                        return self._build_record_from_dir(child)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to read experiment manifest in %s: %s", child, exc)
                    continue
        return None

    def list(
        self,
        *,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> list[ExperimentSummary]:
        """List experiments matching query filters."""
        summaries: list[ExperimentSummary] = []
        if not self.base_dir.is_dir():
            return summaries

        for child in self.base_dir.iterdir():
            if child.is_dir() and (child / "manifest.json").is_file():
                try:
                    rec = self._build_record_from_dir(child)
                    if experiment_type and rec.experiment_type != experiment_type:
                        continue
                    if status and rec.status != status:
                        continue
                    if checkpoint_sha256 and rec.checkpoint_sha256 != checkpoint_sha256:
                        continue

                    # Extract summary metrics
                    summary_metrics = rec.metrics.get("summary", {})
                    summaries.append(
                        ExperimentSummary(
                            id=rec.id,
                            name=rec.name,
                            experiment_type=rec.experiment_type,
                            status=rec.status,
                            created_at=rec.created_at,
                            completed_at=rec.completed_at,
                            checkpoint_sha256=rec.checkpoint_sha256,
                            dataset_fingerprint=rec.dataset_fingerprint,
                            metrics_summary=summary_metrics,
                            artifacts_count=len(rec.artifacts),
                            config=rec.config,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Failed to read experiment in %s: %s", child, exc)
                    continue

        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    def delete(self, experiment_id: str, *, delete_files: bool = True) -> bool:
        """Delete an experiment and optionally its on-disk files."""
        rec = self.get(experiment_id)
        if not rec:
            return False

        exp_dir = self._exp_dir(rec.id)
        if delete_files and exp_dir.is_dir():
            shutil.rmtree(exp_dir)
        return True

    def verify_checksums(self, experiment_id: str) -> dict[str, bool]:
        """Verify SHA-256 checksums of all persisted experiment artifacts."""
        rec = self.get(experiment_id)
        if rec is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self._exp_dir(rec.id)
        manifest = self._read_manifest(exp_dir)
        checksums = manifest.get("checksums", {})

        results: dict[str, bool] = {}
        for filename, expected_hash in checksums.items():
            fpath = exp_dir / filename
            if not fpath.is_file():
                results[filename] = False
                continue
            actual_hash = compute_file_sha256(fpath)
            results[filename] = actual_hash == expected_hash

        return results

    def export(self, experiment_id: str, export_path: str | Path) -> Path:
        """Export experiment artifacts to a portable tar.gz archive."""
        rec = self.get(experiment_id)
        if rec is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self._exp_dir(rec.id)
        target = Path(export_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(target, "w:gz") as tar:
            for item in exp_dir.iterdir():
                if not item.name.endswith(".tmp"):
                    tar.add(item, arcname=item.name)

        return target

    def import_experiment(
        self,
        archive_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Import an experiment from an archive and verify all checksums."""
        src = Path(archive_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Archive not found: {src}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with tarfile.open(src, "r:*") as tar:
                _safe_extract_tar(tar, tmp_path)

            # Check manifest in extracted files
            manifest_file = tmp_path / "manifest.json"
            if not manifest_file.is_file():
                raise CorruptedArtifactError(f"Invalid archive {src}: manifest.json not found.")

            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)

            exp_id = manifest.get("id")
            if not exp_id:
                raise CorruptedArtifactError(f"Invalid archive {src}: experiment id missing in manifest.")

            # Verify checksums of unpacked artifacts
            checksums = manifest.get("checksums", {})
            for filename, expected_hash in checksums.items():
                fpath = tmp_path / filename
                if not fpath.is_file():
                    raise CorruptedArtifactError(
                        f"Archive corrupted: expected file {filename} is missing."
                    )
                actual_hash = compute_file_sha256(fpath)
                if actual_hash != expected_hash:
                    raise CorruptedArtifactError(
                        f"Archive corrupted: checksum mismatch for {filename}."
                    )

            # Check if destination exists
            dest_dir = self._exp_dir(exp_id)
            if dest_dir.exists():
                if not overwrite:
                    raise ExperimentAlreadyExistsError(
                        f"Cannot import experiment {exp_id!r}: already exists in store."
                    )
                shutil.rmtree(dest_dir)

            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in tmp_path.iterdir():
                shutil.copy2(item, dest_dir / item.name)

        return self._build_record_from_dir(dest_dir)
