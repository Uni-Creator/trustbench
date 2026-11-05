"""Base interfaces, models, and protocols for experiment storage backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from trustbench.core.results import EvaluationResult


class ExperimentState(str, Enum):
    """Lifecycle states of an evaluation experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    IMPORTED = "imported"


class StorageError(Exception):
    """Base exception for storage and persistence errors."""


class ExperimentAlreadyExistsError(StorageError, ValueError):
    """Raised when attempting to create an experiment that already exists without overwrite=True."""


class ExperimentNotFoundError(StorageError, KeyError):
    """Raised when a requested experiment cannot be located."""


class CorruptedArtifactError(StorageError, OSError):
    """Raised when an artifact checksum mismatch indicates file corruption."""


class IncompatibleExperimentsError(StorageError, ValueError):
    """Raised when attempting to compare experiments with incompatible configurations or tasks."""


@dataclass
class ExperimentSummary:
    """Lightweight summary of an experiment for fast listing and filtering."""

    id: str
    name: str
    experiment_type: str
    status: str
    created_at: str
    completed_at: str | None = None
    checkpoint_sha256: str | None = None
    dataset_fingerprint: str | None = None
    metrics_summary: dict[str, Any] = field(default_factory=dict)
    artifacts_count: int = 0
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentRecord:
    """Complete record of an experiment including configuration, metadata, and artifacts."""

    id: str
    name: str
    experiment_type: str
    status: str
    created_at: str
    completed_at: str | None = None
    checkpoint_sha256: str | None = None
    dataset_fingerprint: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a clean dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "experiment_type": self.experiment_type,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "checkpoint_sha256": self.checkpoint_sha256,
            "dataset_fingerprint": self.dataset_fingerprint,
            "config": self.config,
            "metadata": self.metadata,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
        }


@runtime_checkable
class ExperimentStore(Protocol):
    """Protocol defining the storage backend interface."""

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
        """Create and initialize a new experiment record."""
        ...

    def update_status(
        self,
        experiment_id: str,
        status: str | ExperimentState,
        *,
        error: str | None = None,
        completed: bool = False,
    ) -> ExperimentRecord:
        """Update experiment status and optional error details."""
        ...

    def save_result(
        self,
        experiment_id: str,
        result: EvaluationResult,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Persist evaluation results, artifacts, and computed checksums."""
        ...

    def get(self, experiment_id_or_name: str) -> ExperimentRecord | None:
        """Retrieve full experiment record by UUID or unique name."""
        ...

    def list(
        self,
        *,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> list[ExperimentSummary]:
        """List experiments matching query filters."""
        ...

    def delete(self, experiment_id: str, *, delete_files: bool = True) -> bool:
        """Delete an experiment and optionally its on-disk files."""
        ...

    def export(self, experiment_id: str, export_path: str | Path) -> Path:
        """Export experiment artifacts to a portable tar.gz or zip archive."""
        ...

    def import_experiment(
        self,
        archive_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Import an experiment from an archive and verify all checksums."""
        ...

    def verify_checksums(self, experiment_id: str) -> dict[str, bool]:
        """Verify SHA-256 checksums of all persisted experiment artifacts."""
        ...

