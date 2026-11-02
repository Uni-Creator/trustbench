"""Pydantic schemas for experiments, metrics, artifacts, and comparisons."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class MetricEntry(BaseModel):
    """Single metric value for a specific condition."""

    condition: str
    metric_name: str
    metric_value: float


class ArtifactEntry(BaseModel):
    """Artifact metadata and file location."""

    id: int | None = None
    artifact_type: str
    file_path: str
    checksum: str | None = None


class ExperimentBase(BaseModel):
    """Base schema for an experiment."""

    name: str
    experiment_type: str = "evaluation"
    checkpoint_sha256: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class ExperimentCreate(ExperimentBase):
    """Schema for registering a new experiment."""

    status: str = "pending"


class ExperimentResponse(ExperimentBase):
    """Schema for an experiment response."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    status: str
    created_at: datetime
    completed_at: datetime | None = None
    metrics_count: int = 0
    artifacts_count: int = 0


class ExperimentDetail(ExperimentResponse):
    """Detailed experiment schema including its metrics and artifacts."""

    metrics: list[MetricEntry] = Field(default_factory=list)
    artifacts: list[ArtifactEntry] = Field(default_factory=list)


class ExperimentComparisonRequest(BaseModel):
    """Request to compare multiple experiments."""

    experiment_ids: list[str]
    conditions: list[str] | None = None
    metrics: list[str] | None = None


class ExperimentComparisonResponse(BaseModel):
    """Response containing comparative metrics across experiments."""

    experiment_ids: list[str]
    experiment_names: dict[str, str]
    checkpoints: dict[str, str | None]
    conditions: list[str]
    metrics: list[str]
    # table: { condition: { metric: { experiment_id: value } } }
    comparison_table: dict[str, dict[str, dict[str, float]]]
