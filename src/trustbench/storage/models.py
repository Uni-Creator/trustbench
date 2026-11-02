"""SQLAlchemy ORM models for experiments, metrics, and artifact indexing."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from trustbench.storage.database import Base


class ExperimentModel(Base):
    """Experiment record storing metadata, status, config, and checkpoint hash."""

    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    experiment_type: Mapped[str] = mapped_column(String(50), nullable=False, default="evaluation")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending", index=True)
    checkpoint_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    metrics = relationship("MetricModel", back_populates="experiment", cascade="all, delete-orphan")
    artifacts = relationship("ArtifactModel", back_populates="experiment", cascade="all, delete-orphan")


class MetricModel(Base):
    """Scalar metric value belonging to an experiment and condition."""

    __tablename__ = "metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    condition: Mapped[str] = mapped_column(String(100), nullable=False, default="Clean", index=True)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)

    experiment = relationship("ExperimentModel", back_populates="metrics")


class ArtifactModel(Base):
    """Artifact file index referencing on-disk files and SHA-256 checksums."""

    __tablename__ = "artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(36), ForeignKey("experiments.id", ondelete="CASCADE"), index=True)
    artifact_type: Mapped[str] = mapped_column(String(50), nullable=False)  # metadata, config, metrics, predictions, summary
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)

    experiment = relationship("ExperimentModel", back_populates="artifacts")
