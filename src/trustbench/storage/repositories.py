"""Repository layer for persisting and querying experiments, metrics, and artifacts."""

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from trustbench.storage.models import ArtifactModel, ExperimentModel, MetricModel


class ExperimentRepository:
    """Repository handling CRUD and queries for experiments."""

    def __init__(self, session: Session):
        self.session = session

    def create(
        self,
        name: str,
        experiment_type: str = "evaluation",
        status: str = "pending",
        checkpoint_sha256: str | None = None,
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
    ) -> ExperimentModel:
        """Create and store a new experiment."""
        exp = ExperimentModel(
            name=name,
            experiment_type=experiment_type,
            status=status,
            checkpoint_sha256=checkpoint_sha256,
            config=config or {},
        )
        if experiment_id:
            exp.id = experiment_id
        self.session.add(exp)
        self.session.commit()
        self.session.refresh(exp)
        return exp

    def get_by_id(self, experiment_id: str) -> ExperimentModel | None:
        """Find experiment by primary UUID."""
        stmt = select(ExperimentModel).where(ExperimentModel.id == experiment_id)
        return self.session.scalars(stmt).first()

    def get_by_name(self, name: str) -> ExperimentModel | None:
        """Find experiment by name."""
        stmt = select(ExperimentModel).where(ExperimentModel.name == name)
        return self.session.scalars(stmt).first()

    def list_all(
        self,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ExperimentModel]:
        """List experiments with optional filters."""
        stmt = select(ExperimentModel)
        if experiment_type:
            stmt = stmt.where(ExperimentModel.experiment_type == experiment_type)
        if status:
            stmt = stmt.where(ExperimentModel.status == status)
        if checkpoint_sha256:
            stmt = stmt.where(ExperimentModel.checkpoint_sha256 == checkpoint_sha256)
        stmt = stmt.order_by(ExperimentModel.created_at.desc()).limit(limit).offset(offset)
        return list(self.session.scalars(stmt).all())

    def update_status(
        self,
        experiment_id: str,
        status: str,
        completed: bool = False,
    ) -> ExperimentModel | None:
        """Update status and optionally mark completed."""
        exp = self.get_by_id(experiment_id)
        if not exp:
            return None
        exp.status = status
        if completed:
            exp.completed_at = datetime.now(UTC)
        self.session.commit()
        self.session.refresh(exp)
        return exp

    def add_metric(
        self,
        experiment_id: str,
        condition: str,
        metric_name: str,
        metric_value: float,
    ) -> MetricModel:
        """Record an individual scalar metric."""
        metric = MetricModel(
            experiment_id=experiment_id,
            condition=condition,
            metric_name=metric_name,
            metric_value=float(metric_value),
        )
        self.session.add(metric)
        self.session.commit()
        self.session.refresh(metric)
        return metric

    def add_metrics_bulk(
        self,
        experiment_id: str,
        metrics_list: list[dict[str, Any]],
    ) -> list[MetricModel]:
        """Batch record metrics."""
        models = [
            MetricModel(
                experiment_id=experiment_id,
                condition=m["condition"],
                metric_name=m["metric_name"],
                metric_value=float(m["metric_value"]),
            )
            for m in metrics_list
        ]
        self.session.add_all(models)
        self.session.commit()
        return models

    def get_metrics(
        self,
        experiment_id: str,
        condition: str | None = None,
    ) -> list[MetricModel]:
        """Retrieve recorded metrics for an experiment."""
        stmt = select(MetricModel).where(MetricModel.experiment_id == experiment_id)
        if condition:
            stmt = stmt.where(MetricModel.condition == condition)
        return list(self.session.scalars(stmt).all())

    def add_artifact(
        self,
        experiment_id: str,
        artifact_type: str,
        file_path: str,
        checksum: str | None = None,
    ) -> ArtifactModel:
        """Index a persisted artifact file."""
        art = ArtifactModel(
            experiment_id=experiment_id,
            artifact_type=artifact_type,
            file_path=file_path,
            checksum=checksum,
        )
        self.session.add(art)
        self.session.commit()
        self.session.refresh(art)
        return art

    def get_artifacts(self, experiment_id: str) -> list[ArtifactModel]:
        """Get all indexed artifacts for an experiment."""
        stmt = select(ArtifactModel).where(ArtifactModel.experiment_id == experiment_id)
        return list(self.session.scalars(stmt).all())
