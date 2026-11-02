"""High-level experiment lifecycle manager coordinating execution, registration, and status."""

import uuid
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from trustbench.evaluation.runner import run_evaluation
from trustbench.experiments.comparison import ExperimentComparator
from trustbench.experiments.registry import ExperimentRegistry
from trustbench.robustness.evaluator import RobustnessEvaluator
from trustbench.storage.database import get_session_factory
from trustbench.storage.repositories import ExperimentRepository


class ExperimentManager:
    """High-level coordinator for experiment lifecycle, execution, indexing, and comparison."""

    def __init__(self, session: Session | None = None):
        self.session_provided = session is not None
        self._session = session
        self.registry = ExperimentRegistry(session=self.session)
        self.comparator = ExperimentComparator(session=self.session)

    @property
    def session(self) -> Session:
        if self._session is None:
            factory = get_session_factory()
            self._session = factory()
        return self._session

    def list_experiments(
        self,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments with summary statistics."""
        repo = ExperimentRepository(self.session)
        models = repo.list_all(
            experiment_type=experiment_type,
            status=status,
            checkpoint_sha256=checkpoint_sha256,
        )
        results = []
        for m in models:
            metrics = repo.get_metrics(m.id)
            artifacts = repo.get_artifacts(m.id)
            results.append({
                "id": m.id,
                "name": m.name,
                "experiment_type": m.experiment_type,
                "status": m.status,
                "checkpoint_sha256": m.checkpoint_sha256,
                "config": m.config,
                "created_at": m.created_at,
                "completed_at": m.completed_at,
                "metrics_count": len(metrics),
                "artifacts_count": len(artifacts),
            })
        return results

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve full details of an experiment including metrics and indexed artifacts."""
        repo = ExperimentRepository(self.session)
        exp = repo.get_by_id(experiment_id)
        if not exp:
            exp = repo.get_by_name(experiment_id)
        if not exp:
            return None

        metrics = repo.get_metrics(exp.id)
        artifacts = repo.get_artifacts(exp.id)

        return {
            "id": exp.id,
            "name": exp.name,
            "experiment_type": exp.experiment_type,
            "status": exp.status,
            "checkpoint_sha256": exp.checkpoint_sha256,
            "config": exp.config,
            "created_at": exp.created_at,
            "completed_at": exp.completed_at,
            "metrics": [
                {
                    "condition": m.condition,
                    "metric_name": m.metric_name,
                    "metric_value": m.metric_value,
                }
                for m in metrics
            ],
            "artifacts": [
                {
                    "id": a.id,
                    "artifact_type": a.artifact_type,
                    "file_path": a.file_path,
                    "checksum": a.checksum,
                }
                for a in artifacts
            ],
        }

    def run_baseline_experiment(
        self,
        name: str,
        checkpoint_path: str = "./checkpoint/ckpt.pth",
        output_dir: str | Path | None = None,
        data_dir: str = "./dataset",
        batch_size: int = 128,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a baseline evaluation and register it in the experiment manager."""
        exp_id = str(uuid.uuid4())
        out_dir = Path(output_dir or f"./artifacts/experiments/{name}")

        repo = ExperimentRepository(self.session)
        exp_model = repo.create(
            name=name,
            experiment_type="baseline",
            status="running",
            experiment_id=exp_id,
            config={
                "checkpoint_path": checkpoint_path,
                "data_dir": data_dir,
                "batch_size": batch_size,
                "seed": seed,
            },
        )

        try:
            run_evaluation(
                checkpoint_path=checkpoint_path,
                output_dir=out_dir,
                data_dir=data_dir,
                batch_size=batch_size,
                seed=seed,
            )
            # Index generated artifacts into DB
            self.registry.index_artifact_directory(out_dir, experiment_name=name, experiment_type="baseline")
            repo.update_status(exp_model.id, status="completed", completed=True)
        except Exception:
            repo.update_status(exp_model.id, status="failed", completed=True)
            raise

        return self.get_experiment_detail(exp_model.id) or {}

    def run_robustness_experiment(
        self,
        name: str,
        checkpoint_path: str = "./checkpoint/ckpt.pth",
        output_dir: str | Path | None = None,
        data_dir: str = "./dataset",
        corruptions: list[str] | None = None,
        severities: list[int] | None = None,
        batch_size: int = 128,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a robustness evaluation and index all artifacts."""
        exp_id = str(uuid.uuid4())
        out_dir = Path(output_dir or f"./artifacts/experiments/{name}")

        repo = ExperimentRepository(self.session)
        exp_model = repo.create(
            name=name,
            experiment_type="robustness",
            status="running",
            experiment_id=exp_id,
            config={
                "checkpoint_path": checkpoint_path,
                "data_dir": data_dir,
                "corruptions": corruptions,
                "severities": severities,
                "batch_size": batch_size,
                "seed": seed,
            },
        )

        try:
            evaluator = RobustnessEvaluator(
                checkpoint_path=checkpoint_path,
                output_dir=out_dir,
                data_dir=data_dir,
                batch_size=batch_size,
                seed=seed,
            )
            evaluator.run_full_suite(corruption_types=corruptions, severities=severities)
            self.registry.index_artifact_directory(out_dir, experiment_name=name, experiment_type="robustness")
            repo.update_status(exp_model.id, status="completed", completed=True)
        except Exception:
            repo.update_status(exp_model.id, status="failed", completed=True)
            raise

        return self.get_experiment_detail(exp_model.id) or {}

    def compare_experiments(
        self,
        experiment_ids: list[str],
        conditions: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple experiments."""
        return self.comparator.compare(experiment_ids, conditions=conditions, metric_names=metrics)
