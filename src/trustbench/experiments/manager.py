"""High-level experiment lifecycle manager coordinating execution, registration, and status."""

from __future__ import annotations

import traceback
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trustbench.experiments.comparison import ExperimentComparator
from trustbench.experiments.lineage import build_lineage_record
from trustbench.storage.base import (
    ExperimentRecord,
    ExperimentState,
    ExperimentStore,
)
from trustbench.storage.filesystem import FilesystemExperimentStore

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from trustbench.core.protocols import DatasetAdapter, ModelAdapter


class ExperimentManager:
    """High-level coordinator for experiment lifecycle, execution, indexing, and comparison."""

    def __init__(
        self,
        session: Session | None = None,
        store: ExperimentStore | None = None,
        backend: str = "auto",
        base_dir: str | Path = "./artifacts/experiments",
        db_path: str | Path | None = None,
    ) -> None:
        self.session_provided = session is not None
        self._session = session
        self.base_dir = Path(base_dir)

        if store is not None:
            self.store: ExperimentStore = store
        elif session is not None or backend == "sqlite":
            from trustbench.storage.sqlite import SQLiteExperimentStore

            self.store = SQLiteExperimentStore(
                session=session,
                db_path=db_path,
                artifacts_dir=base_dir,
            )
        else:
            self.store = FilesystemExperimentStore(base_dir=base_dir)

        self.comparator = ExperimentComparator(
            session=self._session,
            store=self.store,
        )

    @property
    def session(self) -> Session:
        """Lazily initialize or return SQLAlchemy session for backward compatibility."""
        if self._session is None:
            from trustbench.storage.database import get_session_factory

            factory = get_session_factory()
            self._session = factory()
        return self._session

    @property
    def registry(self):
        """Lazily initialize or return ExperimentRegistry."""
        from trustbench.experiments.registry import ExperimentRegistry

        return ExperimentRegistry(session=self.session)

    # -----------------------------------------------------------------------
    # Lifecycle Management
    # -----------------------------------------------------------------------

    def create_experiment(
        self,
        name: str,
        *,
        experiment_type: str = "evaluation",
        config: dict[str, Any] | None = None,
        experiment_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Create and register a new experiment in PENDING state."""
        return self.store.create(
            name=name,
            experiment_type=experiment_type,
            config=config,
            experiment_id=experiment_id,
            metadata=metadata,
            overwrite=overwrite,
        )

    def run_experiment(
        self,
        model: ModelAdapter,
        dataset: DatasetAdapter,
        name: str,
        *,
        metrics: list[str] | tuple[str, ...] | None = None,
        batch_size: int = 100,
        class_names: list[str] | tuple[str, ...] | None = None,
        seed: int = 42,
        checkpoint_path: str | Path | None = None,
        experiment_type: str = "evaluation",
        config: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
        raise_on_error: bool = True,
    ) -> ExperimentRecord:
        """Execute evaluation experiment under full lifecycle tracking.

        Transitions state:
        PENDING -> RUNNING -> COMPLETED (or FAILED on error).
        """
        cfg = dict(config or {})
        cfg.update({
            "metrics": list(metrics) if metrics else ["classification", "calibration", "uncertainty"],
            "batch_size": batch_size,
            "seed": seed,
        })
        if checkpoint_path:
            cfg["checkpoint_path"] = str(checkpoint_path)

        lineage = build_lineage_record(
            model=model,
            dataset=dataset,
            config=cfg,
            checkpoint_path=checkpoint_path,
            seed=seed,
            extra_metadata=extra_metadata,
        )

        # 1. Create in PENDING
        rec = self.create_experiment(
            name=name,
            experiment_type=experiment_type,
            config=cfg,
            metadata=lineage,
            overwrite=overwrite,
        )

        # 2. Transition to RUNNING
        self.store.update_status(rec.id, status=ExperimentState.RUNNING)

        try:
            from trustbench.core.evaluator import evaluate

            result = evaluate(
                model=model,
                dataset=dataset,
                metrics=metrics,
                batch_size=batch_size,
                class_names=class_names,
                seed=seed,
            )

            # 3. Save artifacts and transition to COMPLETED
            updated_rec = self.store.save_result(
                rec.id,
                result,
                extra_metadata=lineage,
            )
            return updated_rec

        except Exception as exc:
            err_trace = traceback.format_exc()
            self.store.update_status(
                rec.id,
                status=ExperimentState.FAILED,
                error=f"{exc}\n{err_trace}",
                completed=True,
            )
            if raise_on_error:
                raise
            return self.store.get(rec.id)  # type: ignore[return-value]

    def get_experiment(self, experiment_id_or_name: str) -> ExperimentRecord | None:
        """Retrieve full details of an experiment by ID or name."""
        return self.store.get(experiment_id_or_name)

    def list_experiments(
        self,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> list[dict[str, Any]]:
        """List experiments with summary statistics, compatible with existing callers."""
        summaries = self.store.list(
            experiment_type=experiment_type,
            status=status,
            checkpoint_sha256=checkpoint_sha256,
        )
        return [
            {
                "id": s.id,
                "name": s.name,
                "experiment_type": s.experiment_type,
                "status": s.status,
                "checkpoint_sha256": s.checkpoint_sha256,
                "dataset_fingerprint": s.dataset_fingerprint,
                "config": s.config,
                "created_at": s.created_at,
                "completed_at": s.completed_at,
                "metrics_count": len(s.metrics_summary),
                "artifacts_count": s.artifacts_count,
            }
            for s in summaries
        ]

    def get_experiment_detail(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve full details of an experiment including metrics and indexed artifacts."""
        rec = self.store.get(experiment_id)
        if not rec:
            return None

        # Build metrics list for compatibility with existing tests
        metrics_list = []
        summary = rec.metrics.get("summary", {})
        for k, v in summary.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                metrics_list.append({
                    "condition": "Clean",
                    "metric_name": k,
                    "metric_value": float(v),
                })

        return {
            "id": rec.id,
            "name": rec.name,
            "experiment_type": rec.experiment_type,
            "status": rec.status,
            "checkpoint_sha256": rec.checkpoint_sha256,
            "dataset_fingerprint": rec.dataset_fingerprint,
            "config": rec.config,
            "created_at": rec.created_at,
            "completed_at": rec.completed_at,
            "metrics": metrics_list,
            "artifacts": rec.artifacts,
            "error": rec.error,
        }

    def delete_experiment(self, experiment_id: str, *, delete_files: bool = True) -> bool:
        """Delete an experiment and its indexed artifacts."""
        return self.store.delete(experiment_id, delete_files=delete_files)

    def export_experiment(self, experiment_id: str, export_path: str | Path) -> Path:
        """Export experiment artifacts to a portable tar.gz archive."""
        return self.store.export(experiment_id, export_path)

    def import_experiment(
        self,
        archive_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Import an experiment from an archive and verify all checksums."""
        return self.store.import_experiment(archive_path, overwrite=overwrite)

    def verify_experiment(self, experiment_id: str) -> dict[str, bool]:
        """Verify checksums of all persisted artifacts for an experiment."""
        return self.store.verify_checksums(experiment_id)

    def compare_experiments(
        self,
        experiment_ids: list[str],
        conditions: list[str] | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple experiments."""
        return self.comparator.compare(
            experiments=experiment_ids,
            conditions=conditions,
            metric_names=metrics,
        )

    # -----------------------------------------------------------------------
    # Existing Specialized Runners (Kept for 100% Backward Compatibility)
    # -----------------------------------------------------------------------

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
        from trustbench.evaluation.runner import run_evaluation

        out_dir = Path(output_dir or f"./artifacts/experiments/{name}")
        exp_id = str(uuid.uuid4())

        from trustbench.storage.repositories import ExperimentRepository

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
            self.registry.index_artifact_directory(
                out_dir, experiment_name=name, experiment_type="baseline"
            )
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
        from trustbench.robustness.evaluator import RobustnessEvaluator

        out_dir = Path(output_dir or f"./artifacts/experiments/{name}")
        exp_id = str(uuid.uuid4())

        from trustbench.storage.repositories import ExperimentRepository

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
            self.registry.index_artifact_directory(
                out_dir, experiment_name=name, experiment_type="robustness"
            )
            repo.update_status(exp_model.id, status="completed", completed=True)
        except Exception:
            repo.update_status(exp_model.id, status="failed", completed=True)
            raise

        return self.get_experiment_detail(exp_model.id) or {}
