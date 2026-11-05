"""SQLite and SQLAlchemy-backed experiment storage implementation."""

from __future__ import annotations

import json
import logging
import shutil

logger = logging.getLogger(__name__)
import tarfile
import tempfile
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from trustbench.reporting.artifacts import save_experiment_artifacts
from trustbench.reporting.hashing import compute_file_sha256
from trustbench.storage.base import (
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRecord,
    ExperimentState,
    ExperimentSummary,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

    from trustbench.core.results import EvaluationResult




class SQLiteExperimentStore:
    """SQLAlchemy-backed SQLite experiment store conforming to ExperimentStore protocol."""

    def __init__(
        self,
        session: Session | None = None,
        db_path: str | Path | None = None,
        artifacts_dir: str | Path = "./artifacts/experiments",
    ) -> None:
        self._session_provided = session is not None
        self._session = session
        self.db_path = db_path
        self.artifacts_dir = Path(artifacts_dir).resolve()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    @property
    def session(self) -> Session:
        """Lazily initialize and return database session."""
        if self._session is None:
            from trustbench.storage.database import (
                create_db_engine,
                get_database_url,
                init_db,
                sessionmaker,
            )

            url = get_database_url(self.db_path)
            engine = create_db_engine(url)
            init_db(engine)
            maker = sessionmaker(autocommit=False, autoflush=False, bind=engine)
            self._session = maker()
        return self._session

    def close(self) -> None:
        """Close managed database session if owned."""
        if not self._session_provided and self._session is not None:
            self._session.close()
            self._session = None

    def _get_repo(self):
        from trustbench.storage.repositories import ExperimentRepository

        return ExperimentRepository(self.session)

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
        """Create and initialize a new experiment in the database."""
        repo = self._get_repo()
        exp_id = experiment_id or str(uuid.uuid4())

        existing_id = repo.get_by_id(exp_id)
        if existing_id and not overwrite:
            raise ExperimentAlreadyExistsError(f"Experiment with ID {exp_id!r} already exists.")

        existing_name = repo.get_by_name(name)
        if existing_name and existing_name.id != exp_id and not overwrite:
            raise ExperimentAlreadyExistsError(
                f"Experiment with name {name!r} already exists (ID: {existing_name.id})."
            )

        if existing_id and overwrite:
            repo.delete(exp_id)

        meta = metadata or {}
        cfg = config or {}
        ckpt_sha = meta.get("checkpoint_sha256") or cfg.get("checkpoint_sha256")

        model = repo.create(
            name=name,
            experiment_type=experiment_type,
            status=ExperimentState.PENDING.value,
            checkpoint_sha256=ckpt_sha,
            config=cfg,
            experiment_id=exp_id,
        )

        return self.get(model.id)  # type: ignore[return-value]

    def update_status(
        self,
        experiment_id: str,
        status: str | ExperimentState,
        *,
        error: str | None = None,
        completed: bool = False,
    ) -> ExperimentRecord:
        """Update status and completion timestamp in SQLite."""
        repo = self._get_repo()
        status_str = status.value if isinstance(status, ExperimentState) else str(status)
        is_completed = completed or status_str in (
            ExperimentState.COMPLETED.value,
            ExperimentState.FAILED.value,
        )

        exp = repo.get_by_id(experiment_id)
        if not exp:
            exp = repo.get_by_name(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        updated = repo.update_status(exp.id, status=status_str, completed=is_completed)
        if not updated:
            raise ExperimentNotFoundError(f"Failed to update experiment {experiment_id!r}.")

        return self.get(exp.id)  # type: ignore[return-value]

    def save_result(
        self,
        experiment_id: str,
        result: EvaluationResult,
        *,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ExperimentRecord:
        """Save evaluation result artifacts to disk and index into SQLite."""
        repo = self._get_repo()
        exp = repo.get_by_id(experiment_id)
        if not exp:
            exp = repo.get_by_name(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self.artifacts_dir / exp.id
        exp_dir.mkdir(parents=True, exist_ok=True)

        merged_meta = {**(exp.config or {}), **result.metadata}
        if extra_metadata:
            merged_meta.update(extra_metadata)

        # 1. Save artifact files
        save_experiment_artifacts(
            output_dir=exp_dir,
            metadata=merged_meta,
            metrics=result.metrics,
            config=exp.config or {},
            predictions=result.to_predictions_dict(),
        )

        # 2. Record artifacts in DB with checksums
        checksums = {}
        for fname, atype in [
            ("metrics.json", "metrics"),
            ("metadata.json", "metadata"),
            ("config.json", "config"),
            ("predictions.npz", "predictions"),
        ]:
            fpath = exp_dir / fname
            if fpath.is_file():
                sha = compute_file_sha256(fpath)
                checksums[fname] = sha
                repo.add_artifact(
                    experiment_id=exp.id,
                    artifact_type=atype,
                    file_path=str(fpath),
                    checksum=sha,
                )

        # 3. Record metrics in DB
        metrics_to_add: list[dict[str, Any]] = []

        def _flatten_metrics(prefix: str, data: Any):
            if isinstance(data, dict):
                for k, v in data.items():
                    sub_prefix = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metrics_to_add.append({
                            "condition": "Clean",
                            "metric_name": sub_prefix,
                            "metric_value": float(v),
                        })
                    elif isinstance(v, dict):
                        _flatten_metrics(sub_prefix, v)

        _flatten_metrics("", result.metrics)
        if metrics_to_add:
            repo.add_metrics_bulk(exp.id, metrics_to_add)

        # Update status
        repo.update_status(exp.id, status=ExperimentState.COMPLETED.value, completed=True)
        return self.get(exp.id)  # type: ignore[return-value]

    def get(self, experiment_id_or_name: str) -> ExperimentRecord | None:
        """Retrieve full experiment record from SQLite."""
        repo = self._get_repo()
        exp = repo.get_by_id(experiment_id_or_name)
        if not exp:
            exp = repo.get_by_name(experiment_id_or_name)
        if not exp:
            return None

        metrics_models = repo.get_metrics(exp.id)
        artifact_models = repo.get_artifacts(exp.id)

        # Reconstruct metrics dictionary
        metrics_dict: dict[str, Any] = {"summary": {}}
        for m in metrics_models:
            metrics_dict["summary"][m.metric_name] = m.metric_value

        artifacts_list = [
            {
                "id": a.id,
                "artifact_type": a.artifact_type,
                "file_path": a.file_path,
                "checksum": a.checksum,
            }
            for a in artifact_models
        ]

        # Read full metrics.json and metadata.json if artifacts directory exists
        exp_dir = self.artifacts_dir / exp.id
        meta_data: dict[str, Any] = {}
        if exp_dir.is_dir():
            m_file = exp_dir / "metrics.json"
            if m_file.is_file():
                try:
                    with open(m_file, "r", encoding="utf-8") as f:
                        metrics_dict = json.load(f)
                except Exception:
                    logger.debug("Failed to load metrics from %s", m_file, exc_info=True)
            meta_file = exp_dir / "metadata.json"
            if meta_file.is_file():
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                except Exception:
                    logger.debug("Failed to load metadata from %s", meta_file, exc_info=True)

        created_str = exp.created_at.isoformat() if hasattr(exp.created_at, "isoformat") else str(exp.created_at)
        completed_str = exp.completed_at.isoformat() if exp.completed_at and hasattr(exp.completed_at, "isoformat") else None

        return ExperimentRecord(
            id=exp.id,
            name=exp.name,
            experiment_type=exp.experiment_type,
            status=exp.status,
            created_at=created_str,
            completed_at=completed_str,
            checkpoint_sha256=exp.checkpoint_sha256,
            dataset_fingerprint=meta_data.get("dataset_fingerprint"),
            config=exp.config or {},
            metadata=meta_data,
            metrics=metrics_dict,
            artifacts=artifacts_list,
        )

    def list(
        self,
        *,
        experiment_type: str | None = None,
        status: str | None = None,
        checkpoint_sha256: str | None = None,
    ) -> list[ExperimentSummary]:
        """List experiments from SQLite."""
        repo = self._get_repo()
        models = repo.list_all(
            experiment_type=experiment_type,
            status=status,
            checkpoint_sha256=checkpoint_sha256,
        )
        summaries: list[ExperimentSummary] = []
        for m in models:
            metrics_models = repo.get_metrics(m.id)
            artifact_models = repo.get_artifacts(m.id)
            summary_metrics = {met.metric_name: met.metric_value for met in metrics_models[:10]}

            created_str = m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at)
            completed_str = m.completed_at.isoformat() if m.completed_at and hasattr(m.completed_at, "isoformat") else None

            summaries.append(
                ExperimentSummary(
                    id=m.id,
                    name=m.name,
                    experiment_type=m.experiment_type,
                    status=m.status,
                    created_at=created_str,
                    completed_at=completed_str,
                    checkpoint_sha256=m.checkpoint_sha256,
                    metrics_summary=summary_metrics,
                    artifacts_count=len(artifact_models),
                    config=m.config or {},
                )
            )
        return summaries

    def delete(self, experiment_id: str, *, delete_files: bool = True) -> bool:
        """Delete an experiment from database and optionally remove artifacts."""
        repo = self._get_repo()
        exp = repo.get_by_id(experiment_id)
        if not exp:
            exp = repo.get_by_name(experiment_id)
        if not exp:
            return False

        eid = exp.id
        deleted = repo.delete(eid)
        if deleted and delete_files:
            exp_dir = self.artifacts_dir / eid
            if exp_dir.is_dir():
                shutil.rmtree(exp_dir)
        return deleted

    def verify_checksums(self, experiment_id: str) -> dict[str, bool]:
        """Verify artifact checksums recorded in the database."""
        repo = self._get_repo()
        exp = repo.get_by_id(experiment_id)
        if not exp:
            exp = repo.get_by_name(experiment_id)
        if not exp:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        artifacts = repo.get_artifacts(exp.id)
        results: dict[str, bool] = {}
        for art in artifacts:
            p = Path(art.file_path)
            if not p.is_file():
                results[art.artifact_type] = False
                continue
            if not art.checksum:
                results[art.artifact_type] = True
                continue
            actual = compute_file_sha256(p)
            results[art.artifact_type] = actual == art.checksum

        return results

    def export(self, experiment_id: str, export_path: str | Path) -> Path:
        """Export experiment artifacts to a portable tar.gz archive."""
        rec = self.get(experiment_id)
        if rec is None:
            raise ExperimentNotFoundError(f"Experiment {experiment_id!r} not found.")

        exp_dir = self.artifacts_dir / rec.id
        if not exp_dir.is_dir():
            raise FileNotFoundError(f"Artifact directory for experiment {rec.id} not found.")

        target = Path(export_path).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)

        with tarfile.open(target, "w:gz") as tar:
            for item in exp_dir.iterdir():
                tar.add(item, arcname=item.name)

        return target

    def import_experiment(
        self,
        archive_path: str | Path,
        *,
        overwrite: bool = False,
    ) -> ExperimentRecord:
        """Import experiment from archive into SQLite and artifacts directory."""
        src = Path(archive_path).resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Archive not found: {src}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            with tarfile.open(src, "r:*") as tar:
                tar.extractall(path=tmp_path, filter="data")

            meta_file = tmp_path / "metadata.json"
            cfg_file = tmp_path / "config.json"
            metrics_file = tmp_path / "metrics.json"

            metadata = {}
            if meta_file.is_file():
                with open(meta_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

            config = {}
            if cfg_file.is_file():
                with open(cfg_file, "r", encoding="utf-8") as f:
                    config = json.load(f)

            exp_id = metadata.get("experiment_id") or str(uuid.uuid4())
            exp_name = metadata.get("experiment_name") or src.stem.split(".")[0]

            dest_dir = self.artifacts_dir / exp_id
            if dest_dir.exists():
                if not overwrite:
                    raise ExperimentAlreadyExistsError(f"Experiment {exp_id!r} already exists.")
                shutil.rmtree(dest_dir)

            dest_dir.mkdir(parents=True, exist_ok=True)
            for item in tmp_path.iterdir():
                shutil.copy2(item, dest_dir / item.name)

            self.create(
                name=exp_name,
                experiment_type=metadata.get("experiment_type", "evaluation"),
                config=config,
                experiment_id=exp_id,
                metadata=metadata,
                overwrite=overwrite,
            )

            # Index metrics if present
            if metrics_file.is_file():
                with open(metrics_file, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
                repo = self._get_repo()
                metrics_to_add = []
                for k, v in metrics_data.get("summary", {}).items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        metrics_to_add.append({
                            "condition": "Clean",
                            "metric_name": k,
                            "metric_value": float(v),
                        })
                if metrics_to_add:
                    repo.add_metrics_bulk(exp_id, metrics_to_add)

            self.update_status(exp_id, status=ExperimentState.COMPLETED.value, completed=True)
            return self.get(exp_id)  # type: ignore[return-value]
