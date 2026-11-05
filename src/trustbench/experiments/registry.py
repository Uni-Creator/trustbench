"""Experiment registry for discovering, registering, and indexing evaluation runs."""

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from trustbench.reporting.hashing import (
    compute_file_sha256_or_none as compute_file_sha256,
)
from trustbench.storage.database import get_session_factory
from trustbench.storage.repositories import ExperimentRepository


class ExperimentRegistry:
    """Registry coordinating persistence and artifact indexing for benchmark runs."""

    def __init__(self, session: Session | None = None):
        self.session_provided = session is not None
        self._session = session

    @property
    def session(self) -> Session:
        if self._session is None:
            factory = get_session_factory()
            self._session = factory()
        return self._session

    def close(self) -> None:
        if not self.session_provided and self._session is not None:
            self._session.close()
            self._session = None

    def register_run(
        self,
        name: str,
        experiment_type: str = "evaluation",
        checkpoint_sha256: str | None = None,
        config: dict[str, Any] | None = None,
        status: str = "pending",
        experiment_id: str | None = None,
    ):
        """Register a new benchmark run in the database."""
        repo = ExperimentRepository(self.session)
        return repo.create(
            name=name,
            experiment_type=experiment_type,
            status=status,
            checkpoint_sha256=checkpoint_sha256,
            config=config or {},
            experiment_id=experiment_id,
        )

    def index_artifact_directory(
        self,
        experiment_dir: str | Path,
        experiment_name: str | None = None,
        experiment_type: str | None = None,
    ):
        """Index a completed on-disk experiment artifact directory into SQLite.

        Reads metadata.json, config.json, metrics.json, summary.json, and corruptions.
        """
        exp_path = Path(experiment_dir)
        if not exp_path.is_dir():
            raise FileNotFoundError(f"Experiment directory not found: {exp_path}")

        name = experiment_name or exp_path.name
        repo = ExperimentRepository(self.session)

        # 1. Read metadata
        meta_file = exp_path / "metadata.json"
        metadata = {}
        checkpoint_sha256 = None
        if meta_file.is_file():
            with open(meta_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            checkpoint_sha256 = metadata.get("checkpoint_sha256")

        # 2. Read config
        cfg_file = exp_path / "config.json"
        config = {}
        if cfg_file.is_file():
            with open(cfg_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            if not checkpoint_sha256:
                checkpoint_sha256 = config.get("checkpoint_sha256")

        # Determine experiment type
        if experiment_type is None:
            if (exp_path / "corruption_results").is_dir() or "robustness" in name:
                experiment_type = "robustness"
            else:
                experiment_type = "baseline"

        # Check if already registered
        existing = repo.get_by_name(name)
        if existing:
            exp_model = existing
            repo.update_status(exp_model.id, status="completed", completed=True)
        else:
            exp_model = repo.create(
                name=name,
                experiment_type=experiment_type,
                status="completed",
                checkpoint_sha256=checkpoint_sha256,
                config=config,
            )
            repo.update_status(exp_model.id, status="completed", completed=True)

        # 3. Index Artifact Files
        for f in exp_path.iterdir():
            if f.is_file():
                art_type = f.stem
                csum = compute_file_sha256(f)
                repo.add_artifact(
                    experiment_id=exp_model.id,
                    artifact_type=art_type,
                    file_path=str(f.resolve()),
                    checksum=csum,
                )

        # Index corruption_results subfolder if present
        corruptions_dir = exp_path / "corruption_results"
        if corruptions_dir.is_dir():
            for f in corruptions_dir.iterdir():
                if f.is_file() and f.suffix == ".json":
                    repo.add_artifact(
                        experiment_id=exp_model.id,
                        artifact_type=f"corruption_{f.stem}",
                        file_path=str(f.resolve()),
                        checksum=compute_file_sha256(f),
                    )

        # Index predictions subfolder if present
        preds_dir = exp_path / "predictions"
        if preds_dir.is_dir():
            for f in preds_dir.iterdir():
                if f.is_file() and f.suffix == ".npz":
                    repo.add_artifact(
                        experiment_id=exp_model.id,
                        artifact_type=f"prediction_{f.stem}",
                        file_path=str(f.resolve()),
                        checksum=compute_file_sha256(f),
                    )

        # 4. Extract and index scalar metrics
        metrics_entries = []

        # From baseline metrics.json
        metrics_file = exp_path / "metrics.json"
        if metrics_file.is_file():
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            summary = metrics_data.get("summary", {})
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    metrics_entries.append({
                        "condition": "Clean",
                        "metric_name": k,
                        "metric_value": float(v),
                    })

        # From robustness summary.json
        summary_file = exp_path / "summary.json"
        if summary_file.is_file():
            with open(summary_file, "r", encoding="utf-8") as f:
                sum_data = json.load(f)
            conditions = sum_data.get("conditions", [])
            for cond in conditions:
                cond_name = cond.get("condition", "Unknown")
                for k, v in cond.items():
                    if k != "condition" and isinstance(v, (int, float)):
                        metrics_entries.append({
                            "condition": cond_name,
                            "metric_name": k,
                            "metric_value": float(v),
                        })

        if metrics_entries:
            repo.add_metrics_bulk(exp_model.id, metrics_entries)

        return exp_model
