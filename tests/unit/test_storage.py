"""Unit tests for storage and experiment management."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from trustbench.experiments.comparison import ExperimentComparator
from trustbench.experiments.manager import ExperimentManager
from trustbench.experiments.registry import ExperimentRegistry
from trustbench.storage.database import Base
from trustbench.storage.repositories import ExperimentRepository


@pytest.fixture
def test_db_session():
    """In-memory SQLite database session for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def test_experiment_crud(test_db_session):
    repo = ExperimentRepository(test_db_session)
    exp = repo.create(
        name="test_run_01",
        experiment_type="baseline",
        status="pending",
        checkpoint_sha256="abc123sha",
        config={"lr": 0.01},
    )
    assert exp.id is not None
    assert exp.name == "test_run_01"
    assert exp.status == "pending"

    # Update status
    updated = repo.update_status(exp.id, status="completed", completed=True)
    assert updated.status == "completed"
    assert updated.completed_at is not None

    # Query
    found = repo.get_by_id(exp.id)
    assert found is not None
    assert found.name == "test_run_01"

    # Metrics
    repo.add_metric(exp.id, condition="Clean", metric_name="accuracy", metric_value=0.95)
    metrics = repo.get_metrics(exp.id)
    assert len(metrics) == 1
    assert metrics[0].metric_value == 0.95

    # Artifacts
    repo.add_artifact(exp.id, artifact_type="metadata", file_path="/path/metadata.json", checksum="hash1")
    artifacts = repo.get_artifacts(exp.id)
    assert len(artifacts) == 1
    assert artifacts[0].checksum == "hash1"


def test_experiment_comparison(test_db_session):
    repo = ExperimentRepository(test_db_session)
    e1 = repo.create(name="exp_1", checkpoint_sha256="hash1")
    e2 = repo.create(name="exp_2", checkpoint_sha256="hash2")

    repo.add_metric(e1.id, condition="Clean", metric_name="accuracy", metric_value=0.95)
    repo.add_metric(e2.id, condition="Clean", metric_name="accuracy", metric_value=0.92)
    repo.add_metric(e1.id, condition="Noise S1", metric_name="accuracy", metric_value=0.88)
    repo.add_metric(e2.id, condition="Noise S1", metric_name="accuracy", metric_value=0.85)

    comparator = ExperimentComparator(session=test_db_session)
    res = comparator.compare([e1.id, e2.id])

    assert len(res["experiment_ids"]) == 2
    assert "Clean" in res["comparison_table"]
    assert res["comparison_table"]["Clean"]["accuracy"][e1.id] == 0.95
    assert res["comparison_table"]["Clean"]["accuracy"][e2.id] == 0.92


def test_experiment_manager_listing(test_db_session):
    repo = ExperimentRepository(test_db_session)
    repo.create(name="run_a", experiment_type="baseline")
    repo.create(name="run_b", experiment_type="robustness")

    manager = ExperimentManager(session=test_db_session)
    exps = manager.list_experiments()
    assert len(exps) == 2

    baselines = manager.list_experiments(experiment_type="baseline")
    assert len(baselines) == 1
    assert baselines[0]["name"] == "run_a"
