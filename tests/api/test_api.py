"""Tests for FastAPI endpoints in TrustBench API."""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from trustbench.api.app import app
from trustbench.storage.database import Base, get_db_session


@pytest.fixture
def client_and_session():
    """Create in-memory SQLite session with StaticPool and override FastAPI dependency."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    def override_get_db():
        session = session_factory()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db_session] = override_get_db
    client = TestClient(app)
    yield client, session_factory
    app.dependency_overrides.clear()


def test_health_check(client_and_session):
    client, _ = client_and_session
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "service": "trustbench-api"}


def test_list_experiments_empty(client_and_session):
    client, _ = client_and_session
    response = client.get("/experiments")
    assert response.status_code == 200
    assert response.json() == []


def test_index_and_get_experiment(client_and_session, tmp_path):
    client, _session_factory = client_and_session

    # Prepare temporary experiment artifacts
    exp_dir = tmp_path / "baseline_sample"
    exp_dir.mkdir()
    (exp_dir / "metadata.json").write_text('{"checkpoint_sha256": "fake123", "created_at": "2026-11-03T00:00:00Z"}')
    (exp_dir / "config.json").write_text('{"model": "ResNet18"}')
    (exp_dir / "metrics.json").write_text('{"summary": {"accuracy": 0.95, "macro_f1": 0.94}}')
    (exp_dir / "predictions.npz").write_text('dummy')

    # Call index API
    res = client.post(f"/experiments/index-directory?directory_path={exp_dir}")
    assert res.status_code == 200
    data = res.json()
    assert data["name"] == "baseline_sample"
    exp_id = data["experiment_id"]

    # Get detail API
    detail_res = client.get(f"/experiments/{exp_id}")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["checkpoint_sha256"] == "fake123"
    assert len(detail["metrics"]) >= 2
    assert len(detail["artifacts"]) >= 3
