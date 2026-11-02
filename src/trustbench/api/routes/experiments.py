"""FastAPI routes for experiment management, metrics querying, artifacts, and comparisons."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from trustbench.experiments.manager import ExperimentManager
from trustbench.experiments.schemas import (
    ExperimentComparisonRequest,
    ExperimentComparisonResponse,
    ExperimentDetail,
    ExperimentResponse,
)
from trustbench.storage.database import get_db_session

router = APIRouter(prefix="/experiments", tags=["experiments"])


def get_manager(session: Session = Depends(get_db_session)) -> ExperimentManager:
    """Dependency for ExperimentManager."""
    return ExperimentManager(session=session)


@router.get("", response_model=list[ExperimentResponse])
def list_experiments(
    experiment_type: str | None = Query(None, description="Filter by experiment type (e.g. baseline, robustness)"),
    status: str | None = Query(None, description="Filter by status (pending, running, completed, failed)"),
    checkpoint_sha256: str | None = Query(None, description="Filter by checkpoint SHA-256 hash"),
    manager: ExperimentManager = Depends(get_manager),
):
    """List all registered experiments."""
    return manager.list_experiments(
        experiment_type=experiment_type,
        status=status,
        checkpoint_sha256=checkpoint_sha256,
    )


@router.get("/{experiment_id}", response_model=ExperimentDetail)
def get_experiment_detail(
    experiment_id: str,
    manager: ExperimentManager = Depends(get_manager),
):
    """Get full details of a specific experiment."""
    exp = manager.get_experiment_detail(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return exp


@router.post("/compare", response_model=ExperimentComparisonResponse)
def compare_experiments(
    req: ExperimentComparisonRequest,
    manager: ExperimentManager = Depends(get_manager),
):
    """Compare multiple experiments across shared conditions and metrics."""
    result = manager.compare_experiments(
        experiment_ids=req.experiment_ids,
        conditions=req.conditions,
        metrics=req.metrics,
    )
    return result


@router.post("/index-directory")
def index_artifact_directory(
    directory_path: str = Query(..., description="Path to on-disk artifact directory"),
    experiment_name: str | None = Query(None, description="Optional custom name"),
    manager: ExperimentManager = Depends(get_manager),
):
    """Index an on-disk artifact folder into the database registry."""
    try:
        model = manager.registry.index_artifact_directory(directory_path, experiment_name=experiment_name)
        return {
            "message": "Experiment indexed successfully.",
            "experiment_id": model.id,
            "name": model.name,
            "status": model.status,
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, RuntimeError, OSError) as e:
        raise HTTPException(status_code=500, detail=str(e))
