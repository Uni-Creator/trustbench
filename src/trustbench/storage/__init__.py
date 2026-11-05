"""Storage module for TrustBench."""

from trustbench.storage.base import (
    CorruptedArtifactError,
    ExperimentAlreadyExistsError,
    ExperimentNotFoundError,
    ExperimentRecord,
    ExperimentState,
    ExperimentStore,
    ExperimentSummary,
    IncompatibleExperimentsError,
    StorageError,
)
from trustbench.storage.filesystem import FilesystemExperimentStore

__all__ = [
    "ArtifactModel",
    "Base",
    "CorruptedArtifactError",
    "ExperimentAlreadyExistsError",
    "ExperimentModel",
    "ExperimentNotFoundError",
    "ExperimentRecord",
    "ExperimentRepository",
    "ExperimentState",
    "ExperimentStore",
    "ExperimentSummary",
    "FilesystemExperimentStore",
    "IncompatibleExperimentsError",
    "MetricModel",
    "SQLiteExperimentStore",
    "StorageError",
    "create_db_engine",
    "get_database_url",
    "get_db_session",
    "get_session_factory",
    "init_db",
]


def __getattr__(name: str):
    """Lazily load SQLAlchemy and SQLite models/repositories on demand."""
    if name in (
        "Base",
        "create_db_engine",
        "get_database_url",
        "get_db_session",
        "get_session_factory",
        "init_db",
    ):
        import trustbench.storage.database as db

        return getattr(db, name)
    if name in ("ArtifactModel", "ExperimentModel", "MetricModel"):
        import trustbench.storage.models as m

        return getattr(m, name)
    if name == "ExperimentRepository":
        from trustbench.storage.repositories import ExperimentRepository

        return ExperimentRepository
    if name == "SQLiteExperimentStore":
        from trustbench.storage.sqlite import SQLiteExperimentStore

        return SQLiteExperimentStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
