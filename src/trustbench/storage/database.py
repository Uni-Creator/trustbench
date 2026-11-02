"""SQLAlchemy database connection and session management."""

import os
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()

DEFAULT_DB_PATH = Path("./artifacts/trustbench.db")


def get_database_url(db_path: str | Path | None = None) -> str:
    """Get SQLite database URL."""
    if db_path is None:
        db_path = os.getenv("TRUSTBENCH_DB_PATH", str(DEFAULT_DB_PATH))
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path.resolve()}"


def create_db_engine(db_url: str | None = None):
    """Create a SQLAlchemy engine for SQLite."""
    if db_url is None:
        db_url = get_database_url()
    return create_engine(
        db_url,
        connect_args={"check_same_thread": False},
        echo=False,
    )


_engine = None
_SessionFactory = None


def init_db(engine=None) -> None:
    """Initialize database schema."""
    global _engine, _SessionFactory
    if engine is None:
        if _engine is None:
            _engine = create_db_engine()
        engine = _engine
    else:
        _engine = engine

    Base.metadata.create_all(bind=engine)
    _SessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_session_factory():
    """Retrieve or create session factory."""
    if _SessionFactory is None:
        init_db()
    return _SessionFactory


def get_db_session() -> Generator[Session, None, None]:
    """Dependency yielding a database session."""
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()
