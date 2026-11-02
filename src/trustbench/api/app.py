"""FastAPI application factory for TrustBench API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trustbench.api.routes.experiments import router as experiments_router
from trustbench.storage.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database tables on startup
    init_db()
    yield


def create_app() -> FastAPI:
    """Create and configure TrustBench FastAPI application."""
    app = FastAPI(
        title="TrustBench API",
        description="ML Reliability, Robustness & Uncertainty Evaluation Framework API",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(experiments_router)

    @app.get("/health", tags=["health"])
    def health_check():
        return {"status": "ok", "service": "trustbench-api"}

    return app


app = create_app()
