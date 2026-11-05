"""Compatibility shim for experiment artifact I/O."""

from trustbench.reporting.artifacts import (
    NpEncoder,
    load_experiment_artifacts,
    save_experiment_artifacts,
)

__all__ = [
    "NpEncoder",
    "load_experiment_artifacts",
    "save_experiment_artifacts",
]
