"""Reproducibility utilities for seeding and system environment recording."""

import os
import platform
import random
import subprocess

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across standard Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed.
        deterministic: Whether to set PyTorch cudnn backend to deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_git_commit() -> str | None:
    """Get the current git commit hash if in a git repository."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
        return commit
    except (subprocess.SubprocessError, OSError):
        return None


def get_environment_info() -> dict[str, str | None]:
    """Capture current system and dependency versions for reproducibility."""
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "os": platform.platform(),
        "git_commit": get_git_commit(),
    }
