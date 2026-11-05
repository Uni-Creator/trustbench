"""Reproducibility utilities for seeding and system environment recording."""

import os
import platform
import random
import subprocess
import sys
from typing import Any

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set random seeds across standard Python, NumPy, and PyTorch if available.

    Args:
        seed: Integer seed.
        deterministic: Whether to set PyTorch cudnn backend to deterministic.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if "torch" not in sys.modules:
        return

    try:
        import torch
    except ImportError:
        return

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


def get_environment_info() -> dict[str, Any]:
    """Capture current system and dependency versions for reproducibility."""
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "os": platform.platform(),
        "git_commit": get_git_commit(),
        "torch_version": None,
        "cuda_available": "False",
        "cuda_device_name": None,
    }
    if "torch" not in sys.modules:
        return info

    try:
        import torch
    except ImportError:
        return info

    info["torch_version"] = torch.__version__
    info["cuda_available"] = str(torch.cuda.is_available())
    info["cuda_device_name"] = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    )
    return info
