"""File hashing helpers for artifact and checkpoint lineage."""

import hashlib
from pathlib import Path


def compute_file_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(65536), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def compute_file_sha256_or_none(filepath: str | Path) -> str | None:
    """Compute SHA-256 hash of a file if it exists."""
    path = Path(filepath)
    if not path.is_file():
        return None
    return compute_file_sha256(path)
