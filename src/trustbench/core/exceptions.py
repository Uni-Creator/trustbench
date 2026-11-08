"""Errors raised by TrustBench core evaluation."""


class TrustBenchError(Exception):
    """Base exception for TrustBench core errors."""


class UnsupportedCapabilityError(TrustBenchError):
    """Raised when a model adapter does not implement an optional capability."""


class ShapeError(TrustBenchError):
    """Raised when model outputs do not match the expected shape."""


class SecurityError(TrustBenchError):
    """Raised when an operation violates security boundaries (e.g. path traversal)."""
