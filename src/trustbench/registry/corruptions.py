"""Corruption registration interfaces for user and plugin custom image corruptions."""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = Any  # type: ignore[assignment,misc]

from trustbench.robustness.severity import SeverityConfig


@dataclass
class RegisteredCorruption:
    """Wrapper holding a registered corruption function and its severity definitions."""

    name: str
    fn: Callable[..., Image.Image]
    severities: dict[int, SeverityConfig]
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def apply(self, image: Image.Image, severity: int, seed: int | None = None) -> Image.Image:
        """Apply the corruption at the given severity level."""
        if severity not in self.severities:
            raise ValueError(
                f"Severity level {severity} not configured for corruption {self.name!r}. "
                f"Available levels: {sorted(self.severities.keys())}."
            )
        cfg = self.severities[severity]
        return self.fn(image, cfg.param_value, seed=seed)


class CorruptionRegistry:
    """Thread-safe registry for custom corruption operations."""

    def __init__(self) -> None:
        self._registry: dict[str, RegisteredCorruption] = {}

    def register(
        self,
        name: str | None = None,
        fn: Callable[..., Image.Image] | None = None,
        *,
        severities: dict[int, SeverityConfig | float | int] | None = None,
        param_name: str = "severity_param",
        unit: str = "custom",
        description: str = "",
        metadata: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> Any:
        """Register a custom image corruption function.

        Input contract for `fn`:
            `fn(image: Image.Image, param_value: Any, seed: int | None = None) -> Image.Image`
        """
        def decorator(func: Callable[..., Image.Image]) -> Callable[..., Image.Image]:
            corr_name = name or getattr(func, "__name__", None)
            if not corr_name or not isinstance(corr_name, str):
                raise ValueError("Corruption name must be a non-empty string.")

            if not callable(func):
                raise TypeError(f"Corruption function {func!r} is not callable.")

            if corr_name in self._registry and not overwrite:
                raise ValueError(
                    f"Corruption {corr_name!r} already registered. Set overwrite=True to replace."
                )

            # Build severity configs dictionary
            sev_dict: dict[int, SeverityConfig] = {}
            if severities is not None:
                for s_lvl, s_val in severities.items():
                    if isinstance(s_val, SeverityConfig):
                        sev_dict[s_lvl] = s_val
                    else:
                        sev_dict[s_lvl] = SeverityConfig(
                            severity=int(s_lvl),
                            param_name=param_name,
                            param_value=s_val,
                            unit=unit,
                            description=f"{corr_name} at severity {s_lvl} ({param_name}={s_val})",
                        )
            else:
                # Default 1 to 5 linear severities if not provided
                for s_lvl in range(1, 6):
                    sev_dict[s_lvl] = SeverityConfig(
                        severity=s_lvl,
                        param_name=param_name,
                        param_value=s_lvl,
                        unit=unit,
                        description=f"{corr_name} at severity {s_lvl}",
                    )

            self._registry[corr_name] = RegisteredCorruption(
                name=corr_name,
                fn=func,
                severities=sev_dict,
                description=description or getattr(func, "__doc__", "") or "",
                metadata=metadata or {},
            )
            return func

        if fn is not None:
            return decorator(fn)
        return decorator

    def get(self, name: str) -> RegisteredCorruption | None:
        """Retrieve a registered corruption by name."""
        return self._registry.get(name)

    def is_registered(self, name: str) -> bool:
        """Check if a corruption name is registered."""
        return name in self._registry

    def list_corruptions(self) -> dict[str, RegisteredCorruption]:
        """List all registered custom corruptions."""
        return dict(self._registry)

    def unregister(self, name: str) -> bool:
        """Unregister a corruption by name."""
        if name in self._registry:
            del self._registry[name]
            return True
        return False

    def clear(self) -> None:
        """Clear all registered custom corruptions."""
        self._registry.clear()


CORRUPTION_REGISTRY = CorruptionRegistry()


def register_corruption(
    name: str | None = None,
    fn: Callable[..., Image.Image] | None = None,
    *,
    severities: dict[int, SeverityConfig | float | int] | None = None,
    param_name: str = "severity_param",
    unit: str = "custom",
    description: str = "",
    metadata: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> Any:
    """Register a custom corruption function.

    Example::

        @register_corruption("snow", severities={1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.5})
        def apply_snow(image, param_value, seed=None):
            ...
            return corrupted_image
    """
    return CORRUPTION_REGISTRY.register(
        name=name,
        fn=fn,
        severities=severities,
        param_name=param_name,
        unit=unit,
        description=description,
        metadata=metadata,
        overwrite=overwrite,
    )


def get_registered_corruption(name: str) -> RegisteredCorruption | None:
    """Retrieve a registered custom corruption by name."""
    return CORRUPTION_REGISTRY.get(name)


def list_registered_corruptions() -> dict[str, RegisteredCorruption]:
    """List all registered custom corruptions."""
    return CORRUPTION_REGISTRY.list_corruptions()


def unregister_corruption(name: str) -> bool:
    """Unregister a custom corruption."""
    return CORRUPTION_REGISTRY.unregister(name)
