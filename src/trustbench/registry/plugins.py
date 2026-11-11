"""Plugin discovery and lifecycle management via Python entry points."""

import importlib
import importlib.metadata
import logging
from dataclasses import dataclass, field
from typing import Any

try:
    CURRENT_VERSION = importlib.metadata.version("trustbench")
except importlib.metadata.PackageNotFoundError:
    CURRENT_VERSION = "0.2.2"

logger = logging.getLogger(__name__)

PLUGIN_ENTRY_POINT_GROUP = "trustbench.plugins"


class PluginError(Exception):
    """Base error for plugin loading and lifecycle failures."""


class PluginVersionMismatchError(PluginError):
    """Raised when a plugin requires an incompatible TrustBench version."""


@dataclass
class PluginInfo:
    """Information and lifecycle status of a discovered or loaded plugin."""

    name: str
    entry_point_str: str
    status: str  # "loaded", "failed", "incompatible"
    version: str | None = None
    min_trustbench_version: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _check_version_compatibility(required_min: str | None, current: str) -> bool:
    """Simple semantic version comparison helper (e.g. '0.2.0' vs '0.1.0')."""
    if not required_min:
        return True
    try:
        req_parts = [int(p) for p in required_min.split(".")[:3]]
        cur_parts = [int(p) for p in current.split(".")[:3]]
        return cur_parts >= req_parts
    except (ValueError, TypeError):
        return True


class PluginManager:
    """Manages plugin discovery, validation, and lifecycle without arbitrary file execution."""

    def __init__(self) -> None:
        self._plugins: dict[str, PluginInfo] = {}

    def load_plugin(
        self,
        name_or_module: str | Any,
        *,
        name: str | None = None,
        safe: bool = True,
    ) -> PluginInfo:
        """Load and register a plugin from a module path or imported module.

        Args:
            name_or_module: Dotted module name string or imported module object.
            name: Optional explicit name.
            safe: If True, catches exceptions and records error in PluginInfo instead of raising.

        Returns:
            PluginInfo instance.
        """
        mod_name = name_or_module if isinstance(name_or_module, str) else getattr(name_or_module, "__name__", "unknown")
        plugin_name = name or mod_name.split(".")[-1]

        try:
            if isinstance(name_or_module, str):
                mod = importlib.import_module(name_or_module)
            else:
                mod = name_or_module

            # Check version compatibility if specified
            min_ver = getattr(mod, "__min_trustbench_version__", None)
            plugin_ver = getattr(mod, "__version__", None)

            if min_ver and not _check_version_compatibility(min_ver, CURRENT_VERSION):
                msg = (
                    f"Plugin {plugin_name!r} requires TrustBench >= {min_ver}, "
                    f"but current version is {CURRENT_VERSION}."
                )
                if not safe:
                    raise PluginVersionMismatchError(msg)
                info = PluginInfo(
                    name=plugin_name,
                    entry_point_str=mod_name,
                    status="incompatible",
                    version=plugin_ver,
                    min_trustbench_version=min_ver,
                    error=msg,
                )
                self._plugins[plugin_name] = info
                return info

            # Execute setup() / register() hook if provided
            setup_fn = getattr(mod, "setup", None)
            register_fn = getattr(mod, "register", None)
            if callable(setup_fn):
                setup_fn()
            elif callable(register_fn):
                register_fn()

            info = PluginInfo(
                name=plugin_name,
                entry_point_str=mod_name,
                status="loaded",
                version=plugin_ver,
                min_trustbench_version=min_ver,
            )
            self._plugins[plugin_name] = info
            return info

        except PluginError:
            if not safe:
                raise
            info = PluginInfo(
                name=plugin_name,
                entry_point_str=mod_name,
                status="failed",
            )
            self._plugins[plugin_name] = info
            return info
        except Exception as exc:
            logger.warning("Failed to load TrustBench plugin %r: %s", plugin_name, exc)
            if not safe:
                raise PluginError(f"Plugin load failure for {plugin_name!r}: {exc}") from exc
            info = PluginInfo(
                name=plugin_name,
                entry_point_str=mod_name,
                status="failed",
                error=str(exc),
            )
            self._plugins[plugin_name] = info
            return info

    def discover(
        self,
        group: str = PLUGIN_ENTRY_POINT_GROUP,
        safe: bool = True,
    ) -> dict[str, PluginInfo]:
        """Discover and load all plugins registered via Python entry points in `group`.

        Args:
            group: Entry point group name (default: 'trustbench.plugins').
            safe: If True, failures in individual plugins do not abort discovery.

        Returns:
            Dictionary of plugin name to PluginInfo.
        """
        try:
            eps = importlib.metadata.entry_points()
            # Python 3.12+ vs 3.10/3.11 compatibility for entry_points
            if hasattr(eps, "select"):
                matching_eps = eps.select(group=group)
            else:
                matching_eps = eps.get(group, [])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error querying entry points for group %r: %s", group, exc)
            return dict(self._plugins)

        for ep in matching_eps:
            plugin_name = ep.name
            try:
                plugin_obj = ep.load()
                self.load_plugin(plugin_obj, name=plugin_name, safe=safe)
            except Exception as exc:
                logger.warning("Failed to load entry point %r: %s", ep.name, exc)
                if not safe:
                    raise PluginError(f"Entry point {ep.name!r} failed to load: {exc}") from exc
                self._plugins[plugin_name] = PluginInfo(
                    name=plugin_name,
                    entry_point_str=str(ep.value),
                    status="failed",
                    error=str(exc),
                )

        return dict(self._plugins)

    def get_plugin(self, name: str) -> PluginInfo | None:
        """Get information for a discovered or loaded plugin."""
        return self._plugins.get(name)

    def list_plugins(self) -> dict[str, PluginInfo]:
        """List all tracked plugins."""
        return dict(self._plugins)

    def clear(self) -> None:
        """Clear all registered plugin states."""
        self._plugins.clear()


# Global plugin manager
PLUGIN_MANAGER = PluginManager()


def discover_plugins(
    group: str = PLUGIN_ENTRY_POINT_GROUP,
    safe: bool = True,
) -> dict[str, PluginInfo]:
    """Discover and load all registered TrustBench plugins via entry points."""
    return PLUGIN_MANAGER.discover(group=group, safe=safe)


def load_plugin(
    name_or_module: str | Any,
    *,
    name: str | None = None,
    safe: bool = True,
) -> PluginInfo:
    """Explicitly load a plugin from module string or object."""
    return PLUGIN_MANAGER.load_plugin(name_or_module, name=name, safe=safe)
