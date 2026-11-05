"""Load and validate TrustBench configuration from YAML files."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from trustbench.config.schema import TrustBenchConfig


def load_config(config_path: str | Path) -> TrustBenchConfig:
    """Load a TrustBench YAML config file and return a validated :class:`TrustBenchConfig`.

    Args:
        config_path: Path to the ``trustbench.yaml`` (or any ``.yml``/``.yaml``) file.

    Returns:
        Validated :class:`TrustBenchConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the YAML is structurally invalid or fails Pydantic validation.
        TypeError: If the top-level YAML value is not a mapping.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")

    # Safe YAML load — never executes arbitrary Python
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc

    if data is None:
        data = {}

    if not isinstance(data, dict):
        raise TypeError(
            f"Expected a YAML mapping at the top level in {path}, "
            f"got {type(data).__name__}."
        )

    try:
        config = TrustBenchConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Config validation failed for {path}:\n{exc}") from exc

    # Attach the config file's directory so relative paths can be resolved
    object.__setattr__(config, "_config_dir", path.parent.resolve())
    return config


def validate_config(config_path: str | Path) -> list[str]:
    """Validate a config file and return a list of error strings (empty = valid).

    Args:
        config_path: Path to the config file.

    Returns:
        List of human-readable error messages; empty if the config is valid.
    """
    try:
        load_config(config_path)
        return []
    except FileNotFoundError as exc:
        return [str(exc)]
    except (ValueError, TypeError) as exc:
        return [str(exc)]
