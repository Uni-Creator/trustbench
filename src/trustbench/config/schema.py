"""TrustBench configuration schemas (Pydantic v2)."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONFIG_VERSION = "1"


class MetricConfig(BaseModel):
    """Which metric groups to compute."""

    model_config = ConfigDict(extra="forbid")

    groups: list[str] = Field(
        default=["classification", "calibration", "uncertainty"],
        description="Metric group names to compute.",
    )
    temperature: float | None = Field(
        default=None,
        description="Temperature scaling value for error-detection metrics.",
    )


class ModelConfig(BaseModel):
    """Model loading configuration."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: str | None = Field(
        default=None,
        description="Path to a .pth checkpoint file.",
    )
    arch_name: str | None = Field(
        default=None,
        description="Architecture name registered in trustbench.models (e.g. 'ResNet18').",
    )
    num_classes: int = Field(default=10, ge=1, description="Number of output classes.")
    device: str | None = Field(
        default=None,
        description="Torch device string (e.g. 'cpu', 'cuda:0'). Auto-detected if None.",
    )


class DatasetConfig(BaseModel):
    """Dataset loading configuration."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str | None = Field(
        default=None, description="Root directory for dataset files."
    )
    dataset_type: str = Field(
        default="cifar10",
        description="Dataset type identifier (e.g. 'cifar10').",
    )
    batch_size: int = Field(default=100, ge=1, description="Inference batch size.")
    num_workers: int = Field(
        default=0, ge=0, description="Number of DataLoader worker processes."
    )
    split: str = Field(default="test", description="Dataset split to use.")


class RobustnessConfig(BaseModel):
    """Robustness evaluation configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=False, description="Whether to run robustness eval.")
    corruption_types: list[str] = Field(
        default_factory=list,
        description="List of corruption type names to apply.",
    )
    severity_levels: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 5],
        description="Corruption severity levels (1–5).",
    )


class ReportingConfig(BaseModel):
    """Artifact and report output configuration."""

    model_config = ConfigDict(extra="forbid")

    output_dir: str = Field(
        default="trustbench_output",
        description="Directory for output artifacts.",
    )
    save_html: bool = Field(default=True, description="Generate HTML report.")
    save_json: bool = Field(default=True, description="Save metrics as JSON.")


class TrustBenchConfig(BaseModel):
    """Top-level TrustBench configuration."""

    model_config = ConfigDict(extra="forbid")

    config_version: str = Field(
        default=CONFIG_VERSION, description="Config schema version."
    )
    experiment_name: str = Field(
        default="unnamed_experiment", description="Human-readable experiment label."
    )
    seed: int | None = Field(default=None, description="Global random seed.")

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    metrics: MetricConfig = Field(default_factory=MetricConfig)
    robustness: RobustnessConfig = Field(default_factory=RobustnessConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    # Internal field: set by loader, not part of YAML
    _config_dir: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_config_version(cls, values: Any) -> Any:
        """Accept integer config_version from YAML (e.g. ``config_version: 1``)."""
        if isinstance(values, dict) and "config_version" in values:
            values["config_version"] = str(values["config_version"])
        return values

    def resolve_path(self, relative: str | Path) -> Path:
        """Resolve *relative* against the config file's directory when available."""
        p = Path(relative)
        if p.is_absolute() or self._config_dir is None:
            return p
        return self._config_dir / p
