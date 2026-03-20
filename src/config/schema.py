"""YAML schema loader for configuration files with validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TrainingConfig(BaseModel):
    """Schema for training configuration YAML files."""

    model_id: str = Field(description="Base model identifier")
    dataset: str = Field(description="Dataset name or path")
    output_dir: str = Field(description="Output directory for checkpoints")
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=4, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    lora_rank: int = Field(default=64, ge=1)
    lora_alpha: int = Field(default=128, ge=1)
    use_qlora: bool = Field(default=True)
    use_dora: bool = Field(default=False)


class EvaluationConfig(BaseModel):
    """Schema for evaluation configuration YAML files."""

    model_path: str = Field(description="Path to fine-tuned model")
    benchmarks: list[str] = Field(default_factory=list, description="Benchmark names to run")
    metrics: list[str] = Field(default_factory=list, description="Metrics to compute")
    output_dir: str = Field(default="results/", description="Results output directory")


class DeploymentConfig(BaseModel):
    """Schema for deployment configuration YAML files."""

    endpoint_name: str = Field(description="SageMaker endpoint name")
    instance_type: str = Field(default="ml.g5.2xlarge")
    instance_count: int = Field(default=1, ge=1)
    model_data_url: str = Field(description="S3 URI of model artifacts")


class YAMLSchemaLoader:
    """Load and validate YAML configuration files against Pydantic schemas."""

    _SCHEMA_MAP: dict[str, type[BaseModel]] = {
        "training": TrainingConfig,
        "evaluation": EvaluationConfig,
        "deployment": DeploymentConfig,
    }

    @staticmethod
    def load(path: Path, schema_type: str) -> BaseModel:
        """Load a YAML file and validate against the specified schema type.

        Args:
            path: Path to the YAML configuration file.
            schema_type: One of 'training', 'evaluation', 'deployment'.

        Returns:
            Validated Pydantic model instance.

        Raises:
            ValueError: If schema_type is unknown.
            FileNotFoundError: If path does not exist.
        """
        if schema_type not in YAMLSchemaLoader._SCHEMA_MAP:
            raise ValueError(
                f"Unknown schema type '{schema_type}'. "
                f"Valid types: {list(YAMLSchemaLoader._SCHEMA_MAP.keys())}"
            )

        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        schema_cls = YAMLSchemaLoader._SCHEMA_MAP[schema_type]
        return schema_cls.model_validate(raw)

    @staticmethod
    def load_raw(path: Path) -> dict[str, Any]:
        """Load a YAML file without schema validation.

        Args:
            path: Path to the YAML file.

        Returns:
            Parsed YAML content as a dictionary.
        """
        with open(path) as f:
            result: dict[str, Any] = yaml.safe_load(f)
        return result


__all__: list[str] = [
    "TrainingConfig",
    "EvaluationConfig",
    "DeploymentConfig",
    "YAMLSchemaLoader",
]
