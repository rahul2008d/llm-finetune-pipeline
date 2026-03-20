"""Pydantic settings models for application and training configuration."""

from pydantic import Field
from pydantic_settings import BaseSettings


class AppSettings(BaseSettings):
    """Application-level settings loaded from environment variables."""

    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Log output format")
    aws_region: str = Field(default="us-east-1", description="AWS region")
    s3_bucket_data: str = Field(default="", description="S3 bucket for datasets")
    s3_bucket_models: str = Field(default="", description="S3 bucket for model artifacts")
    s3_bucket_artifacts: str = Field(default="", description="S3 bucket for pipeline artifacts")

    model_config = {"env_prefix": "", "case_sensitive": False}


class TrainingSettings(BaseSettings):
    """Training hyperparameters and model configuration."""

    base_model_id: str = Field(
        default="meta-llama/Llama-2-7b-hf", description="HuggingFace model identifier"
    )
    huggingface_token: str = Field(default="", description="HuggingFace API token")
    epochs: int = Field(default=3, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=4, ge=1, description="Per-device batch size")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate")
    lora_rank: int = Field(default=64, ge=1, description="LoRA rank")
    lora_alpha: int = Field(default=128, ge=1, description="LoRA alpha")
    max_seq_length: int = Field(default=2048, ge=1, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation")
    warmup_ratio: float = Field(default=0.03, ge=0, le=1, description="Warmup ratio")
    weight_decay: float = Field(default=0.001, ge=0, description="Weight decay")

    model_config = {"env_prefix": "TRAIN_", "case_sensitive": False}


__all__: list[str] = ["AppSettings", "TrainingSettings"]
