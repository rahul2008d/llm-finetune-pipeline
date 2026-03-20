"""Pydantic v2 configuration models for LLM fine-tuning training."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, model_validator


# ── sub-configs ─────────────────────────────────────────────────


class ModelConfig(BaseModel):
    """Base-model loading configuration."""

    model_name_or_path: str = Field(
        description="HuggingFace model ID or S3 path to model artefacts",
    )
    trust_remote_code: bool = False
    torch_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    attn_implementation: Literal[
        "eager", "sdpa", "flash_attention_2"
    ] = "flash_attention_2"
    max_seq_length: int = Field(ge=128, le=32768, default=4096)
    use_cache: bool = False  # must be False for gradient checkpointing


class QuantizationConfig(BaseModel):
    """BitsAndBytes 4-bit quantization settings."""

    method: Literal["qlora", "dora"] = "qlora"
    load_in_4bit: bool = True
    bnb_4bit_quant_type: Literal["nf4", "fp4"] = "nf4"
    bnb_4bit_compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True  # nested quantization


class LoRAConfig(BaseModel):
    """LoRA / DoRA / RS-LoRA adapter configuration."""

    r: int = Field(ge=4, le=256, default=64)
    lora_alpha: int = Field(ge=1, le=512, default=128)
    lora_dropout: float = Field(ge=0.0, le=0.5, default=0.05)
    target_modules: list[str] = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"
    use_dora: bool = False  # True enables DoRA (Weight-Decomposed LoRA)
    use_rslora: bool = False  # Rank-Stabilized LoRA
    modules_to_save: list[str] | None = None  # e.g. ["embed_tokens", "lm_head"]

    @model_validator(mode="after")
    def _dora_method_consistency(self) -> LoRAConfig:
        """When created standalone, ``use_dora`` is accepted as-is.

        Cross-model validation with :class:`QuantizationConfig` is
        handled by :class:`TrainingJobConfig`.
        """
        return self


class TrainingHyperparameters(BaseModel):
    """Core training hyper-parameters (mirrors HF TrainingArguments)."""

    num_train_epochs: int = Field(ge=1, le=20, default=3)
    per_device_train_batch_size: int = Field(ge=1, le=64, default=4)
    per_device_eval_batch_size: int = Field(ge=1, le=64, default=8)
    gradient_accumulation_steps: int = Field(ge=1, le=128, default=4)
    learning_rate: float = Field(ge=1e-6, le=1e-2, default=2e-4)
    weight_decay: float = Field(ge=0.0, le=0.5, default=0.01)
    warmup_ratio: float = Field(ge=0.0, le=0.3, default=0.03)
    lr_scheduler_type: Literal[
        "cosine", "linear", "constant", "cosine_with_restarts"
    ] = "cosine"
    max_grad_norm: float = Field(ge=0.0, le=10.0, default=1.0)
    optim: str = "paged_adamw_8bit"
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    gradient_checkpointing_kwargs: dict = Field(
        default_factory=lambda: {"use_reentrant": False},
    )
    logging_steps: int = 10
    eval_strategy: str = "steps"
    eval_steps: int = 100
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    seed: int = 42
    data_seed: int = 42
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    group_by_length: bool = True
    report_to: list[str] = Field(default_factory=lambda: ["mlflow"])
    no_cuda: bool = False

    @model_validator(mode="after")
    def _bf16_fp16_exclusive(self) -> TrainingHyperparameters:
        if self.bf16 and self.fp16:
            raise ValueError(
                "bf16 and fp16 cannot both be True — choose one precision mode"
            )
        return self


# ── SageMaker ───────────────────────────────────────────────────


class VPCConfig(BaseModel):
    """VPC networking for SageMaker training jobs."""

    security_group_ids: list[str] = Field(min_length=1)
    subnets: list[str] = Field(min_length=1)


class SageMakerConfig(BaseModel):
    """SageMaker training-job resource configuration."""

    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    volume_size_gb: int = 200
    max_run_seconds: int = 86400  # 24 hours
    max_wait_seconds: int = 86400  # for spot
    use_spot_instances: bool = False
    spot_max_wait_ratio: float = 2.0
    checkpoint_s3_uri: str | None = None
    role_arn: str
    vpc_config: VPCConfig | None = None
    environment: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _spot_needs_checkpoint(self) -> SageMakerConfig:
        if self.use_spot_instances and not self.checkpoint_s3_uri:
            raise ValueError(
                "checkpoint_s3_uri is required when use_spot_instances is True"
            )
        return self


# ── top-level ───────────────────────────────────────────────────


class TrainingJobConfig(BaseModel):
    """Complete training-job configuration tying all sub-configs together."""

    experiment_name: str
    run_name: str | None = None  # auto-generated if None
    model: ModelConfig
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    training: TrainingHyperparameters = Field(
        default_factory=TrainingHyperparameters,
    )
    sagemaker: SageMakerConfig | None = None
    dataset_id: str | None = None  # from DatasetRegistry (Prompts 17-18)
    dataset_path: str | None = None
    """Direct path to Arrow dataset directory (local or S3).

    Use this when DatasetRegistry is not yet available.
    """
    output_s3_uri: str = ""
    output_local_path: str | None = None
    resume_from_checkpoint: str | None = None

    @model_validator(mode="after")
    def _require_dataset_source(self) -> TrainingJobConfig:
        """At least one of dataset_id or dataset_path must be set.

        When both are provided, dataset_id takes priority.
        """
        if self.dataset_id is None and self.dataset_path is None:
            raise ValueError(
                "At least one of 'dataset_id' or 'dataset_path' must be set. "
                "Use 'dataset_path' for direct Arrow paths, or 'dataset_id' "
                "after running Prompts 17-18 to create the DatasetRegistry."
            )
        return self

    @model_validator(mode="after")
    def _dora_method_sync(self) -> TrainingJobConfig:
        """Ensure LoRA use_dora=True when quantization method is 'dora'."""
        if self.quantization.method == "dora" and not self.lora.use_dora:
            raise ValueError(
                "lora.use_dora must be True when quantization.method is 'dora'"
            )
        return self

    @model_validator(mode="after")
    def _generate_run_name(self) -> TrainingJobConfig:
        if self.run_name is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            self.run_name = f"{self.experiment_name}-{ts}-{short_id}"
        return self
