"""Unit tests for config.training – Pydantic v2 training configuration models."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.config.training import (
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    SageMakerConfig,
    TrainingHyperparameters,
    TrainingJobConfig,
    VPCConfig,
)


# ── helpers ─────────────────────────────────────────────────────

_SM_KWARGS = {
    "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
}


def _job(**overrides) -> TrainingJobConfig:
    """Build a minimal valid TrainingJobConfig, merging *overrides*."""
    defaults = {
        "experiment_name": "test-exp",
        "model": {"model_name_or_path": "meta-llama/Llama-2-7b-hf"},
        "sagemaker": _SM_KWARGS,
        "dataset_path": "./data/prepared/test",
        "output_s3_uri": "s3://bucket/output",
    }
    defaults.update(overrides)
    return TrainingJobConfig(**defaults)


# ── ModelConfig ─────────────────────────────────────────────────


class TestModelConfig:
    def test_defaults(self) -> None:
        m = ModelConfig(model_name_or_path="meta-llama/Llama-2-7b-hf")
        assert m.trust_remote_code is False
        assert m.torch_dtype == "bfloat16"
        assert m.attn_implementation == "flash_attention_2"
        assert m.max_seq_length == 4096
        assert m.use_cache is False

    def test_s3_path(self) -> None:
        m = ModelConfig(model_name_or_path="s3://bucket/models/llama2")
        assert m.model_name_or_path.startswith("s3://")

    def test_max_seq_length_lower_bound(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model_name_or_path="x", max_seq_length=64)

    def test_max_seq_length_upper_bound(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model_name_or_path="x", max_seq_length=65536)

    def test_invalid_dtype(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model_name_or_path="x", torch_dtype="int8")

    def test_invalid_attn(self) -> None:
        with pytest.raises(ValidationError):
            ModelConfig(model_name_or_path="x", attn_implementation="xformers")


# ── QuantizationConfig ──────────────────────────────────────────


class TestQuantizationConfig:
    def test_defaults(self) -> None:
        q = QuantizationConfig()
        assert q.method == "qlora"
        assert q.load_in_4bit is True
        assert q.bnb_4bit_quant_type == "nf4"
        assert q.bnb_4bit_compute_dtype == "bfloat16"
        assert q.bnb_4bit_use_double_quant is True

    def test_dora_method(self) -> None:
        q = QuantizationConfig(method="dora")
        assert q.method == "dora"

    def test_fp4_quant_type(self) -> None:
        q = QuantizationConfig(bnb_4bit_quant_type="fp4")
        assert q.bnb_4bit_quant_type == "fp4"

    def test_invalid_method(self) -> None:
        with pytest.raises(ValidationError):
            QuantizationConfig(method="gptq")


# ── LoRAConfig ──────────────────────────────────────────────────


class TestLoRAConfig:
    def test_defaults(self) -> None:
        l = LoRAConfig()
        assert l.r == 64
        assert l.lora_alpha == 128
        assert l.lora_dropout == 0.05
        assert l.bias == "none"
        assert l.task_type == "CAUSAL_LM"
        assert l.use_dora is False
        assert l.use_rslora is False
        assert l.modules_to_save is None

    def test_default_target_modules(self) -> None:
        l = LoRAConfig()
        expected = ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"]
        assert l.target_modules == expected

    def test_rank_too_low(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(r=2)

    def test_rank_too_high(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(r=512)

    def test_alpha_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(lora_alpha=0)

    def test_dropout_bounds(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(lora_dropout=0.8)

    def test_dora_enabled(self) -> None:
        l = LoRAConfig(use_dora=True)
        assert l.use_dora is True

    def test_rslora_enabled(self) -> None:
        l = LoRAConfig(use_rslora=True)
        assert l.use_rslora is True

    def test_modules_to_save(self) -> None:
        l = LoRAConfig(modules_to_save=["embed_tokens", "lm_head"])
        assert l.modules_to_save == ["embed_tokens", "lm_head"]

    def test_bias_lora_only(self) -> None:
        l = LoRAConfig(bias="lora_only")
        assert l.bias == "lora_only"

    def test_invalid_bias(self) -> None:
        with pytest.raises(ValidationError):
            LoRAConfig(bias="some")


# ── TrainingHyperparameters ─────────────────────────────────────


class TestTrainingHyperparameters:
    def test_defaults(self) -> None:
        t = TrainingHyperparameters()
        assert t.num_train_epochs == 3
        assert t.per_device_train_batch_size == 4
        assert t.per_device_eval_batch_size == 8
        assert t.gradient_accumulation_steps == 4
        assert t.learning_rate == 2e-4
        assert t.weight_decay == 0.01
        assert t.warmup_ratio == 0.03
        assert t.lr_scheduler_type == "cosine"
        assert t.max_grad_norm == 1.0
        assert t.optim == "paged_adamw_8bit"
        assert t.bf16 is True
        assert t.fp16 is False
        assert t.gradient_checkpointing is True
        assert t.gradient_checkpointing_kwargs == {"use_reentrant": False}
        assert t.logging_steps == 10
        assert t.eval_strategy == "steps"
        assert t.eval_steps == 100
        assert t.save_strategy == "steps"
        assert t.save_steps == 200
        assert t.save_total_limit == 3
        assert t.load_best_model_at_end is True
        assert t.metric_for_best_model == "eval_loss"
        assert t.greater_is_better is False
        assert t.seed == 42
        assert t.data_seed == 42
        assert t.dataloader_num_workers == 4
        assert t.dataloader_pin_memory is True
        assert t.group_by_length is True
        assert t.report_to == ["mlflow"]

    def test_bf16_fp16_both_true_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="bf16 and fp16 cannot both be True"
        ):
            TrainingHyperparameters(bf16=True, fp16=True)

    def test_bf16_false_fp16_true(self) -> None:
        t = TrainingHyperparameters(bf16=False, fp16=True)
        assert t.fp16 is True
        assert t.bf16 is False

    def test_both_false(self) -> None:
        t = TrainingHyperparameters(bf16=False, fp16=False)
        assert t.bf16 is False
        assert t.fp16 is False

    def test_epoch_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TrainingHyperparameters(num_train_epochs=0)
        with pytest.raises(ValidationError):
            TrainingHyperparameters(num_train_epochs=21)

    def test_lr_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TrainingHyperparameters(learning_rate=0.0)
        with pytest.raises(ValidationError):
            TrainingHyperparameters(learning_rate=0.1)

    def test_batch_size_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TrainingHyperparameters(per_device_train_batch_size=0)
        with pytest.raises(ValidationError):
            TrainingHyperparameters(per_device_train_batch_size=128)

    def test_scheduler_types(self) -> None:
        for sched in ("cosine", "linear", "constant", "cosine_with_restarts"):
            t = TrainingHyperparameters(lr_scheduler_type=sched)
            assert t.lr_scheduler_type == sched

    def test_invalid_scheduler(self) -> None:
        with pytest.raises(ValidationError):
            TrainingHyperparameters(lr_scheduler_type="polynomial")

    def test_warmup_ratio_bound(self) -> None:
        with pytest.raises(ValidationError):
            TrainingHyperparameters(warmup_ratio=0.5)

    def test_report_to_multiple(self) -> None:
        t = TrainingHyperparameters(report_to=["mlflow", "tensorboard"])
        assert t.report_to == ["mlflow", "tensorboard"]


# ── VPCConfig ───────────────────────────────────────────────────


class TestVPCConfig:
    def test_valid(self) -> None:
        v = VPCConfig(security_group_ids=["sg-123"], subnets=["subnet-1"])
        assert v.security_group_ids == ["sg-123"]

    def test_empty_sg_raises(self) -> None:
        with pytest.raises(ValidationError):
            VPCConfig(security_group_ids=[], subnets=["subnet-1"])

    def test_empty_subnets_raises(self) -> None:
        with pytest.raises(ValidationError):
            VPCConfig(security_group_ids=["sg-1"], subnets=[])


# ── SageMakerConfig ────────────────────────────────────────────


class TestSageMakerConfig:
    def test_defaults(self) -> None:
        s = SageMakerConfig(**_SM_KWARGS)
        assert s.instance_type == "ml.g5.2xlarge"
        assert s.instance_count == 1
        assert s.volume_size_gb == 200
        assert s.max_run_seconds == 86400
        assert s.use_spot_instances is False
        assert s.checkpoint_s3_uri is None
        assert s.vpc_config is None
        assert s.environment == {}

    def test_spot_without_checkpoint_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="checkpoint_s3_uri is required"
        ):
            SageMakerConfig(use_spot_instances=True, **_SM_KWARGS)

    def test_spot_with_checkpoint_ok(self) -> None:
        s = SageMakerConfig(
            use_spot_instances=True,
            checkpoint_s3_uri="s3://bucket/checkpoints",
            **_SM_KWARGS,
        )
        assert s.use_spot_instances is True
        assert s.checkpoint_s3_uri == "s3://bucket/checkpoints"

    def test_vpc_config(self) -> None:
        s = SageMakerConfig(
            vpc_config={
                "security_group_ids": ["sg-1"],
                "subnets": ["subnet-1"],
            },
            **_SM_KWARGS,
        )
        assert s.vpc_config is not None
        assert s.vpc_config.subnets == ["subnet-1"]

    def test_environment_dict(self) -> None:
        s = SageMakerConfig(
            environment={"HF_TOKEN": "tok123"}, **_SM_KWARGS
        )
        assert s.environment["HF_TOKEN"] == "tok123"


# ── TrainingJobConfig ───────────────────────────────────────────


class TestTrainingJobConfig:
    def test_minimal_valid(self) -> None:
        cfg = _job()
        assert cfg.experiment_name == "test-exp"
        assert cfg.run_name is not None  # auto-generated

    def test_run_name_auto_generated(self) -> None:
        cfg = _job()
        assert cfg.run_name.startswith("test-exp-")

    def test_explicit_run_name(self) -> None:
        cfg = _job(run_name="my-run")
        assert cfg.run_name == "my-run"

    def test_dora_method_requires_use_dora(self) -> None:
        with pytest.raises(
            ValidationError, match="lora.use_dora must be True"
        ):
            _job(
                quantization={"method": "dora"},
                lora={"use_dora": False},
            )

    def test_dora_method_with_use_dora_ok(self) -> None:
        cfg = _job(
            quantization={"method": "dora"},
            lora={"use_dora": True},
        )
        assert cfg.quantization.method == "dora"
        assert cfg.lora.use_dora is True

    def test_qlora_with_dora_false_ok(self) -> None:
        cfg = _job(
            quantization={"method": "qlora"},
            lora={"use_dora": False},
        )
        assert cfg.quantization.method == "qlora"

    def test_resume_from_checkpoint(self) -> None:
        cfg = _job(resume_from_checkpoint="s3://bucket/ckpt/step-500")
        assert cfg.resume_from_checkpoint == "s3://bucket/ckpt/step-500"

    def test_defaults_propagate(self) -> None:
        cfg = _job()
        assert cfg.model.torch_dtype == "bfloat16"
        assert cfg.quantization.bnb_4bit_quant_type == "nf4"
        assert cfg.lora.r == 64
        assert cfg.training.learning_rate == 2e-4

    def test_bf16_fp16_cross_validation(self) -> None:
        with pytest.raises(ValidationError, match="bf16 and fp16"):
            _job(training={"bf16": True, "fp16": True})

    def test_spot_cross_validation(self) -> None:
        with pytest.raises(ValidationError, match="checkpoint_s3_uri"):
            _job(sagemaker={**_SM_KWARGS, "use_spot_instances": True})

    def test_serialization_roundtrip(self) -> None:
        cfg = _job()
        dumped = cfg.model_dump(mode="json")
        restored = TrainingJobConfig(**dumped)
        assert restored.experiment_name == cfg.experiment_name
        assert restored.model.model_name_or_path == cfg.model.model_name_or_path
        assert restored.sagemaker.role_arn == cfg.sagemaker.role_arn

    def test_json_roundtrip(self) -> None:
        cfg = _job()
        j = cfg.model_dump_json()
        restored = TrainingJobConfig.model_validate_json(j)
        assert restored.dataset_id == cfg.dataset_id

    def test_override_nested_fields(self) -> None:
        cfg = _job(
            model={"model_name_or_path": "x", "max_seq_length": 2048},
            training={"num_train_epochs": 5, "learning_rate": 1e-4},
            lora={"r": 32, "lora_alpha": 64},
        )
        assert cfg.model.max_seq_length == 2048
        assert cfg.training.num_train_epochs == 5
        assert cfg.training.learning_rate == 1e-4
        assert cfg.lora.r == 32
        assert cfg.lora.lora_alpha == 64

    def test_both_dataset_sources_none_raises(self) -> None:
        with pytest.raises(
            ValidationError, match="dataset_id.*dataset_path",
        ):
            _job(dataset_id=None, dataset_path=None)

    def test_dataset_path_only_ok(self) -> None:
        cfg = _job(dataset_path="./data/prepared/test")
        assert cfg.dataset_path == "./data/prepared/test"
        assert cfg.dataset_id is None

    def test_dataset_id_only_ok(self) -> None:
        cfg = _job(dataset_id="my-ds", dataset_path=None)
        assert cfg.dataset_id == "my-ds"
        assert cfg.dataset_path is None

    def test_both_dataset_sources_set_ok(self) -> None:
        cfg = _job(dataset_id="my-ds", dataset_path="./data")
        assert cfg.dataset_id == "my-ds"
        assert cfg.dataset_path == "./data"
