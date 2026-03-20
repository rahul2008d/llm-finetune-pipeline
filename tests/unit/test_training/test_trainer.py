"""Unit tests for training.trainer – FineTuneTrainer and TrainingResult."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest
from pydantic import ValidationError

from src.config.training import (
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    SageMakerConfig,
    TrainingHyperparameters,
    TrainingJobConfig,
)
from src.training.callbacks import (
    CheckpointCleanupCallback,
    CostTrackingCallback,
    EarlyStoppingWithPatience,
    GradientNormCallback,
    LoggingCallback,
    LossSpikingCallback,
    MemoryMonitorCallback,
)
from src.training.trainer import FineTuneTrainer, TrainingResult


# ── helpers ─────────────────────────────────────────────────────


def _make_config(**overrides: Any) -> TrainingJobConfig:
    """Build a minimal valid TrainingJobConfig with optional overrides."""
    defaults: dict[str, Any] = {
        "experiment_name": "test-experiment",
        "model": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
        "sagemaker": {
            "role_arn": "arn:aws:iam::123456789012:role/SageMakerRole",
            "instance_type": "ml.g5.2xlarge",
        },
        "dataset_path": "/data/datasets/my-dataset",
        "output_s3_uri": "s3://my-bucket/output",
    }
    defaults.update(overrides)
    return TrainingJobConfig(**defaults)


# ═══════════════════════════════════════════════════════════════
# TrainingResult
# ═══════════════════════════════════════════════════════════════


class TestTrainingResult:
    def test_valid_construction(self) -> None:
        result = TrainingResult(
            run_id="abc123",
            experiment_name="exp-1",
            final_train_loss=0.45,
            final_eval_loss=0.52,
            best_eval_loss=0.50,
            total_steps=1000,
            training_time_seconds=3600.0,
            estimated_cost_usd=1.21,
            adapter_s3_uri="s3://bucket/adapter",
            metrics={"train_loss": 0.45, "eval_loss": 0.52},
        )
        assert result.run_id == "abc123"
        assert result.total_steps == 1000
        assert result.metrics["train_loss"] == pytest.approx(0.45)

    def test_default_empty_metrics(self) -> None:
        result = TrainingResult(
            run_id="x",
            experiment_name="e",
            final_train_loss=0.0,
            final_eval_loss=0.0,
            best_eval_loss=0.0,
            total_steps=0,
            training_time_seconds=0.0,
            estimated_cost_usd=0.0,
            adapter_s3_uri="s3://b/a",
        )
        assert result.metrics == {}

    def test_serialization_roundtrip(self) -> None:
        result = TrainingResult(
            run_id="r1",
            experiment_name="exp",
            final_train_loss=1.0,
            final_eval_loss=1.1,
            best_eval_loss=0.9,
            total_steps=500,
            training_time_seconds=1800.0,
            estimated_cost_usd=0.6,
            adapter_s3_uri="s3://b/a",
            metrics={"lr": 0.0002},
        )
        data = result.model_dump()
        restored = TrainingResult(**data)
        assert restored == result

    def test_immutability_via_model_dump(self) -> None:
        result = TrainingResult(
            run_id="r1",
            experiment_name="exp",
            final_train_loss=1.0,
            final_eval_loss=1.1,
            best_eval_loss=0.9,
            total_steps=500,
            training_time_seconds=1800.0,
            estimated_cost_usd=0.6,
            adapter_s3_uri="s3://b/a",
        )
        d = result.model_dump()
        assert isinstance(d, dict)
        assert d["run_id"] == "r1"


# ═══════════════════════════════════════════════════════════════
# FineTuneTrainer.__init__
# ═══════════════════════════════════════════════════════════════


class TestFineTuneTrainerInit:
    def test_config_stored(self) -> None:
        config = _make_config()
        trainer = FineTuneTrainer(config)
        assert trainer.config is config

    def test_model_loader_created(self) -> None:
        from src.training.model_loader import ModelLoader

        trainer = FineTuneTrainer(_make_config())
        assert isinstance(trainer.model_loader, ModelLoader)

    def test_callbacks_built(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        assert len(trainer.callbacks) == 7


# ═══════════════════════════════════════════════════════════════
# _build_callbacks
# ═══════════════════════════════════════════════════════════════


class TestBuildCallbacks:
    def test_callback_types(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        types = [type(cb) for cb in trainer.callbacks]
        assert LoggingCallback in types
        assert CostTrackingCallback in types
        assert MemoryMonitorCallback in types
        assert LossSpikingCallback in types
        assert CheckpointCleanupCallback in types
        assert GradientNormCallback in types
        assert EarlyStoppingWithPatience in types

    def test_cost_callback_uses_config_instance(self) -> None:
        config = _make_config()
        trainer = FineTuneTrainer(config)
        cost_cbs = [cb for cb in trainer.callbacks if isinstance(cb, CostTrackingCallback)]
        assert len(cost_cbs) == 1
        assert cost_cbs[0].instance_count == 1

    def test_checkpoint_cleanup_uses_checkpoint_s3(self) -> None:
        config = _make_config(
            sagemaker={
                "role_arn": "arn:aws:iam::123456789012:role/R",
                "checkpoint_s3_uri": "s3://bucket/ckpt",
            },
        )
        trainer = FineTuneTrainer(config)
        ckpt_cbs = [
            cb for cb in trainer.callbacks
            if isinstance(cb, CheckpointCleanupCallback)
        ]
        assert len(ckpt_cbs) == 1
        assert ckpt_cbs[0]._checkpoint_s3_uri == "s3://bucket/ckpt"

    def test_checkpoint_cleanup_falls_back_to_output_s3(self) -> None:
        config = _make_config()
        trainer = FineTuneTrainer(config)
        ckpt_cbs = [
            cb for cb in trainer.callbacks
            if isinstance(cb, CheckpointCleanupCallback)
        ]
        assert ckpt_cbs[0]._checkpoint_s3_uri == "s3://my-bucket/output"


# ═══════════════════════════════════════════════════════════════
# _set_seeds
# ═══════════════════════════════════════════════════════════════


class TestSetSeeds:
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_sets_all_seeds(
        self, mock_set_seed: MagicMock, mock_torch: MagicMock,
    ) -> None:
        FineTuneTrainer._set_seeds(42)

        mock_set_seed.assert_called_once_with(42)
        mock_torch.manual_seed.assert_called_once_with(42)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(42)

    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_sets_cudnn_flags(
        self, mock_set_seed: MagicMock, mock_torch: MagicMock,
    ) -> None:
        FineTuneTrainer._set_seeds(123)

        assert mock_torch.backends.cudnn.deterministic is True
        assert mock_torch.backends.cudnn.benchmark is False


# ═══════════════════════════════════════════════════════════════
# _build_training_args
# ═══════════════════════════════════════════════════════════════


class TestBuildTrainingArgs:
    def test_maps_hyperparameters(self) -> None:
        config = _make_config()
        trainer = FineTuneTrainer(config)
        args = trainer._build_training_args("/tmp/out")

        hp = config.training
        assert args.num_train_epochs == hp.num_train_epochs
        assert args.per_device_train_batch_size == hp.per_device_train_batch_size
        assert args.learning_rate == pytest.approx(hp.learning_rate)
        assert args.warmup_ratio == pytest.approx(hp.warmup_ratio)
        assert args.weight_decay == pytest.approx(hp.weight_decay)
        assert args.seed == hp.seed

    def test_ddp_find_unused_false(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert args.ddp_find_unused_parameters is False

    def test_remove_unused_columns_false(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert args.remove_unused_columns is False

    def test_output_dir_set(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/my/output")
        assert args.output_dir == "/my/output"

    def test_run_name_from_config(self) -> None:
        config = _make_config(run_name="custom-run")
        trainer = FineTuneTrainer(config)
        args = trainer._build_training_args("/tmp/out")
        assert args.run_name == "custom-run"

    def test_precision_bf16(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert args.bf16 is True
        assert args.fp16 is False

    def test_gradient_checkpointing(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert args.gradient_checkpointing is True

    def test_save_strategy(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert args.save_strategy.value == "steps"

    def test_report_to_mlflow(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        args = trainer._build_training_args("/tmp/out")
        assert "mlflow" in args.report_to


# ═══════════════════════════════════════════════════════════════
# _flat_config_params
# ═══════════════════════════════════════════════════════════════


class TestFlatConfigParams:
    def test_contains_key_fields(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        params = trainer._flat_config_params()

        assert params["model_name"] == "meta-llama/Llama-3.1-8B-Instruct"
        assert params["quant_method"] == "qlora"
        assert params["lora_r"] == 64
        assert params["lora_alpha"] == 128
        assert params["learning_rate"] == pytest.approx(2e-4)
        assert params["instance_type"] == "ml.g5.2xlarge"

    def test_all_params_are_loggable_types(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        params = trainer._flat_config_params()
        for k, v in params.items():
            assert isinstance(v, (str, int, float, bool)), f"{k} has type {type(v)}"


# ═══════════════════════════════════════════════════════════════
# _load_datasets
# ═══════════════════════════════════════════════════════════════


class TestLoadDatasets:
    @patch("src.training.trainer.os.path.isdir", return_value=True)
    @patch("src.training.trainer.load_from_disk")
    def test_direct_path_loads_train_and_validation(
        self, mock_load: MagicMock, mock_isdir: MagicMock,
    ) -> None:
        mock_train = MagicMock(name="train_ds")
        mock_val = MagicMock(name="val_ds")
        mock_load.side_effect = lambda p: mock_train if p.endswith("/train") else mock_val

        trainer = FineTuneTrainer(_make_config(dataset_path="/data/my-dataset"))
        train, val = trainer._load_datasets()

        assert mock_load.call_count == 2
        mock_load.assert_any_call("/data/my-dataset/train")
        mock_load.assert_any_call("/data/my-dataset/validation")
        assert train is mock_train
        assert val is mock_val

    @patch("src.training.trainer.load_from_disk")
    def test_dataset_id_uses_registry(self, mock_load: MagicMock) -> None:
        mock_load.return_value = MagicMock()
        mock_registry = MagicMock()
        mock_info = MagicMock()
        mock_info.train_path = "/registry/train"
        mock_info.validation_path = "/registry/val"
        mock_registry.return_value.get.return_value = mock_info

        with patch("src.data.registry.DatasetRegistry", mock_registry):
            trainer = FineTuneTrainer(
                _make_config(dataset_id="my-ds", dataset_path=None),
            )
            train, val = trainer._load_datasets()

        mock_registry.return_value.get.assert_called_once_with("my-ds")
        mock_load.assert_any_call("/registry/train")
        mock_load.assert_any_call("/registry/val")

    @patch("src.training.trainer.load_from_disk")
    def test_direct_path_constructs_correct_paths(
        self, mock_load: MagicMock,
    ) -> None:
        mock_load.return_value = MagicMock()
        trainer = FineTuneTrainer(
            _make_config(dataset_path="s3://bucket/datasets/v1"),
        )
        trainer._load_datasets()

        calls = [c.args[0] for c in mock_load.call_args_list]
        assert "s3://bucket/datasets/v1/train" in calls
        assert "s3://bucket/datasets/v1/validation" in calls


# ═══════════════════════════════════════════════════════════════
# _get_cost_estimate
# ═══════════════════════════════════════════════════════════════


class TestGetCostEstimate:
    def test_returns_cost_from_callback(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        cost_cbs = [
            cb for cb in trainer.callbacks
            if isinstance(cb, CostTrackingCallback)
        ]
        cost_cbs[0]._estimated_cost = 12.34
        assert trainer._get_cost_estimate() == pytest.approx(12.34)

    def test_returns_zero_when_no_cost_callback(self) -> None:
        trainer = FineTuneTrainer(_make_config())
        trainer.callbacks = [LoggingCallback()]
        assert trainer._get_cost_estimate() == 0.0


# ═══════════════════════════════════════════════════════════════
# _upload_adapter
# ═══════════════════════════════════════════════════════════════


class TestUploadAdapter:
    @patch("src.training.trainer.boto3.client")
    def test_uploads_files_to_s3(self, mock_boto: MagicMock, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_text("weights")
        (adapter_dir / "adapter_config.json").write_text("{}")

        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        config = _make_config(run_name="run-001")
        trainer = FineTuneTrainer(config)
        uri = trainer._upload_adapter(adapter_dir)

        assert mock_s3.upload_file.call_count == 2
        assert "run-001/adapter" in uri

    @patch("src.training.trainer.boto3.client")
    def test_returns_s3_uri(self, mock_boto: MagicMock, tmp_path: Path) -> None:
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        (adapter_dir / "file.bin").write_text("data")

        mock_boto.return_value = MagicMock()

        config = _make_config(
            run_name="myrun",
            output_s3_uri="s3://my-bucket/models",
        )
        trainer = FineTuneTrainer(config)
        uri = trainer._upload_adapter(adapter_dir)

        assert uri == "s3://my-bucket/models/myrun/adapter"


# ═══════════════════════════════════════════════════════════════
# train() – full pipeline with mocks
# ═══════════════════════════════════════════════════════════════


@patch("src.training.trainer.os.path.isdir", return_value=True)
class TestTrainPipeline:
    def _setup_mocks(self) -> dict[str, MagicMock]:
        """Create a coherent set of mocks for the full train pipeline."""
        mocks: dict[str, MagicMock] = {}

        # Model loader
        mocks["model_loader"] = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "</s>"
        mocks["model_loader"].load_tokenizer.return_value = mock_tokenizer
        mocks["model_loader"].load_base_model.return_value = MagicMock(name="base")
        mocks["model_loader"].apply_lora.return_value = MagicMock(name="peft")

        # Datasets
        mocks["train_ds"] = MagicMock(name="train_ds")
        mocks["eval_ds"] = MagicMock(name="eval_ds")

        # Train output
        mocks["train_output"] = MagicMock()
        mocks["train_output"].metrics = {
            "train_loss": 0.35,
            "eval_loss": 0.42,
            "train_runtime": 1200.0,
        }

        # Trainer state
        mocks["trainer_state"] = MagicMock()
        mocks["trainer_state"].best_metric = 0.40
        mocks["trainer_state"].global_step = 500

        # MLflow run
        mocks["mlflow_run"] = MagicMock()
        mocks["mlflow_run"].info.run_id = "mlflow-run-123"

        return mocks

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_full_train_returns_result(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()

        # Dataset
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]

        # SFTTrainer
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer

        # MLflow
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config(run_name="test-run-1")
        ft = FineTuneTrainer(config)
        ft.model_loader = m["model_loader"]
        result = ft.train()

        assert isinstance(result, TrainingResult)
        assert result.run_id == "mlflow-run-123"
        assert result.experiment_name == "test-experiment"
        assert result.final_train_loss == pytest.approx(0.35)
        assert result.final_eval_loss == pytest.approx(0.42)
        assert result.best_eval_loss == pytest.approx(0.40)
        assert result.total_steps == 500
        assert result.adapter_s3_uri.startswith("s3://")

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_seeds_set_before_training(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        ft.train()

        mock_set_seed.assert_called_once_with(42)
        mock_torch.manual_seed.assert_called_once_with(42)

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_resume_from_checkpoint(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config(
            run_name="r",
            resume_from_checkpoint="/ckpt/checkpoint-200",
        )
        ft = FineTuneTrainer(config)
        ft.model_loader = m["model_loader"]
        ft.train()

        mock_trainer.train.assert_called_once_with(
            resume_from_checkpoint="/ckpt/checkpoint-200",
        )

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_no_resume_calls_train_without_arg(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        ft.train()

        mock_trainer.train.assert_called_once_with()

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_sft_trainer_constructed_correctly(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        config = _make_config(run_name="r")
        ft = FineTuneTrainer(config)
        ft.model_loader = m["model_loader"]
        ft.train()

        _, kwargs = mock_sft_cls.call_args
        assert kwargs["packing"] is False
        assert kwargs["dataset_text_field"] is None
        assert kwargs["max_seq_length"] == 4096
        assert kwargs["train_dataset"] is m["train_ds"]
        assert kwargs["eval_dataset"] is m["eval_ds"]
        assert len(kwargs["callbacks"]) == 7

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_adapter_saved_after_training(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        ft.train()

        mock_trainer.model.save_pretrained.assert_called_once()
        m["model_loader"].load_tokenizer.return_value.save_pretrained.assert_called_once()

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_mlflow_experiment_set(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        ft.train()

        mock_mlflow.set_experiment.assert_called_once_with("test-experiment")

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_mlflow_tracking_uri_from_env(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict("os.environ", {"MLFLOW_TRACKING_URI": "http://mlflow:5000"}):
            ft = FineTuneTrainer(_make_config(run_name="r"))
            ft.model_loader = m["model_loader"]
            ft.train()

        mock_mlflow.set_tracking_uri.assert_called_once_with("http://mlflow:5000")

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_mlflow_metrics_logged(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        ft.train()

        assert mock_mlflow.log_metrics.call_count >= 2
        assert mock_mlflow.log_params.call_count == 1

    @patch("src.training.trainer.boto3.client")
    @patch("src.training.trainer.SFTTrainer")
    @patch("src.training.trainer.load_from_disk")
    @patch("src.training.trainer.mlflow")
    @patch("src.training.trainer.torch")
    @patch("src.training.trainer.set_seed")
    def test_best_metric_none_falls_back(
        self,
        mock_set_seed: MagicMock,
        mock_torch: MagicMock,
        mock_mlflow: MagicMock,
        mock_load_disk: MagicMock,
        mock_sft_cls: MagicMock,
        mock_boto: MagicMock,
        mock_isdir: MagicMock,
    ) -> None:
        m = self._setup_mocks()
        m["trainer_state"].best_metric = None
        mock_boto.return_value = MagicMock()
        mock_load_disk.side_effect = [m["train_ds"], m["eval_ds"]]
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = m["train_output"]
        mock_trainer.state = m["trainer_state"]
        mock_sft_cls.return_value = mock_trainer
        mock_mlflow.start_run.return_value.__enter__ = lambda s: m["mlflow_run"]
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        ft = FineTuneTrainer(_make_config(run_name="r"))
        ft.model_loader = m["model_loader"]
        result = ft.train()

        assert result.best_eval_loss == pytest.approx(0.42)


# ═══════════════════════════════════════════════════════════════
# Module exports
# ═══════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_trainer_in_all(self) -> None:
        from src.training import trainer

        assert "FineTuneTrainer" in trainer.__all__
        assert "TrainingResult" in trainer.__all__

    def test_importable_from_package(self) -> None:
        from src.training import FineTuneTrainer as FT
        from src.training import TrainingResult as TR

        assert FT is FineTuneTrainer
        assert TR is TrainingResult
