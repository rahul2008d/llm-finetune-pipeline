"""Unit tests for training.callbacks – all callback classes."""
from __future__ import annotations

import math
import shutil
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.training.callbacks import (
    CheckpointCleanupCallback,
    CostTrackingCallback,
    EarlyStoppingWithPatience,
    GradientNormCallback,
    LoggingCallback,
    LossSpikingCallback,
    MemoryMonitorCallback,
    _INSTANCE_COST_PER_HOUR,
)


# ── helpers ─────────────────────────────────────────────────────


def _make_state(global_step: int = 100, epoch: float = 1.0) -> TrainerState:
    state = TrainerState()
    state.global_step = global_step
    state.epoch = epoch
    return state


def _make_args(output_dir: str = "/tmp/output") -> MagicMock:
    args = MagicMock(spec=TrainingArguments)
    args.output_dir = output_dir
    return args


def _make_control() -> TrainerControl:
    return TrainerControl()


# ═══════════════════════════════════════════════════════════════
# LoggingCallback
# ═══════════════════════════════════════════════════════════════


class TestLoggingCallback:
    def test_on_log_with_metrics(self) -> None:
        cb = LoggingCallback()
        control = _make_control()
        state = _make_state(global_step=10, epoch=0.5)
        cb.on_log(_make_args(), state, control, logs={"loss": 2.5, "lr": 1e-4})

    def test_on_log_without_logs(self) -> None:
        cb = LoggingCallback()
        cb.on_log(_make_args(), _make_state(), _make_control(), logs=None)

    def test_on_log_empty_logs(self) -> None:
        cb = LoggingCallback()
        cb.on_log(_make_args(), _make_state(), _make_control(), logs={})

    def test_on_epoch_end(self) -> None:
        cb = LoggingCallback()
        cb.on_epoch_end(_make_args(), _make_state(epoch=2.0), _make_control())


# ═══════════════════════════════════════════════════════════════
# CostTrackingCallback
# ═══════════════════════════════════════════════════════════════


class TestCostTrackingCallback:
    def test_known_instance_type(self) -> None:
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        assert cb.cost_per_hour == pytest.approx(1.212)

    def test_unknown_instance_type_zero_cost(self) -> None:
        cb = CostTrackingCallback(instance_type="ml.mystery.xlarge")
        assert cb.cost_per_hour == 0.0

    def test_instance_count(self) -> None:
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge", instance_count=4)
        assert cb.instance_count == 4

    def test_on_train_begin_sets_start_time(self) -> None:
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        assert cb._start_time is None
        cb.on_train_begin(_make_args(), _make_state(), _make_control())
        assert cb._start_time is not None

    @patch("src.training.callbacks.time.monotonic")
    def test_cost_computed_correctly(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge", instance_count=2)
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Advance 1 hour
        mock_time.return_value = 3600.0
        cost = cb._compute_cost()
        # 1 hour * 1.212/hr * 2 instances = 2.424
        assert cost == pytest.approx(2.424, abs=0.001)

    @patch("src.training.callbacks.time.monotonic")
    def test_on_log_updates_cost(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        mock_time.return_value = 1800.0  # 30 min
        cb.on_log(_make_args(), _make_state(), _make_control())
        assert cb._estimated_cost > 0.0

    @patch("src.training.callbacks.time.monotonic")
    def test_budget_exceeded_stops_training(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(
            instance_type="ml.g5.2xlarge",
            max_budget_usd=1.0,
        )
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        # Advance enough that cost exceeds $1.0
        # 1.212/hr -> need ~0.825 hrs = ~2970 sec
        mock_time.return_value = 3600.0
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control)
        assert control.should_training_stop is True

    @patch("src.training.callbacks.time.monotonic")
    def test_under_budget_does_not_stop(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(
            instance_type="ml.g5.2xlarge",
            max_budget_usd=100.0,
        )
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        mock_time.return_value = 60.0  # 1 minute
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control)
        assert control.should_training_stop is False

    @patch("src.training.callbacks.time.monotonic")
    def test_no_budget_limit_never_stops(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        mock_time.return_value = 360000.0  # 100 hours
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control)
        assert control.should_training_stop is False

    @patch("src.training.callbacks.time.monotonic")
    def test_on_train_end_logs_final_cost(self, mock_time: MagicMock) -> None:
        mock_time.return_value = 0.0
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        cb.on_train_begin(_make_args(), _make_state(), _make_control())

        mock_time.return_value = 7200.0  # 2 hours
        cb.on_train_end(_make_args(), _make_state(), _make_control())
        assert cb._estimated_cost == pytest.approx(2.424, abs=0.001)

    def test_elapsed_hours_before_start(self) -> None:
        cb = CostTrackingCallback(instance_type="ml.g5.2xlarge")
        assert cb._elapsed_hours() == 0.0

    def test_instance_cost_map_has_entries(self) -> None:
        assert len(_INSTANCE_COST_PER_HOUR) > 0
        assert all(v > 0 for v in _INSTANCE_COST_PER_HOUR.values())


# ═══════════════════════════════════════════════════════════════
# MemoryMonitorCallback
# ═══════════════════════════════════════════════════════════════


class TestMemoryMonitorCallback:
    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_on_log_logs_cpu_memory(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024 * 1024 * 512)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback()
        cb.on_log(_make_args(), _make_state(), _make_control())

    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_on_log_logs_gpu_memory(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 4 * 1024**3  # 4 GB
        mock_torch.cuda.memory_reserved.return_value = 6 * 1024**3
        mock_torch.cuda.max_memory_allocated.return_value = 5 * 1024**3
        mock_torch.cuda.get_device_properties.return_value = SimpleNamespace(
            total_mem=8 * 1024**3,
        )
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024 * 1024 * 256)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback()
        cb.on_log(_make_args(), _make_state(), _make_control())

        assert cb._peak_gpu_allocated == 5 * 1024**3
        assert cb._peak_gpu_reserved == 6 * 1024**3

    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_gpu_warning_at_high_utilization(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        total_mem = 8 * 1024**3
        mock_torch.cuda.memory_allocated.return_value = int(total_mem * 0.96)
        mock_torch.cuda.memory_reserved.return_value = total_mem
        mock_torch.cuda.max_memory_allocated.return_value = int(total_mem * 0.96)
        mock_torch.cuda.get_device_properties.return_value = SimpleNamespace(
            total_mem=total_mem,
        )
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024**2)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback(gpu_warn_threshold=0.95)
        # Should not raise, just log warning internally
        cb.on_log(_make_args(), _make_state(), _make_control())

    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_no_warning_below_threshold(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        total_mem = 8 * 1024**3
        mock_torch.cuda.memory_allocated.return_value = int(total_mem * 0.5)
        mock_torch.cuda.memory_reserved.return_value = int(total_mem * 0.6)
        mock_torch.cuda.max_memory_allocated.return_value = int(total_mem * 0.5)
        mock_torch.cuda.get_device_properties.return_value = SimpleNamespace(
            total_mem=total_mem,
        )
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024**2)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback()
        cb.on_log(_make_args(), _make_state(), _make_control())

    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_on_train_end_with_gpu(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024**2 * 300)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback()
        cb._peak_gpu_allocated = 4 * 1024**3
        cb._peak_gpu_reserved = 6 * 1024**3
        cb.on_train_end(_make_args(), _make_state(), _make_control())

    @patch("src.training.callbacks.torch")
    @patch("src.training.callbacks.psutil.Process")
    def test_on_train_end_without_gpu(
        self, mock_proc_cls: MagicMock, mock_torch: MagicMock,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value = SimpleNamespace(rss=1024**2 * 100)
        mock_proc_cls.return_value = mock_proc

        cb = MemoryMonitorCallback()
        cb.on_train_end(_make_args(), _make_state(), _make_control())

    def test_custom_warn_threshold(self) -> None:
        cb = MemoryMonitorCallback(gpu_warn_threshold=0.80)
        assert cb._gpu_warn_threshold == 0.80


# ═══════════════════════════════════════════════════════════════
# LossSpikingCallback
# ═══════════════════════════════════════════════════════════════


class TestLossSpikingCallback:
    def test_normal_loss_no_stop(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        for i in range(10):
            cb.on_log(
                _make_args(),
                _make_state(global_step=i),
                control,
                logs={"loss": 2.0 - i * 0.1},
            )
        assert control.should_training_stop is False

    def test_nan_loss_stops_training(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs={"loss": float("nan")})
        assert control.should_training_stop is True
        assert control.should_save is True

    def test_inf_loss_stops_training(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs={"loss": float("inf")})
        assert control.should_training_stop is True
        assert control.should_save is True

    def test_negative_inf_loss_stops_training(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs={"loss": float("-inf")})
        assert control.should_training_stop is True

    def test_spike_detection(self) -> None:
        cb = LossSpikingCallback(window_size=5, spike_factor=3.0)
        control = _make_control()
        # Fill window with normal losses
        for i in range(5):
            cb.on_log(
                _make_args(),
                _make_state(global_step=i),
                control,
                logs={"loss": 1.0},
            )
        # Now a spike: 10.0 > 3 * 1.0
        cb.on_log(
            _make_args(),
            _make_state(global_step=5),
            control,
            logs={"loss": 10.0},
        )
        # Spike is logged as critical but does NOT stop training
        assert control.should_training_stop is False

    def test_no_spike_within_factor(self) -> None:
        cb = LossSpikingCallback(window_size=5, spike_factor=3.0)
        control = _make_control()
        for i in range(5):
            cb.on_log(
                _make_args(), _make_state(global_step=i), control,
                logs={"loss": 1.0},
            )
        # 2.5 < 3 * 1.0 = 3.0 -> no spike
        cb.on_log(
            _make_args(), _make_state(global_step=5), control,
            logs={"loss": 2.5},
        )

    def test_plateau_detection(self) -> None:
        cb = LossSpikingCallback(plateau_steps=10)
        control = _make_control()
        # Set best loss at step 0
        cb.on_log(
            _make_args(), _make_state(global_step=0), control,
            logs={"loss": 1.0},
        )
        # No improvement for 15 steps (exceeds plateau_steps=10)
        cb.on_log(
            _make_args(), _make_state(global_step=15), control,
            logs={"loss": 1.5},
        )
        # Should have logged warning but not stopped
        assert control.should_training_stop is False

    def test_plateau_resets_on_improvement(self) -> None:
        cb = LossSpikingCallback(plateau_steps=10)
        control = _make_control()
        cb.on_log(
            _make_args(), _make_state(global_step=0), control,
            logs={"loss": 2.0},
        )
        # Improve at step 5
        cb.on_log(
            _make_args(), _make_state(global_step=5), control,
            logs={"loss": 1.5},
        )
        assert cb._best_loss == pytest.approx(1.5)
        assert cb._best_loss_step == 5

    def test_no_loss_in_logs_skipped(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs={"lr": 1e-4})
        assert len(cb._window) == 0

    def test_none_logs_skipped(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs=None)
        assert len(cb._window) == 0

    def test_window_size_respected(self) -> None:
        cb = LossSpikingCallback(window_size=5)
        control = _make_control()
        for i in range(20):
            cb.on_log(
                _make_args(), _make_state(global_step=i), control,
                logs={"loss": float(i)},
            )
        assert len(cb._window) == 5

    def test_custom_spike_factor(self) -> None:
        cb = LossSpikingCallback(spike_factor=2.0)
        assert cb._spike_factor == 2.0

    def test_nan_does_not_enter_window(self) -> None:
        cb = LossSpikingCallback()
        control = _make_control()
        cb.on_log(_make_args(), _make_state(), control, logs={"loss": float("nan")})
        assert len(cb._window) == 0


# ═══════════════════════════════════════════════════════════════
# CheckpointCleanupCallback
# ═══════════════════════════════════════════════════════════════


class TestCheckpointCleanupCallback:
    def test_init_strips_trailing_slash(self) -> None:
        cb = CheckpointCleanupCallback(
            checkpoint_s3_uri="s3://bucket/prefix/",
            save_total_limit=3,
        )
        assert cb._checkpoint_s3_uri == "s3://bucket/prefix"

    @patch.object(CheckpointCleanupCallback, "_upload_to_s3")
    @patch.object(CheckpointCleanupCallback, "_cleanup_local")
    def test_on_save_calls_upload_and_cleanup(
        self, mock_cleanup: MagicMock, mock_upload: MagicMock,
    ) -> None:
        cb = CheckpointCleanupCallback(
            checkpoint_s3_uri="s3://my-bucket/checkpoints",
        )
        args = _make_args(output_dir="/tmp/output")
        state = _make_state(global_step=200)

        cb.on_save(args, state, _make_control())

        mock_upload.assert_called_once()
        upload_args = mock_upload.call_args[0]
        assert upload_args[0] == Path("/tmp/output/checkpoint-200")
        assert upload_args[1] == "s3://my-bucket/checkpoints/checkpoint-200"

        mock_cleanup.assert_called_once_with(Path("/tmp/output"))

    @patch.object(CheckpointCleanupCallback, "_upload_to_s3")
    def test_cleanup_local_removes_old(
        self, mock_upload: MagicMock, tmp_path: Path,
    ) -> None:
        # Create 5 checkpoint dirs
        for step in [100, 200, 300, 400, 500]:
            (tmp_path / f"checkpoint-{step}").mkdir()

        cb = CheckpointCleanupCallback(
            checkpoint_s3_uri="s3://bucket/prefix",
            save_total_limit=3,
        )
        cb._cleanup_local(tmp_path)

        remaining = sorted(tmp_path.glob("checkpoint-*"))
        assert len(remaining) == 3
        assert remaining[0].name == "checkpoint-300"
        assert remaining[-1].name == "checkpoint-500"

    @patch.object(CheckpointCleanupCallback, "_upload_to_s3")
    def test_cleanup_local_no_excess(
        self, mock_upload: MagicMock, tmp_path: Path,
    ) -> None:
        for step in [100, 200]:
            (tmp_path / f"checkpoint-{step}").mkdir()

        cb = CheckpointCleanupCallback(
            checkpoint_s3_uri="s3://bucket/prefix",
            save_total_limit=5,
        )
        cb._cleanup_local(tmp_path)

        remaining = sorted(tmp_path.glob("checkpoint-*"))
        assert len(remaining) == 2

    @patch("src.training.callbacks.boto3.client")
    def test_upload_to_s3(self, mock_boto: MagicMock, tmp_path: Path) -> None:
        checkpoint = tmp_path / "checkpoint-100"
        checkpoint.mkdir()
        (checkpoint / "model.safetensors").write_text("data")
        (checkpoint / "config.json").write_text("{}")

        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3

        cb = CheckpointCleanupCallback(checkpoint_s3_uri="s3://bucket/prefix")
        cb._upload_to_s3(checkpoint, "s3://bucket/prefix/checkpoint-100")

        assert mock_s3.upload_file.call_count == 2

    def test_default_save_total_limit(self) -> None:
        cb = CheckpointCleanupCallback(checkpoint_s3_uri="s3://bucket/prefix")
        assert cb._save_total_limit == 3


# ═══════════════════════════════════════════════════════════════
# GradientNormCallback
# ═══════════════════════════════════════════════════════════════


class TestGradientNormCallback:
    def test_logs_grad_norm_from_logs(self) -> None:
        cb = GradientNormCallback()
        control = _make_control()
        cb.on_log(
            _make_args(), _make_state(global_step=10), control,
            logs={"grad_norm": 1.234, "loss": 0.5},
        )

    def test_no_grad_norm_in_logs(self) -> None:
        cb = GradientNormCallback()
        cb.on_log(
            _make_args(), _make_state(), _make_control(),
            logs={"loss": 0.5},
        )

    def test_none_logs_skipped(self) -> None:
        cb = GradientNormCallback()
        cb.on_log(_make_args(), _make_state(), _make_control(), logs=None)

    @patch("src.training.callbacks.torch")
    def test_per_layer_norms_at_interval(self, mock_torch: MagicMock) -> None:
        cb = GradientNormCallback(per_layer_interval=100)

        # Create mock model with named parameters
        param1 = MagicMock()
        param1.grad = MagicMock()
        mock_torch.norm.return_value = MagicMock(item=lambda: 0.5)

        model = MagicMock()
        model.named_parameters.return_value = [("layer1.weight", param1)]

        cb.on_log(
            _make_args(),
            _make_state(global_step=100),
            _make_control(),
            logs={"grad_norm": 1.0},
            model=model,
        )

        model.named_parameters.assert_called_once()

    def test_per_layer_norms_skipped_off_interval(self) -> None:
        cb = GradientNormCallback(per_layer_interval=100)
        model = MagicMock()

        cb.on_log(
            _make_args(),
            _make_state(global_step=50),
            _make_control(),
            logs={"grad_norm": 1.0},
            model=model,
        )

        model.named_parameters.assert_not_called()

    def test_per_layer_norms_skipped_at_step_zero(self) -> None:
        cb = GradientNormCallback(per_layer_interval=100)
        model = MagicMock()

        cb.on_log(
            _make_args(),
            _make_state(global_step=0),
            _make_control(),
            logs={"grad_norm": 1.0},
            model=model,
        )

        model.named_parameters.assert_not_called()

    def test_no_model_kwarg_skips_per_layer(self) -> None:
        cb = GradientNormCallback(per_layer_interval=100)
        # No model kwarg -> should not crash
        cb.on_log(
            _make_args(),
            _make_state(global_step=100),
            _make_control(),
            logs={"grad_norm": 1.0},
        )

    @patch("src.training.callbacks.torch")
    def test_skips_params_without_grad(self, mock_torch: MagicMock) -> None:
        cb = GradientNormCallback(per_layer_interval=100)

        param_with_grad = MagicMock()
        param_with_grad.grad = MagicMock()
        mock_torch.norm.return_value = MagicMock(item=lambda: 0.3)

        param_no_grad = MagicMock()
        param_no_grad.grad = None

        model = MagicMock()
        model.named_parameters.return_value = [
            ("layer1.weight", param_with_grad),
            ("layer2.weight", param_no_grad),
        ]

        cb.on_log(
            _make_args(),
            _make_state(global_step=100),
            _make_control(),
            logs={"grad_norm": 1.0},
            model=model,
        )

    def test_custom_per_layer_interval(self) -> None:
        cb = GradientNormCallback(per_layer_interval=50)
        assert cb._per_layer_interval == 50


# ═══════════════════════════════════════════════════════════════
# EarlyStoppingWithPatience
# ═══════════════════════════════════════════════════════════════


class TestEarlyStoppingWithPatience:
    def test_improvement_resets_counter(self) -> None:
        cb = EarlyStoppingWithPatience(patience=3, min_delta=0.01)
        control = _make_control()
        cb.on_evaluate(
            _make_args(), _make_state(), control,
            metrics={"eval_loss": 2.0},
        )
        assert cb._counter == 0
        assert cb._best_loss == pytest.approx(2.0)

    def test_no_improvement_increments_counter(self) -> None:
        cb = EarlyStoppingWithPatience(patience=3, min_delta=0.01)
        control = _make_control()
        cb.on_evaluate(
            _make_args(), _make_state(global_step=100), control,
            metrics={"eval_loss": 2.0},
        )
        cb.on_evaluate(
            _make_args(), _make_state(global_step=200), control,
            metrics={"eval_loss": 2.005},  # not enough improvement
        )
        assert cb._counter == 1

    def test_stops_after_patience_exhausted(self) -> None:
        cb = EarlyStoppingWithPatience(patience=3, min_delta=0.01)
        control = _make_control()

        # First eval sets baseline
        cb.on_evaluate(
            _make_args(), _make_state(global_step=100), control,
            metrics={"eval_loss": 1.0},
        )
        # 3 evals without improvement
        for step in [200, 300, 400]:
            cb.on_evaluate(
                _make_args(), _make_state(global_step=step), control,
                metrics={"eval_loss": 1.0},
            )

        assert cb._counter == 3
        assert control.should_training_stop is True

    def test_improvement_after_partial_patience(self) -> None:
        cb = EarlyStoppingWithPatience(patience=5, min_delta=0.01)
        control = _make_control()

        cb.on_evaluate(
            _make_args(), _make_state(global_step=100), control,
            metrics={"eval_loss": 2.0},
        )
        # 2 evals without improvement
        cb.on_evaluate(
            _make_args(), _make_state(global_step=200), control,
            metrics={"eval_loss": 2.0},
        )
        cb.on_evaluate(
            _make_args(), _make_state(global_step=300), control,
            metrics={"eval_loss": 2.0},
        )
        assert cb._counter == 2

        # Now improve
        cb.on_evaluate(
            _make_args(), _make_state(global_step=400), control,
            metrics={"eval_loss": 1.5},
        )
        assert cb._counter == 0
        assert cb._best_loss == pytest.approx(1.5)
        assert control.should_training_stop is False

    def test_min_delta_threshold(self) -> None:
        cb = EarlyStoppingWithPatience(patience=3, min_delta=0.1)
        control = _make_control()

        cb.on_evaluate(
            _make_args(), _make_state(global_step=100), control,
            metrics={"eval_loss": 2.0},
        )
        # Improve by 0.05 (< min_delta=0.1) -> not counted
        cb.on_evaluate(
            _make_args(), _make_state(global_step=200), control,
            metrics={"eval_loss": 1.95},
        )
        assert cb._counter == 1

    def test_large_improvement_resets(self) -> None:
        cb = EarlyStoppingWithPatience(patience=3, min_delta=0.1)
        control = _make_control()

        cb.on_evaluate(
            _make_args(), _make_state(global_step=100), control,
            metrics={"eval_loss": 2.0},
        )
        cb.on_evaluate(
            _make_args(), _make_state(global_step=200), control,
            metrics={"eval_loss": 2.0},
        )
        assert cb._counter == 1

        # Big drop
        cb.on_evaluate(
            _make_args(), _make_state(global_step=300), control,
            metrics={"eval_loss": 1.5},
        )
        assert cb._counter == 0

    def test_no_metrics_skipped(self) -> None:
        cb = EarlyStoppingWithPatience()
        control = _make_control()
        cb.on_evaluate(_make_args(), _make_state(), control, metrics=None)
        assert cb._counter == 0

    def test_no_eval_loss_skipped(self) -> None:
        cb = EarlyStoppingWithPatience()
        control = _make_control()
        cb.on_evaluate(
            _make_args(), _make_state(), control,
            metrics={"train_loss": 0.5},
        )
        assert cb._counter == 0

    def test_default_patience_and_delta(self) -> None:
        cb = EarlyStoppingWithPatience()
        assert cb._patience == 5
        assert cb._min_delta == pytest.approx(1e-4)


# ═══════════════════════════════════════════════════════════════
# Module exports
# ═══════════════════════════════════════════════════════════════


class TestModuleExports:
    def test_all_callbacks_in_module_all(self) -> None:
        from src.training import callbacks

        expected = {
            "LoggingCallback",
            "CostTrackingCallback",
            "MemoryMonitorCallback",
            "LossSpikingCallback",
            "CheckpointCleanupCallback",
            "GradientNormCallback",
            "EarlyStoppingWithPatience",
        }
        assert expected == set(callbacks.__all__)

    def test_all_importable_from_package(self) -> None:
        from src.training import (
            CheckpointCleanupCallback,
            CostTrackingCallback,
            EarlyStoppingWithPatience,
            GradientNormCallback,
            LoggingCallback,
            LossSpikingCallback,
            MemoryMonitorCallback,
        )

        assert CostTrackingCallback is not None
        assert MemoryMonitorCallback is not None
        assert LossSpikingCallback is not None
        assert CheckpointCleanupCallback is not None
        assert GradientNormCallback is not None
        assert EarlyStoppingWithPatience is not None
