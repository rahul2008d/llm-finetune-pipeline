"""Training callbacks for logging, cost tracking, memory monitoring, and safety."""
from __future__ import annotations

import math
import shutil
import time
from collections import deque
from pathlib import Path
from typing import Any

import boto3
import psutil
import structlog
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = structlog.get_logger(__name__)

# ── On-demand pricing (USD/hour) for common SageMaker training instances ──
_INSTANCE_COST_PER_HOUR: dict[str, float] = {
    "ml.g5.xlarge": 1.006,
    "ml.g5.2xlarge": 1.212,
    "ml.g5.4xlarge": 1.624,
    "ml.g5.8xlarge": 2.448,
    "ml.g5.12xlarge": 4.896,
    "ml.g5.16xlarge": 4.896,
    "ml.g5.24xlarge": 6.528,
    "ml.g5.48xlarge": 13.056,
    "ml.p4d.24xlarge": 32.7726,
    "ml.p5.48xlarge": 98.32,
    "ml.trn1.32xlarge": 21.50,
}


# ═══════════════════════════════════════════════════════════════
# LoggingCallback (preserved from original)
# ═══════════════════════════════════════════════════════════════


class LoggingCallback(TrainerCallback):
    """Callback that logs training metrics via structlog."""

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if logs:
            safe_logs = {k: v for k, v in logs.items() if k not in ("epoch", "global_step")}
            logger.info(
                "Training step",
                global_step=state.global_step,
                epoch=state.epoch,
                **safe_logs,
            )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        logger.info("Epoch complete", epoch=state.epoch, global_step=state.global_step)


# ═══════════════════════════════════════════════════════════════
# CostTrackingCallback
# ═══════════════════════════════════════════════════════════════


class CostTrackingCallback(TrainerCallback):
    """Track estimated training cost and enforce budget limits."""

    def __init__(
        self,
        instance_type: str,
        instance_count: int = 1,
        max_budget_usd: float | None = None,
    ) -> None:
        self.cost_per_hour = _INSTANCE_COST_PER_HOUR.get(instance_type, 0.0)
        self.instance_count = instance_count
        self.max_budget_usd = max_budget_usd
        self._start_time: float | None = None
        self._estimated_cost: float = 0.0

        if self.cost_per_hour == 0.0:
            logger.warning(
                "Unknown instance type for cost tracking, cost will be 0",
                instance_type=instance_type,
            )

    def _elapsed_hours(self) -> float:
        if self._start_time is None:
            return 0.0
        return (time.monotonic() - self._start_time) / 3600.0

    def _compute_cost(self) -> float:
        return self._elapsed_hours() * self.cost_per_hour * self.instance_count

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._start_time = time.monotonic()
        logger.info(
            "Cost tracking started",
            cost_per_hour=self.cost_per_hour,
            instance_count=self.instance_count,
            max_budget_usd=self.max_budget_usd,
        )

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._estimated_cost = self._compute_cost()
        logger.info(
            "Cost update",
            estimated_cost_usd=round(self._estimated_cost, 4),
            elapsed_hours=round(self._elapsed_hours(), 4),
            global_step=state.global_step,
        )

        if self.max_budget_usd is not None and self._estimated_cost >= self.max_budget_usd:
            logger.critical(
                "Budget exceeded — stopping training",
                estimated_cost_usd=round(self._estimated_cost, 4),
                max_budget_usd=self.max_budget_usd,
            )
            control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._estimated_cost = self._compute_cost()
        logger.info(
            "Training cost summary",
            total_cost_usd=round(self._estimated_cost, 4),
            elapsed_hours=round(self._elapsed_hours(), 4),
        )


# ═══════════════════════════════════════════════════════════════
# MemoryMonitorCallback
# ═══════════════════════════════════════════════════════════════


class MemoryMonitorCallback(TrainerCallback):
    """Monitor GPU and CPU memory usage during training."""

    def __init__(self, gpu_warn_threshold: float = 0.95) -> None:
        self._gpu_warn_threshold = gpu_warn_threshold
        self._peak_gpu_allocated: int = 0
        self._peak_gpu_reserved: int = 0

    def _log_memory(self, state: TrainerState) -> None:
        # CPU
        proc = psutil.Process()
        cpu_rss_mb = proc.memory_info().rss / (1024 * 1024)

        log_kwargs: dict[str, Any] = {
            "global_step": state.global_step,
            "cpu_rss_mb": round(cpu_rss_mb, 1),
        }

        # GPU (if available)
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            max_allocated = torch.cuda.max_memory_allocated()
            total = torch.cuda.get_device_properties(0).total_mem

            self._peak_gpu_allocated = max(self._peak_gpu_allocated, max_allocated)
            self._peak_gpu_reserved = max(self._peak_gpu_reserved, reserved)

            utilization = allocated / total if total > 0 else 0.0

            log_kwargs.update(
                gpu_allocated_mb=round(allocated / (1024 * 1024), 1),
                gpu_reserved_mb=round(reserved / (1024 * 1024), 1),
                gpu_max_allocated_mb=round(max_allocated / (1024 * 1024), 1),
                gpu_utilization_pct=round(utilization * 100, 1),
            )

            if utilization > self._gpu_warn_threshold:
                logger.warning(
                    "GPU memory utilization critical",
                    gpu_utilization_pct=round(utilization * 100, 1),
                    threshold_pct=round(self._gpu_warn_threshold * 100, 1),
                    global_step=state.global_step,
                )

        logger.info("Memory stats", **log_kwargs)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        self._log_memory(state)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        summary: dict[str, Any] = {"global_step": state.global_step}

        if torch.cuda.is_available():
            summary.update(
                peak_gpu_allocated_mb=round(self._peak_gpu_allocated / (1024 * 1024), 1),
                peak_gpu_reserved_mb=round(self._peak_gpu_reserved / (1024 * 1024), 1),
            )

        proc = psutil.Process()
        summary["final_cpu_rss_mb"] = round(proc.memory_info().rss / (1024 * 1024), 1)
        logger.info("Peak memory summary", **summary)


# ═══════════════════════════════════════════════════════════════
# LossSpikingCallback
# ═══════════════════════════════════════════════════════════════


class LossSpikingCallback(TrainerCallback):
    """Detect loss spikes, NaN/Inf loss, and training plateaus."""

    def __init__(
        self,
        window_size: int = 50,
        spike_factor: float = 3.0,
        plateau_steps: int = 500,
    ) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._spike_factor = spike_factor
        self._plateau_steps = plateau_steps
        self._best_loss: float = float("inf")
        self._best_loss_step: int = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs or "loss" not in logs:
            return

        loss = logs["loss"]

        # NaN / Inf check
        if math.isnan(loss) or math.isinf(loss):
            logger.critical(
                "Loss is NaN/Inf — saving emergency checkpoint and stopping",
                loss=loss,
                global_step=state.global_step,
            )
            control.should_training_stop = True
            control.should_save = True
            return

        # Spike detection
        if len(self._window) >= 1:
            rolling_mean = sum(self._window) / len(self._window)
            if rolling_mean > 0 and loss > self._spike_factor * rolling_mean:
                logger.critical(
                    "Loss spike detected",
                    current_loss=round(loss, 6),
                    rolling_mean=round(rolling_mean, 6),
                    spike_factor=self._spike_factor,
                    global_step=state.global_step,
                )

        self._window.append(loss)

        # Track best loss / plateau detection
        if loss < self._best_loss:
            self._best_loss = loss
            self._best_loss_step = state.global_step
        else:
            steps_since_improvement = state.global_step - self._best_loss_step
            if steps_since_improvement >= self._plateau_steps:
                logger.warning(
                    "Possible training plateau",
                    steps_since_improvement=steps_since_improvement,
                    best_loss=round(self._best_loss, 6),
                    current_loss=round(loss, 6),
                    global_step=state.global_step,
                )


# ═══════════════════════════════════════════════════════════════
# CheckpointCleanupCallback
# ═══════════════════════════════════════════════════════════════


class CheckpointCleanupCallback(TrainerCallback):
    """Upload checkpoints to S3 and enforce local retention limits."""

    def __init__(self, checkpoint_s3_uri: str, save_total_limit: int = 3) -> None:
        self._checkpoint_s3_uri = checkpoint_s3_uri.rstrip("/")
        self._save_total_limit = save_total_limit

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        output_dir = Path(args.output_dir)
        checkpoint_name = f"checkpoint-{state.global_step}"
        checkpoint_dir = output_dir / checkpoint_name

        # Upload to S3
        s3_uri = f"{self._checkpoint_s3_uri}/{checkpoint_name}"
        logger.info(
            "Uploading checkpoint to S3",
            checkpoint=str(checkpoint_dir),
            s3_uri=s3_uri,
            global_step=state.global_step,
        )
        self._upload_to_s3(checkpoint_dir, s3_uri)

        # Cleanup old local checkpoints
        self._cleanup_local(output_dir)

        logger.info(
            "Checkpoint metadata",
            checkpoint_name=checkpoint_name,
            global_step=state.global_step,
            s3_uri=s3_uri,
            epoch=state.epoch,
        )

    def _upload_to_s3(self, local_dir: Path, s3_uri: str) -> None:
        # Parse s3://bucket/prefix
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client("s3")
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                key = f"{prefix}/{file_path.relative_to(local_dir)}"
                s3.upload_file(str(file_path), bucket, key)

    def _cleanup_local(self, output_dir: Path) -> None:
        checkpoints = sorted(
            output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[-1]),
        )
        while len(checkpoints) > self._save_total_limit:
            oldest = checkpoints.pop(0)
            logger.info("Removing old local checkpoint", path=str(oldest))
            shutil.rmtree(oldest)


# ═══════════════════════════════════════════════════════════════
# GradientNormCallback
# ═══════════════════════════════════════════════════════════════


class GradientNormCallback(TrainerCallback):
    """Log gradient norms at each logging step and per-layer norms periodically."""

    def __init__(self, per_layer_interval: int = 100) -> None:
        self._per_layer_interval = per_layer_interval

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if not logs:
            return

        grad_norm = logs.get("grad_norm")
        if grad_norm is not None:
            logger.info(
                "Gradient norm",
                grad_norm=round(float(grad_norm), 6),
                global_step=state.global_step,
            )

        # Per-layer norms every N steps
        if state.global_step > 0 and state.global_step % self._per_layer_interval == 0:
            model = kwargs.get("model")
            if model is not None:
                self._log_per_layer_norms(model, state.global_step)

    def _log_per_layer_norms(self, model: Any, global_step: int) -> None:
        layer_norms: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_norms[name] = round(
                    torch.norm(param.grad, p=2).item(), 6,
                )

        if layer_norms:
            logger.info(
                "Per-layer gradient norms",
                global_step=global_step,
                num_layers=len(layer_norms),
                layer_norms=layer_norms,
            )


# ═══════════════════════════════════════════════════════════════
# EarlyStoppingWithPatience
# ═══════════════════════════════════════════════════════════════


class EarlyStoppingWithPatience(TrainerCallback):
    """Patience-based early stopping on eval_loss."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._best_loss: float = float("inf")
        self._counter: int = 0

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if metrics is None:
            return

        eval_loss = metrics.get("eval_loss")
        if eval_loss is None:
            return

        if eval_loss < self._best_loss - self._min_delta:
            self._best_loss = eval_loss
            self._counter = 0
            logger.info(
                "Eval loss improved",
                eval_loss=round(eval_loss, 6),
                best_loss=round(self._best_loss, 6),
                patience_counter=self._counter,
                global_step=state.global_step,
            )
        else:
            self._counter += 1
            logger.info(
                "Eval loss did not improve",
                eval_loss=round(eval_loss, 6),
                best_loss=round(self._best_loss, 6),
                patience_counter=self._counter,
                patience=self._patience,
                global_step=state.global_step,
            )

            if self._counter >= self._patience:
                logger.warning(
                    "Early stopping triggered",
                    patience=self._patience,
                    best_loss=round(self._best_loss, 6),
                    global_step=state.global_step,
                )
                control.should_training_stop = True


__all__: list[str] = [
    "LoggingCallback",
    "CostTrackingCallback",
    "MemoryMonitorCallback",
    "LossSpikingCallback",
    "CheckpointCleanupCallback",
    "GradientNormCallback",
    "EarlyStoppingWithPatience",
]
