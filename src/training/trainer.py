"""Fine-tuning trainer: wraps TRL SFTTrainer with config-driven setup."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Any

import boto3
import mlflow
import structlog
import torch
from datasets import load_from_disk
from pydantic import BaseModel, Field
from transformers import TrainerCallback, TrainingArguments, set_seed
from trl import SFTTrainer

from src.config.training import TrainingJobConfig
from src.training.callbacks import (
    CheckpointCleanupCallback,
    CostTrackingCallback,
    EarlyStoppingWithPatience,
    GradientNormCallback,
    LoggingCallback,
    LossSpikingCallback,
    MemoryMonitorCallback,
)
from src.training.model_loader import ModelLoader

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# TrainingResult
# ═══════════════════════════════════════════════════════════════


class TrainingResult(BaseModel):
    """Immutable record of a completed training run."""

    run_id: str
    experiment_name: str
    final_train_loss: float
    final_eval_loss: float
    best_eval_loss: float
    total_steps: int
    training_time_seconds: float
    estimated_cost_usd: float
    adapter_s3_uri: str
    metrics: dict[str, float] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# FineTuneTrainer
# ═══════════════════════════════════════════════════════════════


class FineTuneTrainer:
    """Config-driven fine-tuning orchestrator backed by TRL SFTTrainer."""

    def __init__(self, config: TrainingJobConfig) -> None:
        self.config = config
        self.model_loader = ModelLoader()
        self.callbacks = self._build_callbacks()

    # ── public API ──────────────────────────────────────────────

    def train(self) -> TrainingResult:
        """Run the full fine-tuning pipeline and return results."""
        wall_start = time.monotonic()

        # 1. Reproducibility
        self._set_seeds(self.config.training.seed)

        # 2. Tokenizer
        logger.info("Loading tokenizer")
        tokenizer = self.model_loader.load_tokenizer(self.config.model)

        # 3. Base model + quantization
        logger.info("Loading base model with quantization")
        base_model = self.model_loader.load_base_model(
            self.config.model,
            self.config.quantization,
            no_cuda=self.config.training.no_cuda,
        )

        # 4. LoRA / DoRA adapters
        logger.info("Applying LoRA adapters")
        model = self.model_loader.apply_lora(base_model, self.config.lora)

        # 5. Dataset
        logger.info(
            "dataset_loading_mode",
            mode="registry" if self.config.dataset_id else "direct_path",
        )
        train_dataset, eval_dataset = self._load_datasets()

        # 5b. Auto-format raw datasets (Alpaca instruction/input/output → text)
        train_dataset, dataset_text_field = self._prepare_dataset(train_dataset)
        if eval_dataset is not None:
            eval_dataset, _ = self._prepare_dataset(eval_dataset)

        # 6. MLflow
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        try:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            mlflow_available = True
        except Exception as e:
            logger.warning("MLflow not available, skipping experiment tracking", error=str(e))
            mlflow_available = False

        # 7. TrainingArguments
        output_dir = tempfile.mkdtemp(prefix="finetune-")
        training_args = self._build_training_args(
            output_dir,
            remove_unused_columns=dataset_text_field is not None,
        )

        # 8. SFTTrainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_seq_length=self.config.model.max_seq_length,
            packing=False,
            dataset_text_field=dataset_text_field,
            callbacks=self.callbacks,
        )

        # 9. Train
        logger.info(
            "Starting training",
            resume=self.config.resume_from_checkpoint,
        )

        def _do_train() -> tuple[Any, str]:
            if self.config.resume_from_checkpoint:
                return trainer.train(
                    resume_from_checkpoint=self.config.resume_from_checkpoint,
                ), "resume"
            return trainer.train(), "fresh"

        if mlflow_available:
            with mlflow.start_run(run_name=self.config.run_name) as run:
                mlflow.log_params(self._flat_config_params())
                train_output, _ = _do_train()
                run_id = run.info.run_id
        else:
            train_output, _ = _do_train()
            run_id = "local-no-mlflow"

        # 10. Post-training
        metrics = train_output.metrics
        logger.info("Training complete", metrics=metrics)

        # Save adapter + tokenizer locally
        adapter_dir = Path(output_dir) / "final_adapter"
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))

        # Upload to S3
        adapter_s3_uri = self._upload_adapter(adapter_dir)

        # Log to MLflow
        if mlflow_available:
            mlflow.log_metrics(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            )

        # Extract result values
        train_loss = metrics.get("train_loss", 0.0)
        eval_loss = metrics.get("eval_loss", 0.0)
        best_eval = (
            trainer.state.best_metric
            if trainer.state.best_metric is not None
            else eval_loss
        )
        cost = self._get_cost_estimate()
        wall_secs = time.monotonic() - wall_start

        result = TrainingResult(
            run_id=run_id,
            experiment_name=self.config.experiment_name,
            final_train_loss=float(train_loss),
            final_eval_loss=float(eval_loss),
            best_eval_loss=float(best_eval),
            total_steps=trainer.state.global_step,
            training_time_seconds=round(wall_secs, 2),
            estimated_cost_usd=round(cost, 4),
            adapter_s3_uri=adapter_s3_uri,
            metrics={
                k: float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float))
            },
        )

        if mlflow_available:
            mlflow.log_metrics({
                "total_training_time_seconds": result.training_time_seconds,
                "estimated_cost_usd": result.estimated_cost_usd,
            })

        logger.info("Training result", result=result.model_dump())
        return result

    # ── private helpers ─────────────────────────────────────────

    @staticmethod
    def _set_seeds(seed: int) -> None:
        set_seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _load_datasets(self) -> tuple[Any, Any]:
        if self.config.dataset_id is not None:
            try:
                from src.data.registry import DatasetRegistry
                registry = DatasetRegistry()
                dataset_info = registry.get(self.config.dataset_id)
                train_path = dataset_info.train_path
                val_path = dataset_info.validation_path
            except ImportError:
                raise ImportError(
                    "DatasetRegistry not available. Either run Prompts 17-18 "
                    "to create it, or use 'dataset_path' in your config "
                    "instead of 'dataset_id'."
                )
        elif self.config.dataset_path is not None:
            base = self.config.dataset_path
            train_subdir = os.path.join(base, "train")
            is_local = not base.startswith("s3://")
            if not is_local or os.path.isdir(train_subdir):
                train_path = train_subdir
                val_path = os.path.join(base, "validation")
            else:
                # Flat local dataset directory (no train/validation split)
                train_path = base
                val_path = None
        else:
            raise ValueError("No dataset source configured")

        train_ds = load_from_disk(train_path)
        eval_ds = None
        if val_path:
            try:
                eval_ds = load_from_disk(val_path)
            except (FileNotFoundError, ValueError):
                logger.info("No validation split found, training without eval set")
        return train_ds, eval_ds

    @staticmethod
    def _prepare_dataset(dataset: Any) -> tuple[Any, str | None]:
        """Detect dataset format and convert raw Alpaca data to text if needed.

        Returns (dataset, dataset_text_field) where dataset_text_field is
        ``"text"`` for raw/text datasets or ``None`` for pre-tokenized ones.
        """
        cols = set(dataset.column_names)
        if "input_ids" in cols:
            # Already tokenized — pass through
            return dataset, None
        if "instruction" in cols and "text" not in cols:
            # Alpaca format → convert to text column
            def _fmt(example: dict) -> dict:
                parts = [f"### Instruction:\n{example['instruction']}"]
                if example.get("input"):
                    parts.append(f"\n### Input:\n{example['input']}")
                parts.append(f"\n### Response:\n{example.get('output', '')}")
                return {"text": "\n".join(parts)}

            dataset = dataset.map(_fmt, remove_columns=dataset.column_names)
            return dataset, "text"
        if "text" in cols:
            return dataset, "text"
        return dataset, None

    def _build_training_args(
        self,
        output_dir: str,
        *,
        remove_unused_columns: bool = False,
    ) -> TrainingArguments:
        hp = self.config.training
        return TrainingArguments(
            output_dir=output_dir,
            run_name=self.config.run_name,
            # Epochs / batching
            num_train_epochs=hp.num_train_epochs,
            per_device_train_batch_size=hp.per_device_train_batch_size,
            per_device_eval_batch_size=hp.per_device_eval_batch_size,
            gradient_accumulation_steps=hp.gradient_accumulation_steps,
            # Optimiser
            learning_rate=hp.learning_rate,
            weight_decay=hp.weight_decay,
            warmup_ratio=hp.warmup_ratio,
            lr_scheduler_type=hp.lr_scheduler_type,
            max_grad_norm=hp.max_grad_norm,
            optim=hp.optim,
            # Precision
            bf16=hp.bf16,
            fp16=hp.fp16,
            # Checkpointing
            gradient_checkpointing=hp.gradient_checkpointing,
            gradient_checkpointing_kwargs=hp.gradient_checkpointing_kwargs,
            # Logging / eval / save
            logging_steps=hp.logging_steps,
            eval_strategy=hp.eval_strategy,
            eval_steps=hp.eval_steps,
            save_strategy=hp.save_strategy,
            save_steps=hp.save_steps,
            save_total_limit=hp.save_total_limit,
            load_best_model_at_end=hp.load_best_model_at_end,
            metric_for_best_model=hp.metric_for_best_model,
            greater_is_better=hp.greater_is_better,
            # Reproducibility
            seed=hp.seed,
            # Dataloader
            dataloader_num_workers=hp.dataloader_num_workers,
            dataloader_pin_memory=hp.dataloader_pin_memory,
            group_by_length=hp.group_by_length,
            # Reporting
            report_to=hp.report_to,
            # DDP / misc
            ddp_find_unused_parameters=False,
            remove_unused_columns=remove_unused_columns,
            no_cuda=hp.no_cuda,
        )

    def _build_callbacks(self) -> list[TrainerCallback]:
        sm = self.config.sagemaker
        if sm is None:
            return [
                LoggingCallback(),
                MemoryMonitorCallback(),
                LossSpikingCallback(),
                GradientNormCallback(),
            ]
        return [
            LoggingCallback(),
            CostTrackingCallback(
                instance_type=sm.instance_type,
                instance_count=sm.instance_count,
            ),
            MemoryMonitorCallback(),
            LossSpikingCallback(),
            CheckpointCleanupCallback(
                checkpoint_s3_uri=sm.checkpoint_s3_uri or self.config.output_s3_uri,
                save_total_limit=self.config.training.save_total_limit,
            ),
            GradientNormCallback(),
            EarlyStoppingWithPatience(),
        ]

    def _upload_adapter(self, adapter_dir: Path) -> str:
        if not self.config.output_s3_uri:
            # Local-only mode: return the local path
            local_path = self.config.output_local_path
            if local_path:
                dest = Path(local_path) / (self.config.run_name or "adapter")
                dest.mkdir(parents=True, exist_ok=True)
                import shutil
                for item in adapter_dir.rglob("*"):
                    if item.is_file():
                        target = dest / item.relative_to(adapter_dir)
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(str(item), str(target))
                logger.info("Adapter saved locally", path=str(dest))
                return str(dest)
            return str(adapter_dir)

        s3_uri = f"{self.config.output_s3_uri.rstrip('/')}/{self.config.run_name}/adapter"
        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client("s3")
        for file_path in adapter_dir.rglob("*"):
            if file_path.is_file():
                key = f"{prefix}/{file_path.relative_to(adapter_dir)}"
                s3.upload_file(str(file_path), bucket, key)
                logger.info("Uploaded", key=key, bucket=bucket)

        logger.info("Adapter uploaded to S3", s3_uri=s3_uri)
        return s3_uri

    def _get_cost_estimate(self) -> float:
        for cb in self.callbacks:
            if isinstance(cb, CostTrackingCallback):
                return cb._estimated_cost
        return 0.0

    def _flat_config_params(self) -> dict[str, Any]:
        """Flatten config into a dict for MLflow param logging."""
        params: dict[str, Any] = {
            "model_name": self.config.model.model_name_or_path,
            "quant_method": self.config.quantization.method,
            "lora_r": self.config.lora.r,
            "lora_alpha": self.config.lora.lora_alpha,
            "lora_dropout": self.config.lora.lora_dropout,
            "use_dora": self.config.lora.use_dora,
            "use_rslora": self.config.lora.use_rslora,
            "learning_rate": self.config.training.learning_rate,
            "num_epochs": self.config.training.num_train_epochs,
            "batch_size": self.config.training.per_device_train_batch_size,
            "grad_accum": self.config.training.gradient_accumulation_steps,
            "max_seq_length": self.config.model.max_seq_length,
        }
        if self.config.sagemaker is not None:
            params["instance_type"] = self.config.sagemaker.instance_type
            params["instance_count"] = self.config.sagemaker.instance_count
        else:
            params["instance_type"] = "local"
            params["instance_count"] = 1
        return params


__all__: list[str] = ["FineTuneTrainer", "TrainingResult"]
