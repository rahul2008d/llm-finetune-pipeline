"""Automated model card generation in Hugging Face format."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import structlog
import yaml

from src.config.training import TrainingJobConfig

logger = structlog.get_logger(__name__)

try:
    import mlflow

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class ModelCardGenerator:
    """Generate standardized model cards for fine-tuned models."""

    def generate(
        self,
        config: TrainingJobConfig,
        result: Any,
        eval_results: dict[str, Any] | None = None,
    ) -> str:
        """Generate a complete model card in Markdown format.

        Sections (Hugging Face standard):
        1. Model Details
        2. Training Details
        3. Adapter Details
        4. Performance
        5. Intended Use
        6. Limitations
        7. Ethical Considerations
        8. Lineage
        9. How to Use

        Args:
            config: Training job configuration.
            result: TrainingResult from the training run.
            eval_results: Optional evaluation results dictionary.

        Returns:
            Markdown string of the model card.
        """
        sections: list[str] = []

        # 1. Model Details
        sections.append("# Model Card")
        sections.append("")
        sections.append("## Model Details")
        sections.append("")
        sections.append(f"- **Base Model**: {config.model.model_name_or_path}")
        sections.append(f"- **Fine-tuning Method**: {config.quantization.method}")
        sections.append(f"- **Experiment**: {config.experiment_name}")
        if config.run_name:
            sections.append(f"- **Run Name**: {config.run_name}")
        sections.append("")

        # 2. Training Details
        sections.append("## Training Details")
        sections.append("")
        sections.append("### Hyperparameters")
        sections.append("")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        sections.append(f"| Epochs | {config.training.num_train_epochs} |")
        sections.append(f"| Batch Size (per device) | {config.training.per_device_train_batch_size} |")
        sections.append(f"| Gradient Accumulation Steps | {config.training.gradient_accumulation_steps} |")
        sections.append(f"| Learning Rate | {config.training.learning_rate} |")
        sections.append(f"| Weight Decay | {config.training.weight_decay} |")
        sections.append(f"| Warmup Ratio | {config.training.warmup_ratio} |")
        sections.append(f"| LR Scheduler | {config.training.lr_scheduler_type} |")
        sections.append(f"| Max Grad Norm | {config.training.max_grad_norm} |")
        sections.append(f"| Optimizer | {config.training.optim} |")
        sections.append(f"| Precision | {'bf16' if config.training.bf16 else 'fp16' if config.training.fp16 else 'fp32'} |")
        sections.append(f"| Seed | {config.training.seed} |")
        sections.append("")
        if config.dataset_id:
            sections.append(f"- **Dataset**: {config.dataset_id}")
        elif config.dataset_path:
            sections.append(f"- **Dataset Path**: {config.dataset_path}")
        if config.sagemaker:
            sections.append(f"- **Instance Type**: {config.sagemaker.instance_type}")
        sections.append(f"- **Training Duration**: {result.training_time_seconds:.1f}s")
        sections.append(f"- **Estimated Cost**: ${result.estimated_cost_usd:.2f}")
        sections.append("")

        # 3. Adapter Details
        sections.append("## Adapter Details")
        sections.append("")
        sections.append("| Parameter | Value |")
        sections.append("|-----------|-------|")
        sections.append(f"| Rank (r) | {config.lora.r} |")
        sections.append(f"| Alpha | {config.lora.lora_alpha} |")
        sections.append(f"| Dropout | {config.lora.lora_dropout} |")
        sections.append(f"| Target Modules | {', '.join(config.lora.target_modules)} |")
        sections.append(f"| DoRA | {config.lora.use_dora} |")
        sections.append(f"| RS-LoRA | {config.lora.use_rslora} |")
        sections.append(f"| Task Type | {config.lora.task_type} |")
        sections.append("")

        # 4. Performance
        sections.append("## Performance")
        sections.append("")
        sections.append(f"- **Final Train Loss**: {result.final_train_loss:.4f}")
        sections.append(f"- **Final Eval Loss**: {result.final_eval_loss:.4f}")
        sections.append(f"- **Best Eval Loss**: {result.best_eval_loss:.4f}")
        sections.append(f"- **Total Steps**: {result.total_steps}")
        sections.append("")
        if eval_results:
            sections.append("### Benchmark Results")
            sections.append("")
            sections.append("| Benchmark | Score |")
            sections.append("|-----------|-------|")
            for name, score in eval_results.items():
                if isinstance(score, float):
                    sections.append(f"| {name} | {score:.4f} |")
                else:
                    sections.append(f"| {name} | {score} |")
            sections.append("")
        else:
            sections.append("No evaluation results available.")
            sections.append("")

        # 5. Intended Use
        sections.append("## Intended Use")
        sections.append("")
        sections.append(
            "This model is a fine-tuned adapter for domain-specific tasks. "
            "It should be used with the corresponding base model and PEFT library."
        )
        sections.append("")

        # 6. Limitations
        sections.append("## Limitations")
        sections.append("")
        sections.append(
            "- Fine-tuned on a domain-specific dataset; may not generalize well.\n"
            "- Inherits biases and limitations of the base model.\n"
            "- Performance may degrade on out-of-distribution inputs."
        )
        sections.append("")

        # 7. Ethical Considerations
        sections.append("## Ethical Considerations")
        sections.append("")
        sections.append(
            "- Users should review outputs for potential bias.\n"
            "- PII scanning should be performed on training data.\n"
            "- Data provenance should be documented and verified."
        )
        sections.append("")

        # 8. Lineage
        sections.append("## Lineage")
        sections.append("")
        if config.dataset_id:
            sections.append(f"- **Dataset ID**: {config.dataset_id}")
        if config.dataset_path:
            sections.append(f"- **Dataset Path**: {config.dataset_path}")
        git_sha = self._get_git_sha()
        sections.append(f"- **Code Commit**: {git_sha}")
        config_hash = self._compute_config_hash(config)
        sections.append(f"- **Config Hash**: {config_hash}")
        sections.append("")

        # 9. How to Use
        sections.append("## How to Use")
        sections.append("")
        sections.append("```python")
        sections.append("from peft import PeftModel")
        sections.append("from transformers import AutoModelForCausalLM, AutoTokenizer")
        sections.append("")
        sections.append(f'base_model = AutoModelForCausalLM.from_pretrained("{config.model.model_name_or_path}")')
        sections.append(f'tokenizer = AutoTokenizer.from_pretrained("{config.model.model_name_or_path}")')
        sections.append(f'model = PeftModel.from_pretrained(base_model, "{result.adapter_s3_uri}")')
        sections.append("```")
        sections.append("")

        return "\n".join(sections)

    def save(self, content: str, output_path: str) -> None:
        """Save model card to file (local or S3).

        - If output_path starts with s3://, upload via boto3
        - Otherwise write to local file as README.md

        Args:
            content: Model card markdown content.
            output_path: Local path or S3 URI.
        """
        if output_path.startswith("s3://"):
            import boto3

            parts = output_path.replace("s3://", "").split("/", 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else "README.md"
            s3 = boto3.client("s3")
            s3.put_object(Bucket=bucket, Key=key, Body=content.encode("utf-8"))
            logger.info("Model card saved to S3", path=output_path)
        else:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("Model card saved locally", path=str(path))

    def save_json(
        self,
        config: TrainingJobConfig,
        result: Any,
        eval_results: dict[str, Any] | None,
        output_path: str,
    ) -> None:
        """Save model card metadata as JSON for programmatic access.

        Args:
            config: Training job configuration.
            result: TrainingResult from the training run.
            eval_results: Optional evaluation results dictionary.
            output_path: Path to write the JSON file.
        """
        metadata = {
            "model_details": {
                "base_model": config.model.model_name_or_path,
                "method": config.quantization.method,
                "experiment_name": config.experiment_name,
            },
            "training_results": {
                "final_train_loss": result.final_train_loss,
                "final_eval_loss": result.final_eval_loss,
                "best_eval_loss": result.best_eval_loss,
                "total_steps": result.total_steps,
                "training_time_seconds": result.training_time_seconds,
                "estimated_cost_usd": result.estimated_cost_usd,
            },
            "adapter_config": {
                "r": config.lora.r,
                "alpha": config.lora.lora_alpha,
                "target_modules": config.lora.target_modules,
                "use_dora": config.lora.use_dora,
                "use_rslora": config.lora.use_rslora,
            },
            "eval_results": eval_results or {},
            "lineage": {
                "dataset_id": config.dataset_id,
                "dataset_path": config.dataset_path,
                "git_sha": self._get_git_sha(),
                "config_hash": self._compute_config_hash(config),
            },
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        logger.info("Model card JSON saved", path=str(path))

    def log_to_mlflow(self, content: str) -> None:
        """Log model card as MLflow artifact if MLflow is available.

        Args:
            content: Model card markdown content.
        """
        if not _MLFLOW_AVAILABLE:
            logger.warning("MLflow not available — skipping model card artifact logging")
            return

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, prefix="model_card_"
        ) as f:
            f.write(content)
            mlflow.log_artifact(f.name, "model_card")
        logger.info("Model card logged to MLflow")

    @staticmethod
    def _get_git_sha() -> str:
        """Retrieve the current git SHA."""
        import os

        sha = os.environ.get("GIT_SHA", "")
        if not sha:
            try:
                sha = (
                    subprocess.check_output(
                        ["git", "rev-parse", "HEAD"],
                        stderr=subprocess.DEVNULL,
                    )
                    .decode()
                    .strip()
                )
            except (subprocess.SubprocessError, FileNotFoundError):
                sha = "unknown"
        return sha

    @staticmethod
    def _compute_config_hash(config: TrainingJobConfig) -> str:
        """Compute SHA256 hash of the serialized config.

        Args:
            config: Training job configuration.

        Returns:
            Hex digest of the SHA256 hash.
        """
        serialized = json.dumps(config.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode()).hexdigest()


__all__: list[str] = ["ModelCardGenerator"]
