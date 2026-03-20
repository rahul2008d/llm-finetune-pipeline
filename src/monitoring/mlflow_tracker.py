"""MLflow experiment tracking wrapper for the fine-tuning pipeline."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
from contextlib import contextmanager
from typing import Any, Generator

import structlog
import yaml

from src.config.training import TrainingJobConfig

logger = structlog.get_logger(__name__)

try:
    import mlflow
    from mlflow.entities import RunStatus

    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


class _NoOpContext:
    """No-op context manager returned when MLflow is unavailable."""

    info: Any = None

    def __enter__(self) -> _NoOpContext:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


class ExperimentTracker:
    """Centralized MLflow tracking for training, evaluation, and model registration."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Configure MLflow with S3 artifact store.

        If MLflow is unavailable or tracking_uri is empty, all methods become no-ops
        and log a warning on first call.

        Args:
            tracking_uri: MLflow tracking server URI.
            experiment_name: Name of the MLflow experiment.
        """
        self._enabled = _MLFLOW_AVAILABLE and bool(tracking_uri)
        self._warned = False
        self.experiment_name = experiment_name

        if self._enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            logger.info(
                "MLflow tracking configured",
                tracking_uri=tracking_uri,
                experiment=experiment_name,
            )
        else:
            logger.warning(
                "MLflow tracking disabled",
                mlflow_available=_MLFLOW_AVAILABLE,
                tracking_uri=tracking_uri,
            )

    def _warn_once(self) -> None:
        """Log a warning once if MLflow is unavailable."""
        if not self._warned:
            logger.warning("MLflow is unavailable — tracking call is a no-op")
            self._warned = True

    @staticmethod
    def _get_git_sha() -> str:
        """Retrieve the current git SHA from environment or subprocess."""
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
    def _flatten_config(config: TrainingJobConfig) -> dict[str, str]:
        """Flatten a TrainingJobConfig into dot-separated key-value pairs."""
        flat: dict[str, str] = {}
        config_dict = config.model_dump()
        for section, values in config_dict.items():
            if isinstance(values, dict):
                for k, v in values.items():
                    flat[f"{section}.{k}"] = str(v)
            elif values is not None:
                flat[section] = str(values)
        return flat

    def start_run(self, run_name: str, config: TrainingJobConfig) -> Any:
        """Start an MLflow run and log ALL config parameters.

        - mlflow.start_run(run_name=run_name)
        - Log all config parameters flattened: model.torch_dtype, lora.r, etc.
        - Log tags: git_sha, dataset_id, instance_type, method (qlora/dora)
        - Log the full config YAML as an artifact
        - Return the MLflow run object (or a no-op context if MLflow unavailable)

        Args:
            run_name: Name for the MLflow run.
            config: Training job configuration.

        Returns:
            MLflow run context or a no-op context.
        """
        if not self._enabled:
            self._warn_once()
            return _NoOpContext()

        run = mlflow.start_run(run_name=run_name)

        # Log flattened config parameters
        params = self._flatten_config(config)
        mlflow.log_params(params)

        # Log tags
        tags = {
            "git_sha": self._get_git_sha(),
            "dataset_id": config.dataset_id or "none",
            "method": config.quantization.method,
        }
        if config.sagemaker:
            tags["instance_type"] = config.sagemaker.instance_type
        mlflow.set_tags(tags)

        # Log full config YAML as artifact
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix="config_"
        ) as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False)
            mlflow.log_artifact(f.name, "config")

        logger.info("MLflow run started", run_name=run_name, run_id=run.info.run_id)
        return run

    def log_training_metrics(self, metrics: dict[str, float], step: int) -> None:
        """Log training step metrics.

        Required metrics: train_loss, eval_loss, learning_rate, epoch,
        gpu_memory_mb, gradient_norm, estimated_cost_usd.

        Args:
            metrics: Dictionary of metric values.
            step: Current training step.
        """
        if not self._enabled:
            self._warn_once()
            return

        mlflow.log_metrics(metrics, step=step)
        logger.debug("Training metrics logged", step=step, metrics=list(metrics.keys()))

    def log_eval_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation results: perplexity, bleu, rouge, custom metrics.

        Args:
            metrics: Dictionary of evaluation metric values.
        """
        if not self._enabled:
            self._warn_once()
            return

        mlflow.log_metrics(metrics)
        logger.info("Evaluation metrics logged", metrics=list(metrics.keys()))

    def log_model_artifact(self, model_path: str, artifact_name: str) -> None:
        """Log model artifacts (adapter weights, merged model) to MLflow.

        Args:
            model_path: Local path to the model artifacts.
            artifact_name: Name for the artifact in MLflow.
        """
        if not self._enabled:
            self._warn_once()
            return

        mlflow.log_artifact(model_path, artifact_name)
        logger.info("Model artifact logged", path=model_path, name=artifact_name)

    def log_model_card(self, model_card_content: str) -> None:
        """Serialize model card markdown and log as artifact.

        Args:
            model_card_content: Model card content in Markdown format.
        """
        if not self._enabled:
            self._warn_once()
            return

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, prefix="model_card_"
        ) as f:
            f.write(model_card_content)
            mlflow.log_artifact(f.name, "model_card")

        logger.info("Model card logged to MLflow")

    def end_run(self, status: str = "FINISHED") -> None:
        """End the active MLflow run with given status.

        Args:
            status: Run status (FINISHED, FAILED, KILLED).
        """
        if not self._enabled:
            self._warn_once()
            return

        mlflow.end_run(status=status)
        logger.info("MLflow run ended", status=status)

    def compare_runs(
        self,
        experiment_name: str,
        metric: str = "eval_loss",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Query MLflow for top runs by metric.

        Returns sorted list of dicts with: run_id, run_name, params, metrics.
        Returns empty list if MLflow unavailable.

        Args:
            experiment_name: Name of the experiment to search.
            metric: Metric to sort by.
            top_k: Number of top runs to return.

        Returns:
            List of run summary dictionaries sorted by metric.
        """
        if not self._enabled:
            self._warn_once()
            return []

        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("Experiment not found", experiment=experiment_name)
            return []

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=top_k,
        )

        results: list[dict[str, Any]] = []
        for _, row in runs.iterrows():
            result: dict[str, Any] = {
                "run_id": row.get("run_id", ""),
                "run_name": row.get("tags.mlflow.runName", ""),
                "params": {
                    k.replace("params.", ""): v
                    for k, v in row.items()
                    if isinstance(k, str) and k.startswith("params.")
                },
                "metrics": {
                    k.replace("metrics.", ""): v
                    for k, v in row.items()
                    if isinstance(k, str) and k.startswith("metrics.")
                },
            }
            results.append(result)

        logger.info(
            "Runs compared",
            experiment=experiment_name,
            metric=metric,
            count=len(results),
        )
        return results


__all__: list[str] = ["ExperimentTracker"]
