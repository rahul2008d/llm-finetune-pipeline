"""Unit tests for monitoring.mlflow_tracker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, PropertyMock
from typing import Any

import pytest

from src.config.training import (
    LoRAConfig,
    ModelConfig,
    QuantizationConfig,
    TrainingHyperparameters,
    TrainingJobConfig,
)


@pytest.fixture()
def sample_config() -> TrainingJobConfig:
    """Create a sample TrainingJobConfig for testing."""
    return TrainingJobConfig(
        experiment_name="test-experiment",
        run_name="test-run",
        model=ModelConfig(model_name_or_path="meta-llama/Llama-3-8B"),
        quantization=QuantizationConfig(method="qlora"),
        lora=LoRAConfig(r=64, lora_alpha=128),
        training=TrainingHyperparameters(),
        dataset_path="/tmp/dataset",
    )


class TestExperimentTracker:
    """Tests for ExperimentTracker."""

    @patch("src.monitoring.mlflow_tracker.mlflow")
    @patch("src.monitoring.mlflow_tracker._MLFLOW_AVAILABLE", True)
    def test_start_run_logs_params(
        self, mock_mlflow: MagicMock, sample_config: TrainingJobConfig
    ) -> None:
        """Verify mlflow.log_params called with flattened config."""
        from src.monitoring.mlflow_tracker import ExperimentTracker

        mock_run = MagicMock()
        mock_run.info.run_id = "test-run-id"
        mock_mlflow.start_run.return_value = mock_run

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = True
        tracker._warned = False
        tracker.experiment_name = "test-experiment"

        tracker.start_run("test-run", sample_config)

        mock_mlflow.log_params.assert_called_once()
        params = mock_mlflow.log_params.call_args[0][0]
        assert "model.model_name_or_path" in params
        assert "lora.r" in params

    @patch("src.monitoring.mlflow_tracker.mlflow")
    @patch("src.monitoring.mlflow_tracker._MLFLOW_AVAILABLE", True)
    def test_log_training_metrics(self, mock_mlflow: MagicMock) -> None:
        """Verify mlflow.log_metrics called with step."""
        from src.monitoring.mlflow_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = True
        tracker._warned = False

        metrics = {"train_loss": 0.5, "eval_loss": 0.6}
        tracker.log_training_metrics(metrics, step=100)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=100)

    @patch("src.monitoring.mlflow_tracker.mlflow")
    @patch("src.monitoring.mlflow_tracker._MLFLOW_AVAILABLE", True)
    def test_log_eval_metrics(self, mock_mlflow: MagicMock) -> None:
        """Verify metrics are logged."""
        from src.monitoring.mlflow_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = True
        tracker._warned = False

        metrics = {"perplexity": 12.5, "bleu": 0.45}
        tracker.log_eval_metrics(metrics)

        mock_mlflow.log_metrics.assert_called_once_with(metrics)

    @patch("src.monitoring.mlflow_tracker.mlflow")
    @patch("src.monitoring.mlflow_tracker._MLFLOW_AVAILABLE", True)
    def test_compare_runs_returns_sorted(self, mock_mlflow: MagicMock) -> None:
        """Mock mlflow.search_runs, verify sort order."""
        import pandas as pd
        from src.monitoring.mlflow_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = True
        tracker._warned = False

        mock_experiment = MagicMock()
        mock_experiment.experiment_id = "123"
        mock_mlflow.get_experiment_by_name.return_value = mock_experiment

        mock_df = pd.DataFrame(
            {
                "run_id": ["run1", "run2"],
                "tags.mlflow.runName": ["run-a", "run-b"],
                "params.lr": ["0.001", "0.002"],
                "metrics.eval_loss": [0.3, 0.5],
            }
        )
        mock_mlflow.search_runs.return_value = mock_df

        results = tracker.compare_runs("test-experiment", metric="eval_loss", top_k=5)

        assert len(results) == 2
        assert results[0]["run_id"] == "run1"
        assert "params" in results[0]
        assert "metrics" in results[0]

    def test_graceful_when_mlflow_unavailable(self) -> None:
        """Verify no crash when MLflow is unavailable."""
        from src.monitoring.mlflow_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = False
        tracker._warned = False
        tracker.experiment_name = "test"

        # All these should be no-ops
        run = tracker.start_run("run", MagicMock())
        assert run is not None  # Should return _NoOpContext

        tracker.log_training_metrics({"loss": 0.5}, step=1)
        tracker.log_eval_metrics({"perplexity": 10.0})
        tracker.log_model_artifact("/tmp/model", "model")
        tracker.log_model_card("# Card")
        tracker.end_run()

        results = tracker.compare_runs("test")
        assert results == []

    @patch("src.monitoring.mlflow_tracker.mlflow")
    @patch("src.monitoring.mlflow_tracker._MLFLOW_AVAILABLE", True)
    def test_end_run_sets_status(self, mock_mlflow: MagicMock) -> None:
        """Verify mlflow.end_run called with status."""
        from src.monitoring.mlflow_tracker import ExperimentTracker

        tracker = ExperimentTracker.__new__(ExperimentTracker)
        tracker._enabled = True
        tracker._warned = False

        tracker.end_run(status="FAILED")

        mock_mlflow.end_run.assert_called_once_with(status="FAILED")
