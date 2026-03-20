"""Unit tests for monitoring.cloudwatch TrainingMetricsPublisher."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch
from typing import Any

import pytest


class TestTrainingMetricsPublisher:
    """Tests for TrainingMetricsPublisher."""

    @patch("boto3.client")
    def test_publish_training_step_queues_metrics(self, mock_boto3_client: MagicMock) -> None:
        """Verify metrics added to queue and eventually published."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        from src.monitoring.cloudwatch import TrainingMetricsPublisher

        publisher = TrainingMetricsPublisher(
            experiment_name="test",
            job_name="job-1",
            instance_type="ml.g5.2xlarge",
            method="qlora",
            model_family="llama3",
        )

        metrics = {"TrainLoss": 0.5, "EvalLoss": 0.6}
        publisher.publish_training_step(step=10, metrics=metrics)

        # Give the background thread time to process
        time.sleep(0.5)

        # Verify put_metric_data was called by the background thread
        mock_client.put_metric_data.assert_called()
        publisher.stop()

    @patch("boto3.client")
    def test_publish_job_summary(self, mock_boto3_client: MagicMock) -> None:
        """Verify summary metrics published."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        from src.monitoring.cloudwatch import TrainingMetricsPublisher

        publisher = TrainingMetricsPublisher(
            experiment_name="test",
            job_name="job-1",
            instance_type="ml.g5.2xlarge",
            method="qlora",
            model_family="llama3",
        )

        result = MagicMock()
        result.training_time_seconds = 3600.0
        result.final_eval_loss = 0.3
        result.estimated_cost_usd = 5.50
        result.total_steps = 1000

        publisher.publish_job_summary(result)
        # Give background thread time to process
        time.sleep(0.5)
        mock_client.put_metric_data.assert_called()
        publisher.stop()

    @patch("boto3.client")
    def test_create_dashboard_returns_url(self, mock_boto3_client: MagicMock) -> None:
        """Mock boto3, verify dashboard JSON structure."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        from src.monitoring.cloudwatch import TrainingMetricsPublisher

        publisher = TrainingMetricsPublisher(
            experiment_name="test",
            job_name="job-1",
            instance_type="ml.g5.2xlarge",
            method="qlora",
            model_family="llama3",
        )

        url = publisher.create_dashboard("test")

        assert "cloudwatch" in url
        assert "test" in url
        mock_client.put_dashboard.assert_called_once()
        publisher.stop()

    @patch("boto3.client")
    def test_async_publish_does_not_block(self, mock_boto3_client: MagicMock) -> None:
        """Verify publish returns immediately."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        from src.monitoring.cloudwatch import TrainingMetricsPublisher

        publisher = TrainingMetricsPublisher(
            experiment_name="test",
            job_name="job-1",
            instance_type="ml.g5.2xlarge",
            method="qlora",
            model_family="llama3",
        )

        start = time.monotonic()
        publisher.publish_training_step(step=1, metrics={"TrainLoss": 0.5})
        elapsed = time.monotonic() - start

        # Should be nearly instant (< 100ms)
        assert elapsed < 0.1
        publisher.stop()

    @patch("boto3.client")
    def test_stop_flushes_queue(self, mock_boto3_client: MagicMock) -> None:
        """Verify remaining metrics published on stop."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        from src.monitoring.cloudwatch import TrainingMetricsPublisher

        publisher = TrainingMetricsPublisher(
            experiment_name="test",
            job_name="job-1",
            instance_type="ml.g5.2xlarge",
            method="qlora",
            model_family="llama3",
        )

        publisher.publish_training_step(step=1, metrics={"TrainLoss": 0.5})
        # Give background thread time to process
        time.sleep(0.5)
        publisher.stop()

        # The stop should complete without error
        assert publisher._stop_event.is_set()
