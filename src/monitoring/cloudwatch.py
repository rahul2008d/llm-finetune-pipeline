"""CloudWatch metrics publishing for training and inference monitoring."""

from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class CloudWatchMetrics:
    """Publish custom metrics to AWS CloudWatch."""

    def __init__(self, namespace: str, region: str = "us-east-1") -> None:
        """Initialize CloudWatch metrics publisher.

        Args:
            namespace: CloudWatch namespace for the metrics.
            region: AWS region.
        """
        self.namespace = namespace
        self.region = region

    def _get_client(self) -> Any:
        """Get a CloudWatch boto3 client.

        Returns:
            Boto3 CloudWatch client.
        """
        import boto3

        return boto3.client("cloudwatch", region_name=self.region)

    def put_metric(
        self,
        metric_name: str,
        value: float,
        unit: str = "None",
        dimensions: dict[str, str] | None = None,
    ) -> None:
        """Publish a single metric to CloudWatch.

        Args:
            metric_name: Name of the metric.
            value: Metric value.
            unit: CloudWatch unit type.
            dimensions: Optional dimension key-value pairs.
        """
        client = self._get_client()
        metric_data: dict[str, Any] = {
            "MetricName": metric_name,
            "Value": value,
            "Unit": unit,
            "Timestamp": datetime.now(tz=timezone.utc),
        }
        if dimensions:
            metric_data["Dimensions"] = [
                {"Name": k, "Value": v} for k, v in dimensions.items()
            ]

        client.put_metric_data(
            Namespace=self.namespace,
            MetricData=[metric_data],
        )
        logger.info("Metric published", metric=metric_name, value=value)

    def put_training_metrics(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        epoch: float,
    ) -> None:
        """Publish a batch of training metrics.

        Args:
            step: Current training step.
            loss: Training loss value.
            learning_rate: Current learning rate.
            epoch: Current epoch number.
        """
        dims = {"Step": str(step)}
        self.put_metric("TrainingLoss", loss, dimensions=dims)
        self.put_metric("LearningRate", learning_rate, dimensions=dims)
        self.put_metric("Epoch", epoch, dimensions=dims)

    def put_inference_metrics(
        self,
        endpoint_name: str,
        latency_ms: float,
        tokens_generated: int,
    ) -> None:
        """Publish inference metrics.

        Args:
            endpoint_name: Name of the serving endpoint.
            latency_ms: Inference latency in milliseconds.
            tokens_generated: Number of tokens generated.
        """
        dims = {"Endpoint": endpoint_name}
        self.put_metric("InferenceLatency", latency_ms, unit="Milliseconds", dimensions=dims)
        self.put_metric("TokensGenerated", float(tokens_generated), dimensions=dims)


__all__: list[str] = ["CloudWatchMetrics"]
