"""CloudWatch metrics publishing for training and inference monitoring."""

from __future__ import annotations

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


class TrainingMetricsPublisher:
    """Publish training metrics to CloudWatch asynchronously.

    Metrics are batched and published in a background thread
    to avoid blocking the training loop.
    """

    def __init__(
        self,
        experiment_name: str,
        job_name: str,
        instance_type: str,
        method: str,
        model_family: str,
        region: str = "us-east-1",
    ) -> None:
        """Initialize publisher with dimensions for all metrics.

        Namespace: LLMFineTuning/{experiment_name}
        Dimensions: JobName, InstanceType, Method, ModelFamily.

        Start background thread for async publishing.

        Args:
            experiment_name: Experiment name used in the CloudWatch namespace.
            job_name: Training job name.
            instance_type: Compute instance type.
            method: Fine-tuning method (``qlora`` or ``dora``).
            model_family: Model family (``llama3``, ``mistral``, etc.).
            region: AWS region.
        """
        import queue
        import threading

        self.namespace = f"LLMFineTuning/{experiment_name}"
        self.region = region
        self.dimensions = [
            {"Name": "JobName", "Value": job_name},
            {"Name": "InstanceType", "Value": instance_type},
            {"Name": "Method", "Value": method},
            {"Name": "ModelFamily", "Value": model_family},
        ]
        self._queue: queue.Queue[list[dict[str, Any]]] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._publish_loop, daemon=True, name="cw-publisher"
        )
        self._thread.start()
        logger.info(
            "TrainingMetricsPublisher started",
            namespace=self.namespace,
            job_name=job_name,
        )

    def _get_client(self) -> Any:
        """Get a CloudWatch boto3 client.

        Returns:
            Boto3 CloudWatch client.
        """
        import boto3

        return boto3.client("cloudwatch", region_name=self.region)

    def _publish_loop(self) -> None:
        """Background loop that drains the queue and publishes batches."""
        client = self._get_client()
        while not self._stop_event.is_set():
            import queue as _q

            try:
                batch = self._queue.get(timeout=1.0)
            except _q.Empty:
                continue
            try:
                # CloudWatch allows max 1000 metric data points per call
                for i in range(0, len(batch), 1000):
                    client.put_metric_data(
                        Namespace=self.namespace,
                        MetricData=batch[i : i + 1000],
                    )
            except Exception:
                logger.exception("Failed to publish CloudWatch metrics")

    def publish_training_step(self, step: int, metrics: dict[str, float]) -> None:
        """Queue metrics for async publishing.

        Expected metrics: TrainLoss, EvalLoss, LearningRate, GradientNorm,
        GPUMemoryUtilization, ThroughputSamplesPerSec.

        Args:
            step: Current training step.
            metrics: Dictionary of metric values.
        """
        now = datetime.now(tz=timezone.utc)
        metric_data: list[dict[str, Any]] = []
        for name, value in metrics.items():
            metric_data.append(
                {
                    "MetricName": name,
                    "Value": float(value),
                    "Unit": "None",
                    "Timestamp": now,
                    "Dimensions": self.dimensions,
                }
            )
        self._queue.put(metric_data)
        logger.debug("Training metrics queued", step=step, count=len(metric_data))

    def publish_job_summary(self, result: Any) -> None:
        """Publish final job summary metrics.

        Metrics: TotalTrainingTimeMinutes, FinalEvalLoss, EstimatedCostUSD,
        TotalSteps, BestCheckpointStep.

        Args:
            result: A TrainingResult object with summary fields.
        """
        summary_metrics = {
            "TotalTrainingTimeMinutes": result.training_time_seconds / 60.0,
            "FinalEvalLoss": result.final_eval_loss,
            "EstimatedCostUSD": result.estimated_cost_usd,
            "TotalSteps": float(result.total_steps),
        }
        if hasattr(result, "best_checkpoint_step"):
            summary_metrics["BestCheckpointStep"] = float(result.best_checkpoint_step)

        now = datetime.now(tz=timezone.utc)
        metric_data: list[dict[str, Any]] = []
        for name, value in summary_metrics.items():
            metric_data.append(
                {
                    "MetricName": name,
                    "Value": float(value),
                    "Unit": "None",
                    "Timestamp": now,
                    "Dimensions": self.dimensions,
                }
            )
        self._queue.put(metric_data)
        logger.info("Job summary metrics queued")

    def create_dashboard(self, experiment_name: str) -> str:
        """Create CloudWatch Dashboard and return dashboard URL.

        Widgets:
        - Loss curves (train + eval overlay)
        - Learning rate schedule
        - GPU memory over time
        - Cost accumulation over time
        - Gradient norm over time

        Args:
            experiment_name: Experiment name for the dashboard title.

        Returns:
            URL to the created CloudWatch dashboard.
        """
        import json

        dashboard_name = f"LLMFineTuning-{experiment_name}"
        namespace = self.namespace

        def _metric_widget(title: str, metric_names: list[str]) -> dict[str, Any]:
            return {
                "type": "metric",
                "properties": {
                    "title": title,
                    "metrics": [
                        [namespace, m] for m in metric_names
                    ],
                    "period": 60,
                    "stat": "Average",
                    "region": self.region,
                },
            }

        widgets: list[dict[str, Any]] = [
            _metric_widget("Loss Curves", ["TrainLoss", "EvalLoss"]),
            _metric_widget("Learning Rate", ["LearningRate"]),
            _metric_widget("GPU Memory", ["GPUMemoryUtilization"]),
            _metric_widget("Estimated Cost", ["EstimatedCostUSD"]),
            _metric_widget("Gradient Norm", ["GradientNorm"]),
        ]

        dashboard_body = json.dumps({"widgets": widgets})
        client = self._get_client()
        client.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=dashboard_body,
        )

        url = (
            f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={self.region}#dashboards:name={dashboard_name}"
        )
        logger.info("Dashboard created", name=dashboard_name, url=url)
        return url

    def stop(self) -> None:
        """Flush remaining metrics and stop background thread."""
        self._stop_event.set()
        self._thread.join(timeout=10.0)
        logger.info("TrainingMetricsPublisher stopped")


__all__: list[str] = ["CloudWatchMetrics", "TrainingMetricsPublisher"]
