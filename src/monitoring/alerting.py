"""Alerting system for sending notifications via SNS."""

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class AlertManager:
    """Manage alerts and notifications via AWS SNS."""

    def __init__(self, topic_arn: str, region: str = "us-east-1") -> None:
        """Initialize the alert manager.

        Args:
            topic_arn: SNS topic ARN for sending alerts.
            region: AWS region.
        """
        self.topic_arn = topic_arn
        self.region = region

    def _get_client(self) -> Any:
        """Get an SNS boto3 client.

        Returns:
            Boto3 SNS client.
        """
        import boto3

        return boto3.client("sns", region_name=self.region)

    def send_alert(
        self,
        subject: str,
        message: str,
        severity: str = "WARNING",
    ) -> str:
        """Send an alert notification via SNS.

        Args:
            subject: Alert subject line.
            message: Alert message body.
            severity: Alert severity level (INFO, WARNING, CRITICAL).

        Returns:
            SNS message ID.
        """
        client = self._get_client()
        full_message = f"[{severity}] {message}"
        response = client.publish(
            TopicArn=self.topic_arn,
            Subject=subject[:100],
            Message=full_message,
        )
        message_id: str = response["MessageId"]
        logger.info(
            "Alert sent",
            subject=subject,
            severity=severity,
            message_id=message_id,
        )
        return message_id

    def send_training_alert(
        self,
        model_id: str,
        event: str,
        details: dict[str, Any],
    ) -> str:
        """Send a training-specific alert.

        Args:
            model_id: Model identifier.
            event: Event type (e.g., 'training_complete', 'training_failed').
            details: Event details dictionary.

        Returns:
            SNS message ID.
        """
        import json

        subject = f"Training Alert: {event} - {model_id}"
        message = json.dumps(
            {"model_id": model_id, "event": event, "details": details},
            indent=2,
            default=str,
        )
        severity = "CRITICAL" if "fail" in event.lower() else "INFO"
        return self.send_alert(subject, message, severity)

    def send_drift_alert(
        self,
        endpoint_name: str,
        drift_results: dict[str, Any],
    ) -> str:
        """Send a drift detection alert.

        Args:
            endpoint_name: Name of the affected endpoint.
            drift_results: Drift detection results.

        Returns:
            SNS message ID.
        """
        import json

        subject = f"Drift Alert: {endpoint_name}"
        message = json.dumps(
            {"endpoint": endpoint_name, "drift": drift_results},
            indent=2,
            default=str,
        )
        return self.send_alert(subject, message, severity="WARNING")


__all__: list[str] = ["AlertManager"]
