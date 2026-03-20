"""Alerting system for sending notifications via SNS."""

from __future__ import annotations

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

    def send_deployment_event(
        self,
        event_type: str,
        endpoint_name: str,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Send deployment lifecycle notification.

        Args:
            event_type: Event type (started, canary, complete, rollback).
            endpoint_name: Name of the endpoint being deployed.
            details: Optional additional event details.

        Returns:
            SNS message ID.
        """
        import json

        severity_map = {
            "started": "INFO",
            "canary": "INFO",
            "complete": "INFO",
            "rollback": "CRITICAL",
        }
        severity = severity_map.get(event_type, "WARNING")
        subject = f"Deployment {event_type}: {endpoint_name}"
        payload: dict[str, Any] = {
            "event_type": event_type,
            "endpoint_name": endpoint_name,
        }
        if details:
            payload["details"] = details
        message = json.dumps(payload, indent=2, default=str)
        return self.send_alert(subject, message, severity=severity)

    def send_cost_alert(
        self,
        current_cost: float,
        budget_limit: float,
        details: dict[str, Any] | None = None,
    ) -> str:
        """Send cost threshold alert.

        Args:
            current_cost: Current accumulated cost in USD.
            budget_limit: Budget limit in USD.
            details: Optional additional cost details.

        Returns:
            SNS message ID.
        """
        import json

        pct = (current_cost / budget_limit * 100) if budget_limit > 0 else 0.0
        severity = "CRITICAL" if pct >= 100 else "WARNING"
        subject = f"Cost Alert: {pct:.0f}% of budget (${current_cost:.2f}/${budget_limit:.2f})"
        payload: dict[str, Any] = {
            "current_cost_usd": current_cost,
            "budget_limit_usd": budget_limit,
            "utilization_pct": round(pct, 2),
        }
        if details:
            payload["details"] = details
        message = json.dumps(payload, indent=2, default=str)
        return self.send_alert(subject, message, severity=severity)


__all__: list[str] = ["AlertManager"]
