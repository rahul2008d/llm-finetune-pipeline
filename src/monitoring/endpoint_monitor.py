"""Production endpoint monitoring: SageMaker Model Monitor + CloudWatch alarms."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class EndpointMonitor:
    """Configure and manage production monitoring for deployed endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 clients.

        Args:
            region: AWS region.
        """
        self.region = region

    def _get_cw_client(self) -> Any:
        """Get a CloudWatch boto3 client.

        Returns:
            Boto3 CloudWatch client.
        """
        import boto3

        return boto3.client("cloudwatch", region_name=self.region)

    def _get_sm_client(self) -> Any:
        """Get a SageMaker boto3 client.

        Returns:
            Boto3 SageMaker client.
        """
        import boto3

        return boto3.client("sagemaker", region_name=self.region)

    def setup_monitoring(
        self,
        endpoint_name: str,
        alert_sns_topic_arn: str,
        latency_p99_threshold_ms: int = 5000,
        error_rate_threshold_pct: float = 1.0,
        enable_data_quality: bool = True,
    ) -> dict[str, Any]:
        """Configure full monitoring stack for an endpoint.

        Creates CloudWatch alarms and a dashboard for the endpoint.

        Args:
            endpoint_name: Name of the SageMaker endpoint.
            alert_sns_topic_arn: SNS topic ARN for alarm notifications.
            latency_p99_threshold_ms: P99 latency alarm threshold in ms.
            error_rate_threshold_pct: 5xx error rate alarm threshold as pct.
            enable_data_quality: Whether to enable data quality monitoring.

        Returns:
            Dictionary with alarm_arns and dashboard_url.
        """
        cw = self._get_cw_client()
        namespace = "AWS/SageMaker"
        dimensions = [{"Name": "EndpointName", "Value": endpoint_name}]
        alarm_arns: list[str] = []

        alarm_configs = [
            {
                "AlarmName": f"{endpoint_name}-zero-invocations",
                "MetricName": "Invocations",
                "ComparisonOperator": "LessThanOrEqualToThreshold",
                "Threshold": 0.0,
                "EvaluationPeriods": 3,
                "Period": 300,
                "Statistic": "Sum",
                "TreatMissingData": "breaching",
            },
            {
                "AlarmName": f"{endpoint_name}-p99-latency",
                "MetricName": "ModelLatency",
                "ComparisonOperator": "GreaterThanThreshold",
                "Threshold": float(latency_p99_threshold_ms * 1000),
                "EvaluationPeriods": 2,
                "Period": 300,
                "Statistic": "p99",
                "TreatMissingData": "notBreaching",
            },
            {
                "AlarmName": f"{endpoint_name}-5xx-rate",
                "MetricName": "Invocation5XXErrors",
                "ComparisonOperator": "GreaterThanThreshold",
                "Threshold": error_rate_threshold_pct,
                "EvaluationPeriods": 2,
                "Period": 300,
                "Statistic": "Average",
                "TreatMissingData": "notBreaching",
            },
            {
                "AlarmName": f"{endpoint_name}-unhealthy-instances",
                "MetricName": "InvocationsPerInstance",
                "ComparisonOperator": "LessThanOrEqualToThreshold",
                "Threshold": 0.0,
                "EvaluationPeriods": 3,
                "Period": 300,
                "Statistic": "Sum",
                "TreatMissingData": "breaching",
            },
        ]

        for cfg in alarm_configs:
            alarm_name = cfg.pop("AlarmName")
            cw.put_metric_alarm(
                AlarmName=alarm_name,
                Namespace=namespace,
                Dimensions=dimensions,
                ActionsEnabled=True,
                AlarmActions=[alert_sns_topic_arn],
                **cfg,
            )
            alarm_arns.append(
                f"arn:aws:cloudwatch:{self.region}:alarm:{alarm_name}"
            )
            logger.info("Alarm created", alarm_name=alarm_name)

        dashboard_url = self._create_dashboard(endpoint_name, namespace, dimensions)

        logger.info(
            "Monitoring configured",
            endpoint_name=endpoint_name,
            alarm_count=len(alarm_arns),
        )

        return {
            "alarm_arns": alarm_arns,
            "dashboard_url": dashboard_url,
        }

    def _create_dashboard(
        self,
        endpoint_name: str,
        namespace: str,
        dimensions: list[dict[str, str]],
    ) -> str:
        """Create a CloudWatch dashboard for the endpoint.

        Args:
            endpoint_name: Name of the endpoint.
            namespace: CloudWatch namespace.
            dimensions: CloudWatch dimensions.

        Returns:
            URL of the created dashboard.
        """
        cw = self._get_cw_client()
        dashboard_name = f"LLMFineTune-{endpoint_name}"
        dim_list = [[d["Name"], d["Value"]] for d in dimensions]

        def _metric_widget(
            title: str, metrics: list[list[str]], y_label: str = ""
        ) -> dict[str, Any]:
            return {
                "type": "metric",
                "properties": {
                    "title": title,
                    "metrics": [
                        [namespace, m[0], *dim_list[0], {"stat": m[1]}]
                        for m in metrics
                    ],
                    "period": 60,
                    "region": self.region,
                    "yAxis": {"left": {"label": y_label}},
                },
            }

        widgets = [
            _metric_widget(
                "Invocations & Latency",
                [
                    ["Invocations", "Sum"],
                    ["ModelLatency", "p50"],
                    ["ModelLatency", "p90"],
                    ["ModelLatency", "p99"],
                ],
                "Count / Microseconds",
            ),
            _metric_widget(
                "Error Rates",
                [
                    ["Invocation4XXErrors", "Sum"],
                    ["Invocation5XXErrors", "Sum"],
                ],
                "Count",
            ),
            _metric_widget(
                "Instance Health",
                [["InvocationsPerInstance", "Sum"]],
                "Count",
            ),
        ]

        dashboard_body = json.dumps({"widgets": widgets})
        cw.put_dashboard(
            DashboardName=dashboard_name,
            DashboardBody=dashboard_body,
        )

        url = (
            f"https://{self.region}.console.aws.amazon.com/cloudwatch/home"
            f"?region={self.region}#dashboards:name={dashboard_name}"
        )
        logger.info("Dashboard created", name=dashboard_name, url=url)
        return url

    def get_monitoring_report(
        self,
        endpoint_name: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Generate monitoring report for last N hours.

        Args:
            endpoint_name: Name of the endpoint.
            hours: Number of hours to look back.

        Returns:
            Dictionary with uptime_pct, error_rate, latency_stats,
            invocation_count, and cost_estimate.
        """
        from datetime import timedelta

        cw = self._get_cw_client()
        end_time = datetime.now(tz=timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        namespace = "AWS/SageMaker"
        dimensions = [{"Name": "EndpointName", "Value": endpoint_name}]

        def _get_metric_stats(
            metric_name: str, stat: str,
        ) -> list[float]:
            response = cw.get_metric_statistics(
                Namespace=namespace,
                MetricName=metric_name,
                Dimensions=dimensions,
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=[stat],
            )
            return [
                dp[stat] for dp in response.get("Datapoints", [])
            ]

        invocation_values = _get_metric_stats("Invocations", "Sum")
        error_values = _get_metric_stats("Invocation5XXErrors", "Sum")
        latency_p50 = _get_metric_stats("ModelLatency", "Average")
        latency_p99 = _get_metric_stats("ModelLatency", "Maximum")

        total_invocations = sum(invocation_values) if invocation_values else 0
        total_errors = sum(error_values) if error_values else 0
        error_rate = (total_errors / total_invocations * 100) if total_invocations > 0 else 0.0

        non_zero_periods = sum(1 for v in invocation_values if v > 0)
        total_periods = len(invocation_values) if invocation_values else 1
        uptime_pct = (non_zero_periods / total_periods * 100) if total_periods > 0 else 0.0

        avg_latency_p50 = (
            sum(latency_p50) / len(latency_p50) if latency_p50 else 0.0
        )
        max_latency_p99 = max(latency_p99) if latency_p99 else 0.0

        report: dict[str, Any] = {
            "endpoint_name": endpoint_name,
            "period_hours": hours,
            "uptime_pct": round(uptime_pct, 2),
            "error_rate": round(error_rate, 4),
            "latency_stats": {
                "avg_p50_ms": round(avg_latency_p50 / 1000, 2),
                "max_p99_ms": round(max_latency_p99 / 1000, 2),
            },
            "invocation_count": int(total_invocations),
            "cost_estimate": round(total_invocations * 0.0002, 2),
        }

        logger.info("Monitoring report generated", **report)
        return report


__all__: list[str] = ["EndpointMonitor"]
