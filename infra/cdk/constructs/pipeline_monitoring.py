"""CloudWatch alarms, dashboard, and SNS alerting for the ML pipeline."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_cloudwatch as cw,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as subs,
)
from constructs import Construct


class PipelineMonitoring(Construct):
    """Centralized monitoring: SNS topic, alarms, and dashboard."""

    def __init__(
        self, scope: Construct, id: str, *,
        project_name: str, env_name: str, alert_email: str,
    ) -> None:
        super().__init__(scope, id)

        # SNS Topic
        self.alert_topic = sns.Topic(
            self, "AlertTopic",
            topic_name=f"{project_name}-{env_name}-pipeline-alerts",
            display_name=f"LLM Fine-Tuning Alerts ({env_name})",
        )
        self.alert_topic.add_subscription(subs.EmailSubscription(alert_email))

        # ---- Training Job Failure Alarm ----
        training_failures = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="TrainingJobsFailed",
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )
        self.training_failure_alarm = cw.Alarm(
            self, "TrainingJobFailure",
            alarm_name=f"{project_name}-{env_name}-training-job-failure",
            metric=training_failures,
            threshold=1,
            evaluation_periods=1,
            comparison_operator=cw.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            alarm_description="A SageMaker training job has failed",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.training_failure_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # ---- Endpoint 5xx Error Rate Alarm ----
        invocations = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="Invocations",
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )
        errors_5xx = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="Invocation5XXErrors",
            statistic="Sum",
            period=cdk.Duration.minutes(5),
        )
        error_rate = cw.MathExpression(
            expression="IF(invocations > 0, (errors / invocations) * 100, 0)",
            using_metrics={"errors": errors_5xx, "invocations": invocations},
            period=cdk.Duration.minutes(5),
        )
        self.error_rate_alarm = cw.Alarm(
            self, "Endpoint5xxRate",
            alarm_name=f"{project_name}-{env_name}-endpoint-5xx-rate",
            metric=error_rate,
            threshold=1,
            evaluation_periods=3,
            alarm_description="Endpoint 5xx error rate exceeds 1% for 15 minutes",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.error_rate_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # ---- Endpoint P99 Latency Alarm ----
        latency_p99 = cw.Metric(
            namespace="AWS/SageMaker",
            metric_name="ModelLatency",
            statistic="p99",
            period=cdk.Duration.minutes(5),
        )
        self.latency_alarm = cw.Alarm(
            self, "EndpointLatencyP99",
            alarm_name=f"{project_name}-{env_name}-endpoint-latency-p99",
            metric=latency_p99,
            threshold=10_000_000,  # 10 seconds in microseconds
            evaluation_periods=3,
            alarm_description="Endpoint P99 latency exceeds 10s for 15 minutes",
            treat_missing_data=cw.TreatMissingData.NOT_BREACHING,
        )
        self.latency_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))

        # ---- Dashboard ----
        self.dashboard = cw.Dashboard(
            self, "Dashboard",
            dashboard_name=f"{project_name}-{env_name}-overview",
            default_interval=cdk.Duration.hours(6),
        )
        self.dashboard.add_widgets(
            cw.GraphWidget(
                title="Training Job Failures",
                left=[training_failures],
                width=12,
            ),
            cw.GraphWidget(
                title="Endpoint Error Rate (%)",
                left=[error_rate],
                width=12,
            ),
            cw.GraphWidget(
                title="Endpoint Latency (P50/P90/P99)",
                left=[
                    cw.Metric(namespace="AWS/SageMaker", metric_name="ModelLatency",
                              statistic="p50", period=cdk.Duration.minutes(5)),
                    cw.Metric(namespace="AWS/SageMaker", metric_name="ModelLatency",
                              statistic="p90", period=cdk.Duration.minutes(5)),
                    latency_p99,
                ],
                width=12,
            ),
            cw.GraphWidget(
                title="Invocations",
                left=[invocations],
                width=12,
            ),
        )
