"""SageMaker domain, budget alarms, CloudTrail, and EventBridge rules."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_sagemaker as sagemaker,
    aws_budgets as budgets,
    aws_cloudtrail as cloudtrail,
    aws_logs as logs,
    aws_events as events,
    aws_events_targets as targets,
    aws_ec2 as ec2,
)
from constructs import Construct

from constructs.training_vpc import TrainingVpc
from constructs.pipeline_monitoring import PipelineMonitoring
from constructs.secure_bucket import SecureBucket, SecureBucketProps
from stacks.storage_stack import StorageStack
from stacks.iam_stack import IamStack
from config.environments import EnvironmentConfig
from config.constants import PROJECT_NAME


class SageMakerStack(cdk.Stack):
    """SageMaker domain with monitoring and governance controls."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: EnvironmentConfig,
        vpc_construct: TrainingVpc,
        storage_stack: StorageStack,
        iam_stack: IamStack,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- Monitoring (SNS + Alarms + Dashboard) ----
        self.monitoring = PipelineMonitoring(
            self, "Monitoring",
            project_name=PROJECT_NAME,
            env_name=config.env_name,
            alert_email=config.alert_email,
        )

        # ---- SageMaker Domain (VPC-only) ----
        private_subnet_ids = [
            s.subnet_id for s in vpc_construct.vpc.select_subnets(
                subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
            ).subnets
        ]
        self.domain = sagemaker.CfnDomain(
            self, "Domain",
            auth_mode="IAM",
            domain_name=f"{PROJECT_NAME}-{config.env_name}",
            vpc_id=vpc_construct.vpc.vpc_id,
            subnet_ids=private_subnet_ids,
            app_network_access_type="VpcOnly",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=iam_stack.pipeline_role.role_arn,
                security_groups=[vpc_construct.training_sg.security_group_id],
            ),
        )

        # ---- AWS Budget ----
        budgets.CfnBudget(
            self, "MonthlyBudget",
            budget=budgets.CfnBudget.BudgetDataProperty(
                budget_name=f"{PROJECT_NAME}-{config.env_name}-monthly",
                budget_limit=budgets.CfnBudget.SpendProperty(
                    amount=config.budget_limit_usd, unit="USD",
                ),
                time_unit="MONTHLY",
                budget_type="COST",
                cost_filters={"TagKeyValue": [f"Project${PROJECT_NAME}"]},
            ),
            notifications_with_subscribers=[
                budgets.CfnBudget.NotificationWithSubscribersProperty(
                    notification=budgets.CfnBudget.NotificationProperty(
                        comparison_operator="GREATER_THAN",
                        notification_type="ACTUAL",
                        threshold=threshold,
                        threshold_type="PERCENTAGE",
                    ),
                    subscribers=[
                        budgets.CfnBudget.SubscriberProperty(
                            subscription_type="SNS",
                            address=self.monitoring.alert_topic.topic_arn,
                        ),
                    ],
                )
                for threshold in [50, 80, 100]
            ],
        )

        # ---- CloudTrail ----
        trail_bucket = SecureBucket(
            self, "CloudTrailBucket",
            props=SecureBucketProps(
                bucket_name_suffix="cloudtrail",
                encryption_key=storage_stack.kms_key,
                env_name=config.env_name,
            ),
        )
        retention_map = {90: "THREE_MONTHS", 365: "ONE_YEAR"}
        retention_name = retention_map.get(
            config.cloudtrail_retention_days, "THREE_MONTHS"
        )
        trail_log_group = logs.LogGroup(
            self, "TrailLogGroup",
            retention=getattr(logs.RetentionDays, retention_name),
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        cloudtrail.Trail(
            self, "ApiTrail",
            trail_name=f"{PROJECT_NAME}-{config.env_name}-api-trail",
            bucket=trail_bucket.bucket,
            send_to_cloud_watch_logs=True,
            cloud_watch_log_group=trail_log_group,
            is_multi_region_trail=False,
            include_global_service_events=False,
            management_events=cloudtrail.ReadWriteType.ALL,
        )

        # ---- EventBridge Rules ----
        events.Rule(
            self, "TrainingJobStateChange",
            rule_name=f"{PROJECT_NAME}-{config.env_name}-training-state-change",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Training Job State Change"],
                detail={
                    "TrainingJobStatus": ["Completed", "Failed", "Stopped"],
                },
            ),
            targets=[targets.SnsTopic(self.monitoring.alert_topic)],
        )

        events.Rule(
            self, "EndpointStatusChange",
            rule_name=f"{PROJECT_NAME}-{config.env_name}-endpoint-state-change",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Endpoint State Change"],
            ),
            targets=[targets.SnsTopic(self.monitoring.alert_topic)],
        )

        # ---- Outputs ----
        cdk.CfnOutput(self, "DomainId", value=self.domain.attr_domain_id)
        cdk.CfnOutput(self, "SNSTopicArn", value=self.monitoring.alert_topic.topic_arn)
        cdk.CfnOutput(self, "DashboardUrl",
            value=f"https://{config.region}.console.aws.amazon.com/cloudwatch/"
                  f"home?region={config.region}#dashboards:name="
                  f"{PROJECT_NAME}-{config.env_name}-overview")
