"""CDK assertion tests for the SageMaker stack."""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.iam_stack import IamStack
from stacks.sagemaker_stack import SageMakerStack


def _make_template(*, env_name: str = "dev") -> Template:
    """Synthesize a full app and return the SageMaker stack template."""
    app = cdk.App()
    is_prod = env_name == "prod"
    config = EnvironmentConfig(
        env_name=env_name,
        account="111111111111",
        region="us-east-1",
        vpc_cidr="10.0.0.0/16",
        nat_gateways=2 if is_prod else 1,
        enable_bedrock=False,
        budget_limit_usd=500 if is_prod else 200,
        alert_email="test@test.com",
        cloudtrail_retention_days=365 if is_prod else 90,
    )
    env = cdk.Environment(account="111111111111", region="us-east-1")

    net = NetworkStack(app, f"test-net-{env_name}", config=config, env=env)
    storage = StorageStack(
        app, f"test-storage-{env_name}", config=config,
        vpc=net.vpc_construct, env=env,
    )
    iam_stack = IamStack(
        app, f"test-iam-{env_name}", config=config,
        training_data_bucket=storage.training_data_bucket.bucket,
        model_artifacts_bucket=storage.model_artifacts_bucket.bucket,
        mlflow_bucket=storage.mlflow_bucket.bucket,
        kms_key=storage.kms_key,
        training_ecr_repo=storage.training_ecr_repo,
        serving_ecr_repo=storage.serving_ecr_repo,
        vpc=net.vpc_construct.vpc,
        env=env,
    )
    sm_stack = SageMakerStack(
        app, f"test-sagemaker-{env_name}", config=config,
        vpc_construct=net.vpc_construct,
        storage_stack=storage,
        iam_stack=iam_stack,
        env=env,
    )
    return Template.from_stack(sm_stack)


@pytest.fixture
def dev_template() -> Template:
    return _make_template(env_name="dev")


def test_domain_vpc_only_mode(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::SageMaker::Domain",
        {"AppNetworkAccessType": "VpcOnly"},
    )


def test_budget_three_thresholds(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::Budgets::Budget",
        Match.object_like({
            "NotificationsWithSubscribers": Match.array_with([
                Match.object_like({"Notification": {"Threshold": 50}}),
                Match.object_like({"Notification": {"Threshold": 80}}),
                Match.object_like({"Notification": {"Threshold": 100}}),
            ]),
        }),
    )


def test_three_cloudwatch_alarms(dev_template: Template) -> None:
    dev_template.resource_count_is("AWS::CloudWatch::Alarm", 3)


def test_cloudtrail_with_cloudwatch(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::CloudTrail::Trail",
        {"IsLogging": True},
    )


def test_two_eventbridge_rules(dev_template: Template) -> None:
    dev_template.resource_count_is("AWS::Events::Rule", 2)


def test_all_alarms_route_to_sns(dev_template: Template) -> None:
    alarms = dev_template.find_resources("AWS::CloudWatch::Alarm")
    for logical_id, alarm in alarms.items():
        props = alarm["Properties"]
        assert "AlarmActions" in props, f"{logical_id} missing AlarmActions"
        assert len(props["AlarmActions"]) > 0, f"{logical_id} has empty AlarmActions"
