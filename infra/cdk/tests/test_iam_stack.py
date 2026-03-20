"""CDK assertion tests for the IAM stack."""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.iam_stack import IamStack


def _make_template(*, env_name: str = "dev") -> Template:
    """Synthesize an IamStack with cross-stack dependencies and return its Template."""
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
    return Template.from_stack(iam_stack)


@pytest.fixture
def dev_template() -> Template:
    return _make_template(env_name="dev")


def test_four_roles_created(dev_template: Template) -> None:
    """4 project roles + CDK-internal roles may exist; at least 4 must be present."""
    resources = dev_template.find_resources("AWS::IAM::Role")
    project_roles = {
        lid: r for lid, r in resources.items()
        if "Custom" not in lid and "LogRetention" not in lid
    }
    assert len(project_roles) == 4, (
        f"Expected 4 project roles, found {len(project_roles)}: {list(project_roles)}"
    )


def test_training_role_trust_policy(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::IAM::Role",
        {
            "RoleName": Match.string_like_regexp(".*sagemaker-training"),
            "AssumeRolePolicyDocument": Match.object_like({
                "Statement": Match.array_with([
                    Match.object_like({
                        "Principal": {"Service": "sagemaker.amazonaws.com"},
                    }),
                ]),
            }),
        },
    )


def test_endpoint_role_no_s3_write(dev_template: Template) -> None:
    """Endpoint role should only have S3 read, not write actions."""
    policies = dev_template.find_resources("AWS::IAM::Policy")
    write_actions = {"s3:PutObject", "s3:DeleteObject", "s3:PutObject*", "s3:DeleteObject*"}
    for lid, policy in policies.items():
        # Only check policies attached to the endpoint role
        if "Endpoint" not in lid:
            continue
        doc = policy["Properties"]["PolicyDocument"]
        for stmt in doc["Statement"]:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            for a in actions:
                assert a not in write_actions, (
                    f"Endpoint role has write action {a} in {lid}"
                )


def test_bedrock_role_no_s3_write(dev_template: Template) -> None:
    """Bedrock import role should only have S3 read, not write actions."""
    policies = dev_template.find_resources("AWS::IAM::Policy")
    write_actions = {"s3:PutObject", "s3:DeleteObject", "s3:PutObject*", "s3:DeleteObject*"}
    for lid, policy in policies.items():
        if "Bedrock" not in lid:
            continue
        doc = policy["Properties"]["PolicyDocument"]
        for stmt in doc["Statement"]:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            for a in actions:
                assert a not in write_actions, (
                    f"Bedrock role has write action {a} in {lid}"
                )


def test_pipeline_role_passrole_restricted(dev_template: Template) -> None:
    """iam:PassRole should be limited to specific role ARNs, not *."""
    policies = dev_template.find_resources("AWS::IAM::Policy")
    for lid, policy in policies.items():
        if "Pipeline" not in lid:
            continue
        doc = policy["Properties"]["PolicyDocument"]
        for stmt in doc["Statement"]:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            if "iam:PassRole" in actions:
                resources = stmt.get("Resource", [])
                if isinstance(resources, str):
                    resources = [resources]
                for r in resources:
                    assert r != "*", "iam:PassRole must not have Resource=*"


def test_all_roles_have_permission_boundary(dev_template: Template) -> None:
    roles = dev_template.find_resources("AWS::IAM::Role")
    for logical_id, role in roles.items():
        # Skip CDK internal roles
        if "Custom" in logical_id or "LogRetention" in logical_id:
            continue
        props = role.get("Properties", {})
        assert "PermissionsBoundary" in props, (
            f"Role {logical_id} missing PermissionsBoundary"
        )


def test_no_iam_star_actions(dev_template: Template) -> None:
    policies = dev_template.find_resources("AWS::IAM::Policy")
    for lid, p in policies.items():
        doc = p["Properties"]["PolicyDocument"]
        for stmt in doc["Statement"]:
            actions = stmt.get("Action", [])
            if isinstance(actions, str):
                actions = [actions]
            for a in actions:
                assert a != "iam:*", f"Wildcard iam:* action found in {lid}"
                assert a != "sts:*", f"Wildcard sts:* action found in {lid}"
