"""Shared pytest fixtures for CDK infrastructure tests."""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.iam_stack import IamStack
from stacks.sagemaker_stack import SageMakerStack


def _make_config(env_name: str = "dev", **overrides) -> EnvironmentConfig:
    """Create an EnvironmentConfig with test-friendly defaults."""
    defaults = {
        "dev": dict(
            env_name="dev", account="111111111111", region="us-east-1",
            vpc_cidr="10.0.0.0/16", nat_gateways=1, enable_bedrock=True,
            budget_limit_usd=200, alert_email="dev@test.com",
        ),
        "prod": dict(
            env_name="prod", account="333333333333", region="us-east-1",
            vpc_cidr="10.2.0.0/16", nat_gateways=2, enable_bedrock=True,
            budget_limit_usd=2000, alert_email="prod@test.com",
            removal_policy_retain=True, flow_log_retention_days=90,
            cloudtrail_retention_days=365,
        ),
    }
    params = {**defaults.get(env_name, defaults["dev"]), **overrides}
    return EnvironmentConfig(**params)


@pytest.fixture(scope="session")
def dev_config() -> EnvironmentConfig:
    return _make_config("dev")


@pytest.fixture(scope="session")
def prod_config() -> EnvironmentConfig:
    return _make_config("prod")


@pytest.fixture(scope="session")
def dev_network_template(dev_config) -> Template:
    app = cdk.App()
    stack = NetworkStack(app, "test-net", config=dev_config,
        env=cdk.Environment(account=dev_config.account, region=dev_config.region))
    return Template.from_stack(stack)


@pytest.fixture(scope="session")
def dev_storage_template(dev_config) -> Template:
    app = cdk.App()
    env = cdk.Environment(account=dev_config.account, region=dev_config.region)
    net = NetworkStack(app, "test-net-s", config=dev_config, env=env)
    stack = StorageStack(app, "test-storage-s", config=dev_config,
        vpc=net.vpc_construct, env=env)
    return Template.from_stack(stack)


@pytest.fixture(scope="session")
def dev_iam_template(dev_config) -> Template:
    app = cdk.App()
    env = cdk.Environment(account=dev_config.account, region=dev_config.region)
    net = NetworkStack(app, "test-net-i", config=dev_config, env=env)
    storage = StorageStack(app, "test-storage-i", config=dev_config,
        vpc=net.vpc_construct, env=env)
    stack = IamStack(app, "test-iam-i", config=dev_config,
        training_data_bucket=storage.training_data_bucket.bucket,
        model_artifacts_bucket=storage.model_artifacts_bucket.bucket,
        mlflow_bucket=storage.mlflow_bucket.bucket,
        kms_key=storage.kms_key,
        training_ecr_repo=storage.training_ecr_repo,
        serving_ecr_repo=storage.serving_ecr_repo,
        vpc=net.vpc_construct.vpc,
        env=env)
    return Template.from_stack(stack)


@pytest.fixture(scope="session")
def dev_full_app() -> dict[str, Template]:
    """Synthesize the full 4-stack app and return all templates keyed by name."""
    config = _make_config("dev")
    app = cdk.App()
    env = cdk.Environment(account=config.account, region=config.region)
    net = NetworkStack(app, "test-net-full", config=config, env=env)
    storage = StorageStack(app, "test-storage-full", config=config,
        vpc=net.vpc_construct, env=env)
    iam_stack = IamStack(app, "test-iam-full", config=config,
        training_data_bucket=storage.training_data_bucket.bucket,
        model_artifacts_bucket=storage.model_artifacts_bucket.bucket,
        mlflow_bucket=storage.mlflow_bucket.bucket,
        kms_key=storage.kms_key,
        training_ecr_repo=storage.training_ecr_repo,
        serving_ecr_repo=storage.serving_ecr_repo,
        vpc=net.vpc_construct.vpc,
        env=env)
    sm = SageMakerStack(app, "test-sm-full", config=config,
        vpc_construct=net.vpc_construct,
        storage_stack=storage,
        iam_stack=iam_stack,
        env=env)
    return {
        name: Template.from_stack(stack)
        for name, stack in [("network", net), ("storage", storage),
                            ("iam", iam_stack), ("sagemaker", sm)]
    }
