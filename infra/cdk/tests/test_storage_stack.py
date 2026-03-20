"""CDK assertion tests for the Storage stack."""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack


def _make_template(*, env_name: str = "dev") -> Template:
    """Synthesize a StorageStack with the given env and return its Template."""
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
    net_stack = NetworkStack(
        app,
        f"test-net-{env_name}",
        config=config,
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    storage_stack = StorageStack(
        app,
        f"test-storage-{env_name}",
        config=config,
        vpc=net_stack.vpc_construct,
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    return Template.from_stack(storage_stack)


@pytest.fixture
def dev_template() -> Template:
    return _make_template(env_name="dev")


@pytest.fixture
def prod_template() -> Template:
    return _make_template(env_name="prod")


# ---- KMS ----

def test_kms_key_rotation_enabled(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::KMS::Key", {"EnableKeyRotation": True}
    )


# ---- S3 Buckets ----

def test_three_s3_buckets_exist(dev_template: Template) -> None:
    dev_template.resource_count_is("AWS::S3::Bucket", 3)


def test_all_buckets_kms_encrypted(dev_template: Template) -> None:
    resources = dev_template.find_resources("AWS::S3::Bucket")
    for logical_id, resource in resources.items():
        props = resource["Properties"]
        enc = props.get("BucketEncryption", {})
        rules = enc.get("ServerSideEncryptionConfiguration", [])
        assert len(rules) > 0, f"{logical_id} missing encryption config"
        for rule in rules:
            sse = rule.get("ServerSideEncryptionByDefault", {})
            assert sse.get("SSEAlgorithm") == "aws:kms", (
                f"{logical_id} not using aws:kms encryption"
            )


def test_all_buckets_block_public_access(dev_template: Template) -> None:
    resources = dev_template.find_resources("AWS::S3::Bucket")
    for logical_id, resource in resources.items():
        pa = resource["Properties"].get("PublicAccessBlockConfiguration", {})
        assert pa.get("BlockPublicAcls") is True, f"{logical_id} missing BlockPublicAcls"
        assert pa.get("BlockPublicPolicy") is True, f"{logical_id} missing BlockPublicPolicy"
        assert pa.get("IgnorePublicAcls") is True, f"{logical_id} missing IgnorePublicAcls"
        assert pa.get("RestrictPublicBuckets") is True, f"{logical_id} missing RestrictPublicBuckets"


def test_training_data_has_lifecycle(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::S3::Bucket",
        Match.object_like({
            "LifecycleConfiguration": Match.object_like({
                "Rules": Match.any_value(),
            }),
        }),
    )


# ---- ECR ----

def test_ecr_repos_immutable_tags(dev_template: Template) -> None:
    dev_template.has_resource_properties(
        "AWS::ECR::Repository",
        {
            "ImageTagMutability": "IMMUTABLE",
            "ImageScanningConfiguration": {"ScanOnPush": True},
        },
    )


def test_ecr_repos_encrypted(dev_template: Template) -> None:
    resources = dev_template.find_resources("AWS::ECR::Repository")
    for logical_id, resource in resources.items():
        enc = resource["Properties"].get("EncryptionConfiguration", {})
        assert enc.get("EncryptionType") == "KMS", (
            f"{logical_id} not using KMS encryption"
        )


# ---- Env-specific deletion policies ----

def test_prod_buckets_retain_on_delete(prod_template: Template) -> None:
    resources = prod_template.find_resources("AWS::S3::Bucket")
    for logical_id, resource in resources.items():
        assert resource.get("DeletionPolicy") == "Retain", (
            f"Prod bucket {logical_id} should have DeletionPolicy=Retain"
        )


def test_dev_buckets_delete_on_destroy(dev_template: Template) -> None:
    resources = dev_template.find_resources("AWS::S3::Bucket")
    for logical_id, resource in resources.items():
        assert resource.get("DeletionPolicy") == "Delete", (
            f"Dev bucket {logical_id} should have DeletionPolicy=Delete"
        )
