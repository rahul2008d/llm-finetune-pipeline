"""Cross-stack security invariant tests.

Validates security properties across ALL stacks simultaneously.
Synthesizes the complete app and inspects the generated CloudFormation.
These tests MUST pass in CI — they are the security gate.
"""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack
from stacks.storage_stack import StorageStack
from stacks.iam_stack import IamStack
from stacks.sagemaker_stack import SageMakerStack


@pytest.fixture(scope="module")
def all_templates() -> dict[str, Template]:
    """Synthesize full app and return templates for all stacks."""
    app = cdk.App()
    config = EnvironmentConfig(
        env_name="prod", account="333333333333", region="us-east-1",
        vpc_cidr="10.2.0.0/16", nat_gateways=2, enable_bedrock=True,
        budget_limit_usd=2000, alert_email="prod@test.com",
        removal_policy_retain=True,
    )
    env = cdk.Environment(account="333333333333", region="us-east-1")
    network = NetworkStack(app, "net", config=config, env=env)
    storage = StorageStack(app, "store", config=config, vpc=network.vpc_construct, env=env)
    iam = IamStack(
        app, "iam", config=config,
        training_data_bucket=storage.training_data_bucket.bucket,
        model_artifacts_bucket=storage.model_artifacts_bucket.bucket,
        mlflow_bucket=storage.mlflow_bucket.bucket,
        kms_key=storage.kms_key,
        training_ecr_repo=storage.training_ecr_repo,
        serving_ecr_repo=storage.serving_ecr_repo,
        vpc=network.vpc_construct.vpc,
        env=env,
    )
    sm = SageMakerStack(
        app, "sm", config=config,
        vpc_construct=network.vpc_construct,
        storage_stack=storage,
        iam_stack=iam,
        env=env,
    )
    return {
        name: Template.from_stack(stack)
        for name, stack in [("network", network), ("storage", storage),
                            ("iam", iam), ("sagemaker", sm)]
    }


class TestEncryptionInvariants:
    """Every S3 bucket and ECR repo must be encrypted with KMS."""

    def test_all_s3_buckets_kms_encrypted(self, all_templates):
        for name, tpl in all_templates.items():
            buckets = tpl.find_resources("AWS::S3::Bucket")
            for lid, bucket in buckets.items():
                enc = bucket["Properties"].get("BucketEncryption", {})
                rules = enc.get("ServerSideEncryptionConfiguration", [])
                assert len(rules) > 0, f"Bucket {lid} in {name} has no encryption"
                algo = rules[0].get("ServerSideEncryptionByDefault", {}).get("SSEAlgorithm")
                assert algo == "aws:kms", f"Bucket {lid} in {name}: expected aws:kms, got {algo}"

    def test_all_s3_buckets_block_public_access(self, all_templates):
        for name, tpl in all_templates.items():
            buckets = tpl.find_resources("AWS::S3::Bucket")
            for lid, bucket in buckets.items():
                pab = bucket["Properties"].get("PublicAccessBlockConfiguration", {})
                for flag in ["BlockPublicAcls", "BlockPublicPolicy",
                             "IgnorePublicAcls", "RestrictPublicBuckets"]:
                    assert pab.get(flag) is True, f"Bucket {lid} in {name}: {flag} not True"

    def test_kms_key_rotation(self, all_templates):
        tpl = all_templates["storage"]
        tpl.has_resource_properties("AWS::KMS::Key", {"EnableKeyRotation": True})

    def test_ecr_repos_encrypted(self, all_templates):
        tpl = all_templates["storage"]
        repos = tpl.find_resources("AWS::ECR::Repository")
        for lid, repo in repos.items():
            enc = repo["Properties"].get("EncryptionConfiguration", {})
            assert enc.get("EncryptionType") == "KMS", f"ECR {lid} not KMS encrypted"


class TestIamInvariants:
    """No wildcard IAM, all roles bounded."""

    def test_no_iam_star_actions(self, all_templates):
        """No policy should grant iam:* or sts:* actions."""
        tpl = all_templates["iam"]
        policies = tpl.find_resources("AWS::IAM::Policy")
        for lid, pol in policies.items():
            doc = pol["Properties"]["PolicyDocument"]
            for stmt in doc["Statement"]:
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                for a in actions:
                    assert a not in ("iam:*", "sts:*", "kms:*", "s3:*"), \
                        f"Wildcard action {a} found in policy {lid}"

    def test_all_project_roles_have_boundary(self, all_templates):
        tpl = all_templates["iam"]
        roles = tpl.find_resources("AWS::IAM::Role")
        for lid, role in roles.items():
            # Skip CDK auto-generated roles
            if any(skip in lid for skip in ["Custom", "LogRetention", "AWS"]):
                continue
            props = role.get("Properties", {})
            assert "PermissionsBoundary" in props, \
                f"Role {lid} missing PermissionsBoundary"

    def test_endpoint_role_read_only_s3(self, all_templates):
        """Endpoint role must not have any S3 write actions."""
        tpl = all_templates["iam"]
        policies = tpl.find_resources("AWS::IAM::Policy")
        write_actions = {"s3:PutObject", "s3:DeleteObject", "s3:PutObjectAcl"}
        for lid, pol in policies.items():
            if "Endpoint" not in lid:
                continue
            doc = pol["Properties"]["PolicyDocument"]
            for stmt in doc["Statement"]:
                actions = stmt.get("Action", [])
                if isinstance(actions, str):
                    actions = [actions]
                for a in actions:
                    assert a not in write_actions, \
                        f"Endpoint role has write action {a} in {lid}"


class TestNetworkInvariants:
    """VPC endpoints, flow logs, no public ingress."""

    def test_vpc_endpoints_private_dns(self, all_templates):
        tpl = all_templates["network"]
        endpoints = tpl.find_resources("AWS::EC2::VPCEndpoint")
        for lid, ep in endpoints.items():
            props = ep.get("Properties", {})
            if props.get("VpcEndpointType") == "Gateway":
                continue
            assert props.get("PrivateDnsEnabled") is True, \
                f"Endpoint {lid} missing PrivateDnsEnabled"

    def test_flow_logs_enabled(self, all_templates):
        tpl = all_templates["network"]
        tpl.resource_count_is("AWS::EC2::FlowLog", 1)

    def test_ecr_scan_on_push(self, all_templates):
        tpl = all_templates["storage"]
        repos = tpl.find_resources("AWS::ECR::Repository")
        for lid, repo in repos.items():
            scan_config = repo["Properties"].get("ImageScanningConfiguration", {})
            assert scan_config.get("ScanOnPush") is True, \
                f"ECR {lid} missing ScanOnPush"

    def test_ecr_immutable_tags(self, all_templates):
        tpl = all_templates["storage"]
        repos = tpl.find_resources("AWS::ECR::Repository")
        for lid, repo in repos.items():
            assert repo["Properties"].get("ImageTagMutability") == "IMMUTABLE", \
                f"ECR {lid} tags not immutable"


class TestMonitoringInvariants:
    """Budgets, alarms, trail all wired up."""

    def test_budget_thresholds(self, all_templates):
        tpl = all_templates["sagemaker"]
        budget_resources = tpl.find_resources("AWS::Budgets::Budget")
        assert len(budget_resources) >= 1
        for lid, b in budget_resources.items():
            notifs = b["Properties"].get("NotificationsWithSubscribers", [])
            thresholds = sorted([n["Notification"]["Threshold"] for n in notifs])
            assert thresholds == [50, 80, 100], \
                f"Budget {lid} thresholds: expected [50,80,100], got {thresholds}"

    def test_all_alarms_have_actions(self, all_templates):
        tpl = all_templates["sagemaker"]
        alarms = tpl.find_resources("AWS::CloudWatch::Alarm")
        for lid, alarm in alarms.items():
            actions = alarm["Properties"].get("AlarmActions", [])
            assert len(actions) > 0, f"Alarm {lid} has no actions"

    def test_cloudtrail_exists_with_logging(self, all_templates):
        tpl = all_templates["sagemaker"]
        tpl.has_resource_properties("AWS::CloudTrail::Trail", {
            "IsLogging": True,
        })

    def test_eventbridge_rules_target_sns(self, all_templates):
        tpl = all_templates["sagemaker"]
        rules = tpl.find_resources("AWS::Events::Rule")
        assert len(rules) >= 2
        for lid, rule in rules.items():
            rule_targets = rule["Properties"].get("Targets", [])
            assert len(rule_targets) > 0, f"EventBridge rule {lid} has no targets"
