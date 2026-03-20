"""CDK assertion tests for the Network stack."""
from __future__ import annotations

import pytest
import aws_cdk as cdk
from aws_cdk.assertions import Match, Template

from config.environments import EnvironmentConfig
from stacks.network_stack import NetworkStack


def _make_template(
    *,
    nat_gateways: int = 1,
    enable_bedrock: bool = True,
    flow_log_retention_days: int = 7,
) -> Template:
    """Synthesize a NetworkStack with the given overrides and return its Template."""
    app = cdk.App()
    config = EnvironmentConfig(
        env_name="dev",
        account="111111111111",
        region="us-east-1",
        vpc_cidr="10.0.0.0/16",
        nat_gateways=nat_gateways,
        enable_bedrock=enable_bedrock,
        budget_limit_usd=200,
        alert_email="test@test.com",
        flow_log_retention_days=flow_log_retention_days,
    )
    stack = NetworkStack(
        app,
        "test-network",
        config=config,
        env=cdk.Environment(account="111111111111", region="us-east-1"),
    )
    return Template.from_stack(stack)


@pytest.fixture
def dev_template() -> Template:
    """Dev environment template with 1 NAT gateway and Bedrock enabled."""
    return _make_template()


@pytest.fixture
def prod_template() -> Template:
    """Prod-like template with 2 NAT gateways and Bedrock enabled."""
    return _make_template(nat_gateways=2)


@pytest.fixture
def no_bedrock_template() -> Template:
    """Template with Bedrock endpoint disabled."""
    return _make_template(enable_bedrock=False)


# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------


def test_vpc_exists(dev_template: Template) -> None:
    """A single VPC must be created."""
    dev_template.resource_count_is("AWS::EC2::VPC", 1)


# ---------------------------------------------------------------------------
# NAT Gateways
# ---------------------------------------------------------------------------


def test_dev_has_one_nat_gateway(dev_template: Template) -> None:
    """Dev should have exactly 1 NAT gateway."""
    dev_template.resource_count_is("AWS::EC2::NatGateway", 1)


def test_prod_has_two_nat_gateways(prod_template: Template) -> None:
    """Prod should have 2 NAT gateways for HA."""
    prod_template.resource_count_is("AWS::EC2::NatGateway", 2)


# ---------------------------------------------------------------------------
# Subnets
# ---------------------------------------------------------------------------


def test_three_subnet_types(dev_template: Template) -> None:
    """Should have Public + Private + Isolated subnets across 2 AZs = 6 subnets."""
    dev_template.resource_count_is("AWS::EC2::Subnet", 6)


# ---------------------------------------------------------------------------
# VPC Endpoints
# ---------------------------------------------------------------------------


def test_interface_endpoints_created(dev_template: Template) -> None:
    """At least the SageMaker API endpoint should exist with private DNS."""
    dev_template.has_resource_properties(
        "AWS::EC2::VPCEndpoint",
        {
            "ServiceName": Match.string_like_regexp(".*sagemaker\\.api"),
            "PrivateDnsEnabled": True,
        },
    )


def test_no_bedrock_endpoint_when_disabled(no_bedrock_template: Template) -> None:
    """Verify bedrock-runtime endpoint is NOT present when disabled."""
    resources = no_bedrock_template.find_resources("AWS::EC2::VPCEndpoint")
    for r in resources.values():
        service_name = r.get("Properties", {}).get("ServiceName", "")
        assert "bedrock" not in str(service_name).lower()


# ---------------------------------------------------------------------------
# Flow Logs
# ---------------------------------------------------------------------------


def test_flow_logs_enabled(dev_template: Template) -> None:
    """VPC flow logs must be enabled with TrafficType ALL."""
    dev_template.has_resource_properties(
        "AWS::EC2::FlowLog",
        {
            "TrafficType": "ALL",
        },
    )


# ---------------------------------------------------------------------------
# Security Groups
# ---------------------------------------------------------------------------


def test_security_groups_exist(dev_template: Template) -> None:
    """Should have at least 3 SGs: training, endpoint, sagemaker-endpoint."""
    resources = dev_template.find_resources("AWS::EC2::SecurityGroup")
    assert len(resources) >= 3


def test_training_sg_no_unrestricted_ingress(dev_template: Template) -> None:
    """Verify no ingress rule with 0.0.0.0/0."""
    ingress_rules = dev_template.find_resources("AWS::EC2::SecurityGroupIngress")
    for rule in ingress_rules.values():
        cidr = rule.get("Properties", {}).get("CidrIp", "")
        assert cidr != "0.0.0.0/0", "Training SG must not allow unrestricted ingress"
