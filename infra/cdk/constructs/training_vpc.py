"""Reusable VPC construct with SageMaker-optimized networking."""
from __future__ import annotations

from dataclasses import dataclass

import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_logs as logs,
)
from constructs import Construct

from config.constants import BEDROCK_ENDPOINT, VPC_INTERFACE_ENDPOINTS


@dataclass(frozen=True)
class TrainingVpcProps:
    """Properties for TrainingVpc construct."""

    cidr: str
    nat_gateways: int
    enable_bedrock: bool
    env_name: str
    flow_log_retention_days: int


class TrainingVpc(Construct):
    """VPC with private subnets, VPC endpoints, and flow logs for SageMaker training."""

    def __init__(self, scope: Construct, id: str, *, props: TrainingVpcProps) -> None:
        super().__init__(scope, id)

        # ---- VPC ----
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            ip_addresses=ec2.IpAddresses.cidr(props.cidr),
            max_azs=2,
            nat_gateways=props.nat_gateways,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Public",
                    subnet_type=ec2.SubnetType.PUBLIC,
                    cidr_mask=24,
                ),
                ec2.SubnetConfiguration(
                    name="Private",
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS,
                    cidr_mask=22,  # /22 = 1024 IPs per AZ for training
                ),
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                ),
            ],
            enable_dns_hostnames=True,
            enable_dns_support=True,
        )

        # ---- Security Groups ----
        self.training_sg = ec2.SecurityGroup(
            self,
            "TrainingSG",
            vpc=self.vpc,
            description="SageMaker training jobs - outbound to VPC endpoints + NAT",
            allow_all_outbound=False,  # explicit rules only
        )
        # Allow HTTPS to anywhere (NAT for ECR/HF model downloads)
        self.training_sg.add_egress_rule(
            peer=ec2.Peer.any_ipv4(),
            connection=ec2.Port.tcp(443),
            description="HTTPS outbound via NAT for ECR and model downloads",
        )
        # Allow all traffic within the SG (for multi-node distributed training)
        self.training_sg.add_ingress_rule(
            peer=self.training_sg,
            connection=ec2.Port.all_traffic(),
            description="Inter-node communication for distributed training",
        )

        self.endpoint_sg = ec2.SecurityGroup(
            self,
            "EndpointSG",
            vpc=self.vpc,
            description="VPC endpoints - inbound HTTPS from training SG",
            allow_all_outbound=False,
        )
        self.endpoint_sg.add_ingress_rule(
            peer=self.training_sg,
            connection=ec2.Port.tcp(443),
            description="HTTPS from training jobs to VPC endpoints",
        )

        self.sagemaker_endpoint_sg = ec2.SecurityGroup(
            self,
            "SageMakerEndpointSG",
            vpc=self.vpc,
            description="SageMaker inference endpoints - inbound HTTPS from app tier",
            allow_all_outbound=False,
        )
        self.sagemaker_endpoint_sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(props.cidr),
            connection=ec2.Port.tcp(443),
            description="HTTPS from VPC CIDR to inference endpoints",
        )

        # ---- S3 Gateway Endpoint (free, no interface charge) ----
        self.vpc.add_gateway_endpoint(
            "S3Gateway",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[
                ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
                ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED),
            ],
        )

        # ---- Interface VPC Endpoints ----
        # Mapping from our constant names to CDK InterfaceVpcEndpointAwsService attrs
        endpoint_service_map: dict[str, ec2.InterfaceVpcEndpointAwsService] = {
            "sagemaker.api": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_API,
            "sagemaker.runtime": ec2.InterfaceVpcEndpointAwsService.SAGEMAKER_RUNTIME,
            "sts": ec2.InterfaceVpcEndpointAwsService.STS,
            "ecr.api": ec2.InterfaceVpcEndpointAwsService.ECR,
            "ecr.dkr": ec2.InterfaceVpcEndpointAwsService.ECR_DOCKER,
            "logs": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
            "secretsmanager": ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
            "ssm": ec2.InterfaceVpcEndpointAwsService.SSM,
            "kms": ec2.InterfaceVpcEndpointAwsService.KMS,
            "monitoring": ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_MONITORING,
        }
        self._endpoints: dict[str, ec2.InterfaceVpcEndpoint] = {}
        for svc_name in VPC_INTERFACE_ENDPOINTS:
            svc = endpoint_service_map[svc_name]
            logical_id = svc_name.replace(".", "-").title().replace("-", "")
            self._endpoints[svc_name] = ec2.InterfaceVpcEndpoint(
                self,
                f"{logical_id}Endpoint",
                vpc=self.vpc,
                service=svc,
                subnets=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                ),
                security_groups=[self.endpoint_sg],
                private_dns_enabled=True,
            )

        # Conditional Bedrock endpoint
        if props.enable_bedrock:
            self._endpoints[BEDROCK_ENDPOINT] = ec2.InterfaceVpcEndpoint(
                self,
                "BedrockRuntimeEndpoint",
                vpc=self.vpc,
                service=ec2.InterfaceVpcEndpointAwsService.BEDROCK_RUNTIME,
                subnets=ec2.SubnetSelection(
                    subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                ),
                security_groups=[self.endpoint_sg],
                private_dns_enabled=True,
            )

        # ---- Flow Logs ----
        log_group = logs.LogGroup(
            self,
            "FlowLogGroup",
            retention=getattr(
                logs.RetentionDays,
                {
                    7: "ONE_WEEK",
                    30: "ONE_MONTH",
                    90: "THREE_MONTHS",
                    365: "ONE_YEAR",
                }.get(props.flow_log_retention_days, "ONE_WEEK"),
            ),
            removal_policy=cdk.RemovalPolicy.DESTROY,
        )
        self.vpc.add_flow_log(
            "FlowLog",
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(log_group),
            traffic_type=ec2.FlowLogTrafficType.ALL,
        )

    @property
    def private_subnets(self) -> ec2.SubnetSelection:
        """Return subnet selection for private subnets with egress."""
        return ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)

    @property
    def isolated_subnets(self) -> ec2.SubnetSelection:
        """Return subnet selection for isolated subnets."""
        return ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED)
