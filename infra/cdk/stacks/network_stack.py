"""Network stack: VPC with SageMaker-optimized subnets and endpoints."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import aws_ec2 as ec2
from constructs import Construct

from config.environments import EnvironmentConfig
from constructs.training_vpc import TrainingVpc, TrainingVpcProps


class NetworkStack(cdk.Stack):
    """Creates the foundational VPC and networking layer."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: EnvironmentConfig,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)
        self.vpc_construct = TrainingVpc(
            self,
            "TrainingVpc",
            props=TrainingVpcProps(
                cidr=config.vpc_cidr,
                nat_gateways=config.nat_gateways,
                enable_bedrock=config.enable_bedrock,
                env_name=config.env_name,
                flow_log_retention_days=config.flow_log_retention_days,
            ),
        )

        # Outputs for cross-account or CLI reference
        cdk.CfnOutput(self, "VpcId", value=self.vpc_construct.vpc.vpc_id)
        cdk.CfnOutput(
            self,
            "PrivateSubnetIds",
            value=",".join(
                [
                    s.subnet_id
                    for s in self.vpc_construct.vpc.select_subnets(
                        subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS
                    ).subnets
                ]
            ),
        )
        cdk.CfnOutput(
            self,
            "TrainingSecurityGroupId",
            value=self.vpc_construct.training_sg.security_group_id,
        )
