"""IAM stack: four least-privilege roles for the ML pipeline."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_iam as iam,
    aws_s3 as s3,
    aws_kms as kms,
    aws_ecr as ecr,
    aws_ec2 as ec2,
)
from constructs import Construct

from constructs.least_privilege_role import PermissionBoundary
from config.environments import EnvironmentConfig
from config.constants import PROJECT_NAME


class IamStack(cdk.Stack):
    """Provisions IAM roles with permission boundaries."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: EnvironmentConfig,
        training_data_bucket: s3.IBucket,
        model_artifacts_bucket: s3.IBucket,
        mlflow_bucket: s3.IBucket,
        kms_key: kms.IKey,
        training_ecr_repo: ecr.IRepository,
        serving_ecr_repo: ecr.IRepository,
        vpc: ec2.IVpc,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Permission boundary for all roles
        boundary = PermissionBoundary(self, "Boundary", env_name=config.env_name)

        # ════════════════════════════════════════
        # ROLE 1: SageMaker Training
        # ════════════════════════════════════════
        self.training_role = iam.Role(
            self, "TrainingRole",
            role_name=f"{PROJECT_NAME}-{config.env_name}-sagemaker-training",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            permissions_boundary=boundary.policy,
        )

        # S3: read/write training data and model artifacts
        training_data_bucket.grant_read_write(self.training_role)
        model_artifacts_bucket.grant_read_write(self.training_role)
        mlflow_bucket.grant_read_write(self.training_role)

        # ECR: pull images
        training_ecr_repo.grant_pull(self.training_role)

        # KMS: decrypt and generate data keys
        kms_key.grant_encrypt_decrypt(self.training_role)

        # CloudWatch: logs and metrics (inline - no CDK grant method)
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="CloudWatchLogs",
            actions=[
                "logs:CreateLogGroup", "logs:CreateLogStream",
                "logs:PutLogEvents", "logs:DescribeLogStreams",
            ],
            resources=[
                f"arn:aws:logs:{config.region}:{config.account}:log-group:/aws/sagemaker/*",
            ],
        ))
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="CloudWatchMetrics",
            actions=["cloudwatch:PutMetricData"],
            resources=["*"],
            conditions={
                "StringEquals": {"cloudwatch:namespace": f"LLMFineTuning/{PROJECT_NAME}"},
            },
        ))

        # SSM: read project parameters only
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="SSMRead",
            actions=["ssm:GetParameter", "ssm:GetParameters"],
            resources=[
                f"arn:aws:ssm:{config.region}:{config.account}:parameter/{PROJECT_NAME}/*",
            ],
        ))

        # Secrets Manager: read project secrets only
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="SecretsRead",
            actions=["secretsmanager:GetSecretValue"],
            resources=[
                f"arn:aws:secretsmanager:{config.region}:{config.account}:secret:{PROJECT_NAME}/*",
            ],
        ))

        # VPC: create/delete network interfaces for VPC-mode training
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="VPCNetworking",
            actions=[
                "ec2:CreateNetworkInterface",
                "ec2:CreateNetworkInterfacePermission",
                "ec2:DeleteNetworkInterface",
                "ec2:DeleteNetworkInterfacePermission",
                "ec2:DescribeNetworkInterfaces",
                "ec2:DescribeVpcs",
                "ec2:DescribeDhcpOptions",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
            ],
            resources=["*"],  # EC2 Describe actions require *
            conditions={
                "StringEquals": {"ec2:Vpc": vpc.vpc_arn},
            },
        ))

        # ECR Auth (global)
        self.training_role.add_to_policy(iam.PolicyStatement(
            sid="ECRAuth",
            actions=["ecr:GetAuthorizationToken"],
            resources=["*"],
        ))

        # ════════════════════════════════════════
        # ROLE 2: SageMaker Endpoint (inference)
        # ════════════════════════════════════════
        self.endpoint_role = iam.Role(
            self, "EndpointRole",
            role_name=f"{PROJECT_NAME}-{config.env_name}-sagemaker-endpoint",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            permissions_boundary=boundary.policy,
        )
        model_artifacts_bucket.grant_read(self.endpoint_role)  # READ only, no write
        training_ecr_repo.grant_pull(self.endpoint_role)
        serving_ecr_repo.grant_pull(self.endpoint_role)
        kms_key.grant_decrypt(self.endpoint_role)  # Decrypt only, no encrypt
        # CloudWatch logs for inference container
        self.endpoint_role.add_to_policy(iam.PolicyStatement(
            sid="CloudWatchLogs",
            actions=["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"],
            resources=[f"arn:aws:logs:{config.region}:{config.account}:log-group:/aws/sagemaker/*"],
        ))

        # ════════════════════════════════════════
        # ROLE 3: Bedrock Import
        # ════════════════════════════════════════
        self.bedrock_import_role = iam.Role(
            self, "BedrockImportRole",
            role_name=f"{PROJECT_NAME}-{config.env_name}-bedrock-import",
            assumed_by=iam.ServicePrincipal("bedrock.amazonaws.com"),
            permissions_boundary=boundary.policy,
        )
        model_artifacts_bucket.grant_read(self.bedrock_import_role)
        kms_key.grant_decrypt(self.bedrock_import_role)

        # ════════════════════════════════════════
        # ROLE 4: Pipeline Execution (orchestration)
        # ════════════════════════════════════════
        self.pipeline_role = iam.Role(
            self, "PipelineRole",
            role_name=f"{PROJECT_NAME}-{config.env_name}-pipeline-execution",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("sagemaker.amazonaws.com"),
                iam.ServicePrincipal("states.amazonaws.com"),
            ),
            permissions_boundary=boundary.policy,
        )
        # PassRole ONLY to training and endpoint roles
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="PassRoleToTrainingAndEndpoint",
            actions=["iam:PassRole"],
            resources=[
                self.training_role.role_arn,
                self.endpoint_role.role_arn,
                self.bedrock_import_role.role_arn,
            ],
            conditions={
                "StringEquals": {
                    "iam:PassedToService": [
                        "sagemaker.amazonaws.com",
                        "bedrock.amazonaws.com",
                    ],
                },
            },
        ))
        # SageMaker job management
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="SageMakerJobManagement",
            actions=[
                "sagemaker:CreateTrainingJob", "sagemaker:DescribeTrainingJob",
                "sagemaker:StopTrainingJob", "sagemaker:ListTrainingJobs",
                "sagemaker:CreateModel", "sagemaker:DeleteModel",
                "sagemaker:CreateEndpointConfig", "sagemaker:DeleteEndpointConfig",
                "sagemaker:CreateEndpoint", "sagemaker:UpdateEndpoint",
                "sagemaker:DeleteEndpoint", "sagemaker:DescribeEndpoint",
                "sagemaker:InvokeEndpoint",
                "sagemaker:CreateModelPackage", "sagemaker:UpdateModelPackage",
                "sagemaker:DescribeModelPackage", "sagemaker:ListModelPackages",
                "sagemaker:CreateModelPackageGroup",
                "sagemaker:DescribeModelPackageGroup",
                "sagemaker:CreateHyperParameterTuningJob",
                "sagemaker:DescribeHyperParameterTuningJob",
                "sagemaker:StopHyperParameterTuningJob",
                "sagemaker:AddTags",
            ],
            resources=[
                f"arn:aws:sagemaker:{config.region}:{config.account}:*/{PROJECT_NAME}-*",
            ],
        ))
        # S3 access for all project buckets
        training_data_bucket.grant_read_write(self.pipeline_role)
        model_artifacts_bucket.grant_read_write(self.pipeline_role)
        mlflow_bucket.grant_read_write(self.pipeline_role)
        kms_key.grant_encrypt_decrypt(self.pipeline_role)
        # CloudWatch + EventBridge
        self.pipeline_role.add_to_policy(iam.PolicyStatement(
            sid="EventBridgeAndCloudWatch",
            actions=[
                "events:PutEvents", "events:PutRule", "events:PutTargets",
                "logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents",
                "cloudwatch:PutMetricData",
            ],
            resources=["*"],
        ))

        # ---- Outputs ----
        cdk.CfnOutput(self, "TrainingRoleArn", value=self.training_role.role_arn)
        cdk.CfnOutput(self, "EndpointRoleArn", value=self.endpoint_role.role_arn)
        cdk.CfnOutput(self, "BedrockImportRoleArn", value=self.bedrock_import_role.role_arn)
        cdk.CfnOutput(self, "PipelineRoleArn", value=self.pipeline_role.role_arn)
