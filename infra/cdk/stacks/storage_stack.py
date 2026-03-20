"""Storage stack: KMS key, encrypted S3 buckets, and ECR repositories."""
from __future__ import annotations

import aws_cdk as cdk
from aws_cdk import (
    aws_kms as kms,
    aws_s3 as s3,
    aws_ecr as ecr,
    aws_iam as iam,
)
from constructs import Construct

from constructs.secure_bucket import SecureBucket, SecureBucketProps
from constructs.training_vpc import TrainingVpc
from config.environments import EnvironmentConfig
from config.constants import PROJECT_NAME


class StorageStack(cdk.Stack):
    """Provisions encrypted storage and container registries."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        config: EnvironmentConfig,
        vpc: TrainingVpc,
        **kwargs: object,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---- KMS Key ----
        self.kms_key = kms.Key(
            self, "SageMakerKey",
            alias=f"alias/{PROJECT_NAME}-{config.env_name}-sagemaker",
            description="Encryption key for training data and model artifacts",
            enable_key_rotation=True,
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )
        # Allow SageMaker service to use the key
        self.kms_key.grant_encrypt_decrypt(
            iam.ServicePrincipal("sagemaker.amazonaws.com")
        )

        # ---- S3 Buckets ----
        self.training_data_bucket = SecureBucket(
            self, "TrainingData",
            props=SecureBucketProps(
                bucket_name_suffix="training-data",
                encryption_key=self.kms_key,
                env_name=config.env_name,
                lifecycle_rules=[
                    s3.LifecycleRule(
                        transitions=[
                            s3.Transition(
                                storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                                transition_after=cdk.Duration.days(30),
                            ),
                            s3.Transition(
                                storage_class=s3.StorageClass.GLACIER,
                                transition_after=cdk.Duration.days(90),
                            ),
                        ],
                        expiration=cdk.Duration.days(365),
                    ),
                ],
            ),
        )

        self.model_artifacts_bucket = SecureBucket(
            self, "ModelArtifacts",
            props=SecureBucketProps(
                bucket_name_suffix="model-artifacts",
                encryption_key=self.kms_key,
                env_name=config.env_name,
                # No lifecycle - models kept indefinitely
            ),
        )

        self.mlflow_bucket = SecureBucket(
            self, "MLflow",
            props=SecureBucketProps(
                bucket_name_suffix="mlflow-tracking",
                encryption_key=self.kms_key,
                env_name=config.env_name,
                lifecycle_rules=[
                    s3.LifecycleRule(
                        transitions=[
                            s3.Transition(
                                storage_class=s3.StorageClass.INFREQUENT_ACCESS,
                                transition_after=cdk.Duration.days(90),
                            ),
                        ],
                    ),
                ],
            ),
        )

        # ---- ECR Repositories ----
        self.training_ecr_repo = ecr.Repository(
            self, "TrainingRepo",
            repository_name=f"{PROJECT_NAME}-training",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    max_image_count=10,
                    description="Keep last 10 images",
                ),
            ],
            removal_policy=cdk.RemovalPolicy.RETAIN,
            encryption=ecr.RepositoryEncryption.KMS,
            encryption_key=self.kms_key,
        )

        self.serving_ecr_repo = ecr.Repository(
            self, "ServingRepo",
            repository_name=f"{PROJECT_NAME}-serving",
            image_scan_on_push=True,
            image_tag_mutability=ecr.TagMutability.IMMUTABLE,
            lifecycle_rules=[
                ecr.LifecycleRule(max_image_count=10, description="Keep last 10"),
            ],
            removal_policy=cdk.RemovalPolicy.RETAIN,
            encryption=ecr.RepositoryEncryption.KMS,
            encryption_key=self.kms_key,
        )

        # ---- Outputs ----
        cdk.CfnOutput(self, "KmsKeyArn", value=self.kms_key.key_arn)
        cdk.CfnOutput(self, "TrainingDataBucket", value=self.training_data_bucket.bucket.bucket_name)
        cdk.CfnOutput(self, "ModelArtifactsBucket", value=self.model_artifacts_bucket.bucket.bucket_name)
        cdk.CfnOutput(self, "TrainingEcrUri", value=self.training_ecr_repo.repository_uri)
        cdk.CfnOutput(self, "ServingEcrUri", value=self.serving_ecr_repo.repository_uri)
