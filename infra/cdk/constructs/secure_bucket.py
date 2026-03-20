"""Reusable encrypted S3 bucket with security best practices."""
from __future__ import annotations

from dataclasses import dataclass, field

import aws_cdk as cdk
from aws_cdk import (
    aws_s3 as s3,
    aws_kms as kms,
    aws_iam as iam,
)
from constructs import Construct


@dataclass(frozen=True)
class SecureBucketProps:
    """Configuration for a security-hardened S3 bucket."""

    bucket_name_suffix: str
    encryption_key: kms.IKey
    env_name: str
    lifecycle_rules: list[s3.LifecycleRule] = field(default_factory=list)
    versioned: bool = True


class SecureBucket(Construct):
    """S3 bucket with KMS encryption, SSL enforcement, and public access block."""

    def __init__(self, scope: Construct, id: str, *, props: SecureBucketProps) -> None:
        super().__init__(scope, id)
        from config.constants import PROJECT_NAME

        is_prod = props.env_name == "prod"

        self.bucket = s3.Bucket(
            self, "Bucket",
            bucket_name=f"{PROJECT_NAME}-{props.env_name}-{props.bucket_name_suffix}",
            encryption=s3.BucketEncryption.KMS,
            encryption_key=props.encryption_key,
            bucket_key_enabled=True,  # reduce KMS API calls
            versioned=props.versioned,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,  # auto-adds deny-HTTP policy
            removal_policy=(
                cdk.RemovalPolicy.RETAIN if is_prod
                else cdk.RemovalPolicy.DESTROY
            ),
            auto_delete_objects=not is_prod,
            lifecycle_rules=props.lifecycle_rules or [],
            server_access_logs_prefix="access-logs/",
        )

        # Deny uploads without KMS encryption header
        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyUnencryptedUploads",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["s3:PutObject"],
                resources=[self.bucket.arn_for_objects("*")],
                conditions={
                    "StringNotEquals": {
                        "s3:x-amz-server-side-encryption": "aws:kms",
                    },
                },
            )
        )

        # Deny requests not using TLS 1.2+
        self.bucket.add_to_resource_policy(
            iam.PolicyStatement(
                sid="DenyInsecureTLS",
                effect=iam.Effect.DENY,
                principals=[iam.AnyPrincipal()],
                actions=["s3:*"],
                resources=[
                    self.bucket.bucket_arn,
                    self.bucket.arn_for_objects("*"),
                ],
                conditions={
                    "NumericLessThan": {
                        "s3:TlsVersion": "1.2",
                    },
                },
            )
        )
