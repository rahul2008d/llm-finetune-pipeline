"""Shared constants for all CDK stacks."""
from typing import Final

PROJECT_NAME: Final[str] = "llm-finetune"

# VPC endpoint services to create (Interface type unless noted)
VPC_INTERFACE_ENDPOINTS: Final[list[str]] = [
    "sagemaker.api",
    "sagemaker.runtime",
    "sts",
    "ecr.api",
    "ecr.dkr",
    "logs",
    "secretsmanager",
    "ssm",
    "kms",
    "monitoring",  # CloudWatch metrics
]

BEDROCK_ENDPOINT: Final[str] = "bedrock-runtime"

# Tags applied to every resource
DEFAULT_TAGS: Final[dict[str, str]] = {
    "Project": PROJECT_NAME,
    "ManagedBy": "cdk",
    "Repository": "llm-finetune-pipeline",
}
