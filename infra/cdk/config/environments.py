"""Environment-specific configuration using frozen dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnvironmentConfig:
    """Immutable, validated configuration for a deployment environment."""

    env_name: str  # 'dev' | 'staging' | 'prod'
    account: str
    region: str
    vpc_cidr: str
    nat_gateways: int
    enable_bedrock: bool
    budget_limit_usd: int
    alert_email: str
    kms_admin_arns: list[str] = field(default_factory=list)
    training_instance_types: list[str] = field(
        default_factory=lambda: ["ml.g5.2xlarge", "ml.g5.12xlarge"]
    )
    flow_log_retention_days: int = 7
    cloudtrail_retention_days: int = 90
    removal_policy_retain: bool = False  # True for prod

    def __post_init__(self) -> None:
        if self.env_name not in ("dev", "staging", "prod"):
            raise ValueError(f"Invalid env_name: {self.env_name}")
        if self.nat_gateways < 1 or self.nat_gateways > 3:
            raise ValueError("nat_gateways must be 1–3")
        if self.budget_limit_usd < 0:
            raise ValueError("budget_limit_usd must be non-negative")


# Concrete environment definitions
ENVIRONMENTS: dict[str, EnvironmentConfig] = {
    "dev": EnvironmentConfig(
        env_name="dev",
        account="111111111111",  # placeholder - override via context or env var
        region="us-east-1",
        vpc_cidr="10.0.0.0/16",
        nat_gateways=1,
        enable_bedrock=True,
        budget_limit_usd=200,
        alert_email="mlops-dev@company.com",
        flow_log_retention_days=7,
        removal_policy_retain=False,
    ),
    "staging": EnvironmentConfig(
        env_name="staging",
        account="222222222222",
        region="us-east-1",
        vpc_cidr="10.1.0.0/16",
        nat_gateways=1,
        enable_bedrock=True,
        budget_limit_usd=500,
        alert_email="mlops-staging@company.com",
        flow_log_retention_days=30,
        removal_policy_retain=False,
    ),
    "prod": EnvironmentConfig(
        env_name="prod",
        account="333333333333",
        region="us-east-1",
        vpc_cidr="10.2.0.0/16",
        nat_gateways=2,
        enable_bedrock=True,
        budget_limit_usd=2000,
        alert_email="mlops-prod@company.com",
        flow_log_retention_days=90,
        cloudtrail_retention_days=365,
        removal_policy_retain=True,
    ),
}


def get_environment_config(env_name: str) -> EnvironmentConfig:
    """Retrieve validated config for the given environment."""
    if env_name not in ENVIRONMENTS:
        raise KeyError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(ENVIRONMENTS.keys())}"
        )
    return ENVIRONMENTS[env_name]
