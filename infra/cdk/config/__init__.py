"""Configuration package for CDK environment settings and constants."""

from config.constants import BEDROCK_ENDPOINT, DEFAULT_TAGS, PROJECT_NAME, VPC_INTERFACE_ENDPOINTS
from config.environments import ENVIRONMENTS, EnvironmentConfig, get_environment_config

__all__: list[str] = [
    "PROJECT_NAME",
    "VPC_INTERFACE_ENDPOINTS",
    "BEDROCK_ENDPOINT",
    "DEFAULT_TAGS",
    "EnvironmentConfig",
    "ENVIRONMENTS",
    "get_environment_config",
]
