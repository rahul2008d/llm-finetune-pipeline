"""AWS session and service helpers.

Provides cached access to boto3 sessions, Secrets Manager secrets,
SSM Parameter Store parameters, and account/region metadata.
"""

import os
import time
from typing import Any

import boto3
import structlog

logger = structlog.get_logger(__name__)

# Simple TTL caches: mapping of name → (value, expiry_epoch)
_SECRET_CACHE: dict[str, tuple[str, float]] = {}
_PARAM_CACHE: dict[str, tuple[str, float]] = {}
_DEFAULT_TTL: float = 300.0  # 5 minutes


def get_session(profile_name: str | None = None) -> boto3.Session:
    """Create a boto3 session with optional profile fallback for local dev.

    When running locally, pass ``profile_name`` or set ``AWS_PROFILE`` to
    use named profiles.  Inside SageMaker / ECS the default credential
    chain is used automatically.

    Args:
        profile_name: AWS CLI profile name.  Falls back to ``AWS_PROFILE``
            env var when ``None``.

    Returns:
        Configured ``boto3.Session``.
    """
    profile = profile_name or os.environ.get("AWS_PROFILE")
    if profile:
        logger.info("Creating boto3 session with profile", profile=profile)
        return boto3.Session(profile_name=profile)
    logger.info("Creating boto3 session with default credentials")
    return boto3.Session()


def get_secret(
    name: str,
    session: boto3.Session | None = None,
    ttl: float = _DEFAULT_TTL,
) -> str:
    """Retrieve a secret from AWS Secrets Manager (cached with TTL).

    Args:
        name: Secret name or ARN.
        session: Optional pre-configured boto3 session.
        ttl: Cache lifetime in seconds.

    Returns:
        Secret string value.
    """
    now = time.monotonic()
    cached = _SECRET_CACHE.get(name)
    if cached and cached[1] > now:
        return cached[0]

    sess = session or get_session()
    client: Any = sess.client("secretsmanager")
    resp = client.get_secret_value(SecretId=name)
    value: str = resp["SecretString"]
    _SECRET_CACHE[name] = (value, now + ttl)
    logger.info("Secret retrieved", name=name)
    return value


def get_parameter(
    name: str,
    session: boto3.Session | None = None,
    ttl: float = _DEFAULT_TTL,
    decrypt: bool = True,
) -> str:
    """Retrieve a parameter from SSM Parameter Store (cached with TTL).

    Args:
        name: Parameter name.
        session: Optional pre-configured boto3 session.
        ttl: Cache lifetime in seconds.
        decrypt: Whether to decrypt SecureString parameters.

    Returns:
        Parameter string value.
    """
    now = time.monotonic()
    cached = _PARAM_CACHE.get(name)
    if cached and cached[1] > now:
        return cached[0]

    sess = session or get_session()
    client: Any = sess.client("ssm")
    resp = client.get_parameter(Name=name, WithDecryption=decrypt)
    value: str = resp["Parameter"]["Value"]
    _PARAM_CACHE[name] = (value, now + ttl)
    logger.info("Parameter retrieved", name=name)
    return value


def get_account_id(session: boto3.Session | None = None) -> str:
    """Return the current AWS account ID.

    Args:
        session: Optional pre-configured boto3 session.

    Returns:
        12-digit AWS account ID string.
    """
    sess = session or get_session()
    client: Any = sess.client("sts")
    identity: dict[str, Any] = client.get_caller_identity()
    account_id: str = identity["Account"]
    return account_id


def get_region(session: boto3.Session | None = None) -> str:
    """Return the active AWS region.

    Args:
        session: Optional pre-configured boto3 session.

    Returns:
        AWS region string (e.g. ``us-east-1``).
    """
    sess = session or get_session()
    region: str = sess.region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    return region


__all__: list[str] = [
    "get_session",
    "get_secret",
    "get_parameter",
    "get_account_id",
    "get_region",
]
