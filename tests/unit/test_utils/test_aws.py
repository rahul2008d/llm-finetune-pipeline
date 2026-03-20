"""Unit tests for utils.aws module using moto for AWS mocking."""

import json
import time

import boto3
import pytest
from moto import mock_aws

from src.utils import aws as aws_mod
from src.utils.aws import get_account_id, get_parameter, get_region, get_secret, get_session


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    """Clear module-level caches between tests."""
    aws_mod._SECRET_CACHE.clear()
    aws_mod._PARAM_CACHE.clear()


@pytest.fixture()
def _aws_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set minimal AWS env vars for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.delenv("AWS_PROFILE", raising=False)


class TestGetSession:
    """Tests for get_session."""

    def test_default_session(self, _aws_env: None) -> None:
        """Test session creation without a profile."""
        session = get_session()
        assert isinstance(session, boto3.Session)

    def test_profile_from_env(
        self, monkeypatch: pytest.MonkeyPatch, _aws_env: None
    ) -> None:
        """Test that AWS_PROFILE env var is respected when no arg is given."""
        monkeypatch.setenv("AWS_PROFILE", "my-profile")
        # get_session should not raise; it creates a session object
        # (profile resolution happens later on actual API calls)
        session = get_session()
        assert isinstance(session, boto3.Session)


class TestGetSecret:
    """Tests for get_secret with moto Secrets Manager mock."""

    @mock_aws
    def test_retrieves_secret(self, _aws_env: None) -> None:
        """Test fetching a secret from Secrets Manager."""
        sm = boto3.client("secretsmanager", region_name="us-east-1")
        sm.create_secret(Name="my/secret", SecretString="s3cr3t")

        session = boto3.Session(region_name="us-east-1")
        value = get_secret("my/secret", session=session)
        assert value == "s3cr3t"

    @mock_aws
    def test_secret_is_cached(self, _aws_env: None) -> None:
        """Test that repeated calls return the cached value."""
        sm = boto3.client("secretsmanager", region_name="us-east-1")
        sm.create_secret(Name="cached/secret", SecretString="original")

        session = boto3.Session(region_name="us-east-1")
        first = get_secret("cached/secret", session=session, ttl=300)
        # Mutate the secret in SM — cached value should still be returned
        sm.update_secret(SecretId="cached/secret", SecretString="changed")
        second = get_secret("cached/secret", session=session, ttl=300)
        assert first == second == "original"

    @mock_aws
    def test_cache_expires(self, _aws_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that TTL expiry forces a fresh fetch."""
        sm = boto3.client("secretsmanager", region_name="us-east-1")
        sm.create_secret(Name="ttl/secret", SecretString="v1")

        session = boto3.Session(region_name="us-east-1")
        get_secret("ttl/secret", session=session, ttl=0.0)  # immediate expiry

        sm.update_secret(SecretId="ttl/secret", SecretString="v2")
        value = get_secret("ttl/secret", session=session, ttl=0.0)
        assert value == "v2"


class TestGetParameter:
    """Tests for get_parameter with moto SSM mock."""

    @mock_aws
    def test_retrieves_parameter(self, _aws_env: None) -> None:
        """Test fetching a plain-text parameter."""
        ssm = boto3.client("ssm", region_name="us-east-1")
        ssm.put_parameter(Name="/app/db_host", Value="localhost", Type="String")

        session = boto3.Session(region_name="us-east-1")
        value = get_parameter("/app/db_host", session=session)
        assert value == "localhost"

    @mock_aws
    def test_retrieves_secure_string(self, _aws_env: None) -> None:
        """Test fetching a SecureString parameter with decryption."""
        ssm = boto3.client("ssm", region_name="us-east-1")
        ssm.put_parameter(Name="/app/api_key", Value="key123", Type="SecureString")

        session = boto3.Session(region_name="us-east-1")
        value = get_parameter("/app/api_key", session=session, decrypt=True)
        assert value == "key123"

    @mock_aws
    def test_parameter_is_cached(self, _aws_env: None) -> None:
        """Test that repeated calls use the cache."""
        ssm = boto3.client("ssm", region_name="us-east-1")
        ssm.put_parameter(Name="/cached/param", Value="orig", Type="String")

        session = boto3.Session(region_name="us-east-1")
        first = get_parameter("/cached/param", session=session, ttl=300)
        ssm.put_parameter(Name="/cached/param", Value="new", Type="String", Overwrite=True)
        second = get_parameter("/cached/param", session=session, ttl=300)
        assert first == second == "orig"


class TestGetAccountId:
    """Tests for get_account_id."""

    @mock_aws
    def test_returns_account_id(self, _aws_env: None) -> None:
        """Test that a 12-digit account ID is returned."""
        session = boto3.Session(region_name="us-east-1")
        account_id = get_account_id(session=session)
        assert len(account_id) == 12
        assert account_id.isdigit()


class TestGetRegion:
    """Tests for get_region."""

    def test_returns_session_region(self, _aws_env: None) -> None:
        """Test region from session."""
        session = boto3.Session(region_name="eu-west-1")
        assert get_region(session=session) == "eu-west-1"

    def test_falls_back_to_env(
        self, monkeypatch: pytest.MonkeyPatch, _aws_env: None
    ) -> None:
        """Test fallback to AWS_DEFAULT_REGION."""
        monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-south-1")
        session = boto3.Session()
        # boto3 Session picks up AWS_DEFAULT_REGION
        region = get_region(session=session)
        assert region == "ap-south-1"
