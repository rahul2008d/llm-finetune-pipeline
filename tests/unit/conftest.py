"""Unit test configuration and fixtures."""

import pytest


@pytest.fixture()
def mock_env_vars(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up common environment variables for unit tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Dictionary of environment variable values.
    """
    env_vars = {
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "DEBUG",
        "LOG_FORMAT": "console",
        "AWS_REGION": "us-east-1",
        "S3_BUCKET_DATA": "test-data-bucket",
        "S3_BUCKET_MODELS": "test-models-bucket",
        "S3_BUCKET_ARTIFACTS": "test-artifacts-bucket",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
