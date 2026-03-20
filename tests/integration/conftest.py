"""Integration test configuration and fixtures."""

import pytest


@pytest.fixture()
def integration_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up environment variables for integration tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Dictionary of environment variable values.
    """
    env_vars = {
        "ENVIRONMENT": "integration",
        "LOG_LEVEL": "INFO",
        "AWS_REGION": "us-east-1",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
