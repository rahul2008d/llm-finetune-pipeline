"""End-to-end test configuration and fixtures."""

import pytest


@pytest.fixture()
def e2e_env(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    """Set up environment variables for e2e tests.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        Dictionary of environment variable values.
    """
    env_vars = {
        "ENVIRONMENT": "e2e",
        "LOG_LEVEL": "INFO",
        "AWS_REGION": "us-east-1",
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars
