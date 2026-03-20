"""Unit tests for config.environment module."""

import os

import pytest

from src.config.environment import EnvironmentResolver


class TestEnvironmentResolver:
    """Tests for EnvironmentResolver."""

    def test_resolve_simple_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving a simple environment variable."""
        monkeypatch.setenv("TEST_VAR", "hello")
        result = EnvironmentResolver.resolve("${TEST_VAR}")
        assert result == "hello"

    def test_resolve_with_default(self) -> None:
        """Test resolving with a default value when var is not set."""
        result = EnvironmentResolver.resolve("${NONEXISTENT_VAR:-fallback}")
        assert result == "fallback"

    def test_resolve_missing_raises(self) -> None:
        """Test that missing vars without defaults raise KeyError."""
        with pytest.raises(KeyError):
            EnvironmentResolver.resolve("${DEFINITELY_NOT_SET}")

    def test_resolve_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test recursive dictionary resolution."""
        monkeypatch.setenv("DB_HOST", "localhost")
        data = {"host": "${DB_HOST}", "port": 5432, "nested": {"name": "${DB_HOST}"}}
        resolved = EnvironmentResolver.resolve_dict(data)
        assert resolved["host"] == "localhost"
        assert resolved["port"] == 5432
        assert resolved["nested"]["name"] == "localhost"
