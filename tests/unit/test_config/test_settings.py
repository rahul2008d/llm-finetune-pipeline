"""Unit tests for config.settings module."""

import pytest

from src.config.settings import AppSettings, TrainingSettings


class TestAppSettings:
    """Tests for AppSettings."""

    def test_defaults(self) -> None:
        """Test default values are set correctly."""
        settings = AppSettings()
        assert settings.environment == "development"
        assert settings.log_level == "INFO"
        assert settings.log_format == "json"

    def test_from_env(self, mock_env_vars: dict[str, str]) -> None:
        """Test loading settings from environment variables."""
        settings = AppSettings()
        assert settings.environment == "test"
        assert settings.log_level == "DEBUG"
        assert settings.aws_region == "us-east-1"


class TestTrainingSettings:
    """Tests for TrainingSettings."""

    def test_defaults(self) -> None:
        """Test default hyperparameters."""
        settings = TrainingSettings()
        assert settings.epochs == 3
        assert settings.batch_size == 4
        assert settings.lora_rank == 64

    def test_validation(self) -> None:
        """Test that invalid values raise validation errors."""
        with pytest.raises(Exception):
            TrainingSettings(epochs=0)
