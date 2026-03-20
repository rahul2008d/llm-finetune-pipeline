"""Unit tests for config.schema module."""

from pathlib import Path

import pytest

from src.config.schema import YAMLSchemaLoader


class TestYAMLSchemaLoader:
    """Tests for YAMLSchemaLoader."""

    def test_load_training_config(self, sample_yaml_config: Path) -> None:
        """Test loading a valid training config."""
        config = YAMLSchemaLoader.load(sample_yaml_config, "training")
        assert config.model_id == "test-model"  # type: ignore[attr-defined]
        assert config.epochs == 1  # type: ignore[attr-defined]

    def test_unknown_schema_type(self, sample_yaml_config: Path) -> None:
        """Test that unknown schema types raise ValueError."""
        with pytest.raises(ValueError, match="Unknown schema type"):
            YAMLSchemaLoader.load(sample_yaml_config, "unknown")

    def test_load_raw(self, sample_yaml_config: Path) -> None:
        """Test raw YAML loading without validation."""
        raw = YAMLSchemaLoader.load_raw(sample_yaml_config)
        assert isinstance(raw, dict)
        assert "model_id" in raw
