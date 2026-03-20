"""Unit tests for data.validation module."""

import pytest
from datasets import Dataset

from src.data.validation import DataValidator, ValidationError


class TestDataValidator:
    """Tests for DataValidator."""

    def test_validate_columns_pass(self) -> None:
        """Test that valid columns pass validation."""
        ds = Dataset.from_dict({"instruction": ["a"], "output": ["b"]})
        DataValidator.validate_columns(ds, ["instruction", "output"])

    def test_validate_columns_missing(self) -> None:
        """Test that missing columns raise ValidationError."""
        ds = Dataset.from_dict({"instruction": ["a"]})
        with pytest.raises(ValidationError, match="Missing required columns"):
            DataValidator.validate_columns(ds, ["instruction", "output"])

    def test_validate_no_nulls_pass(self) -> None:
        """Test that non-null data passes validation."""
        ds = Dataset.from_dict({"text": ["hello", "world"]})
        DataValidator.validate_no_nulls(ds, ["text"])

    def test_get_stats(self) -> None:
        """Test dataset statistics computation."""
        ds = Dataset.from_dict({"a": [1, 2], "b": ["x", "y"]})
        stats = DataValidator.get_stats(ds)
        assert stats["num_rows"] == 2
        assert stats["num_columns"] == 2
