"""Unit tests for data.transformation module."""

from datasets import Dataset

from src.data.transformation import DataTransformer


class TestDataTransformer:
    """Tests for DataTransformer."""

    def test_apply_chat_template(self) -> None:
        """Test chat template formatting."""
        ds = Dataset.from_dict({
            "instruction": ["Summarize"],
            "input": ["Text here"],
            "output": ["Summary"],
        })
        result = DataTransformer.apply_chat_template(ds)
        assert "text" in result.column_names
        assert "### Instruction:" in result[0]["text"]
        assert "### Response:" in result[0]["text"]

    def test_filter_by_length(self) -> None:
        """Test length-based filtering."""
        ds = Dataset.from_dict({"text": ["short", "a much longer piece of text"]})
        filtered = DataTransformer.filter_by_length(ds, "text", min_length=10)
        assert len(filtered) == 1

    def test_deduplicate(self) -> None:
        """Test deduplication."""
        ds = Dataset.from_dict({"text": ["hello", "hello", "world"]})
        deduped = DataTransformer.deduplicate(ds, "text")
        assert len(deduped) == 2
