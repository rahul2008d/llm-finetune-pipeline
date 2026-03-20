"""Exhaustive unit tests for data schema Pydantic models."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.data.schemas import (
    ConversationSample,
    ConversationTurn,
    DatasetManifest,
    DatasetStatistics,
    RawSample,
)


# ══════════════════════════════════════════════
# RawSample
# ══════════════════════════════════════════════

class TestRawSample:

    def test_valid_minimal(self):
        s = RawSample(instruction="Summarise this.", output="Done.")
        assert s.instruction == "Summarise this."
        assert s.output == "Done."
        assert s.input is None
        assert s.system_prompt is None
        assert s.metadata == {}

    def test_valid_full(self):
        s = RawSample(
            instruction="Translate",
            input="Hello world",
            output="Hola mundo",
            system_prompt="You are a translator.",
            metadata={"lang": "es"},
        )
        assert s.input == "Hello world"
        assert s.system_prompt == "You are a translator."
        assert s.metadata == {"lang": "es"}

    def test_instruction_empty_string_rejected(self):
        with pytest.raises(ValidationError, match="instruction"):
            RawSample(instruction="", output="ok")

    def test_output_empty_string_rejected(self):
        with pytest.raises(ValidationError, match="output"):
            RawSample(instruction="do it", output="")

    def test_instruction_too_long(self):
        with pytest.raises(ValidationError, match="instruction"):
            RawSample(instruction="x" * 4097, output="ok")

    def test_output_too_long(self):
        with pytest.raises(ValidationError, match="output"):
            RawSample(instruction="do it", output="x" * 8193)

    def test_input_too_long(self):
        with pytest.raises(ValidationError, match="input"):
            RawSample(instruction="do it", input="x" * 8193, output="ok")

    def test_system_prompt_too_long(self):
        with pytest.raises(ValidationError, match="system_prompt"):
            RawSample(instruction="do it", output="ok", system_prompt="x" * 2049)

    def test_instruction_at_max_length(self):
        s = RawSample(instruction="x" * 4096, output="ok")
        assert len(s.instruction) == 4096

    def test_output_at_max_length(self):
        s = RawSample(instruction="do it", output="x" * 8192)
        assert len(s.output) == 8192

    def test_metadata_default_factory_independent(self):
        a = RawSample(instruction="a", output="a")
        b = RawSample(instruction="b", output="b")
        a.metadata["key"] = "val"
        assert "key" not in b.metadata

    def test_missing_instruction_rejected(self):
        with pytest.raises(ValidationError):
            RawSample(output="ok")  # type: ignore[call-arg]

    def test_missing_output_rejected(self):
        with pytest.raises(ValidationError):
            RawSample(instruction="do it")  # type: ignore[call-arg]


# ══════════════════════════════════════════════
# ConversationTurn
# ══════════════════════════════════════════════

class TestConversationTurn:

    @pytest.mark.parametrize("role", ["system", "user", "assistant"])
    def test_valid_roles(self, role):
        t = ConversationTurn(role=role, content="hello")
        assert t.role == role

    def test_invalid_role(self):
        with pytest.raises(ValidationError, match="role"):
            ConversationTurn(role="bot", content="hello")  # type: ignore[arg-type]

    def test_empty_content_rejected(self):
        with pytest.raises(ValidationError, match="content"):
            ConversationTurn(role="user", content="")

    def test_missing_content_rejected(self):
        with pytest.raises(ValidationError):
            ConversationTurn(role="user")  # type: ignore[call-arg]

    def test_missing_role_rejected(self):
        with pytest.raises(ValidationError):
            ConversationTurn(content="hi")  # type: ignore[call-arg]


# ══════════════════════════════════════════════
# ConversationSample
# ══════════════════════════════════════════════

class TestConversationSample:

    def test_valid_user_assistant(self):
        cs = ConversationSample(conversations=[
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ])
        assert len(cs.conversations) == 2

    def test_valid_system_user_assistant(self):
        cs = ConversationSample(conversations=[
            ConversationTurn(role="system", content="You are helpful."),
            ConversationTurn(role="user", content="Hi"),
            ConversationTurn(role="assistant", content="Hello!"),
        ])
        assert len(cs.conversations) == 3

    def test_valid_multi_turn(self):
        cs = ConversationSample(conversations=[
            ConversationTurn(role="system", content="Be concise."),
            ConversationTurn(role="user", content="What is 2+2?"),
            ConversationTurn(role="assistant", content="4"),
            ConversationTurn(role="user", content="And 3+3?"),
            ConversationTurn(role="assistant", content="6"),
        ])
        assert len(cs.conversations) == 5

    def test_too_few_turns_rejected(self):
        with pytest.raises(ValidationError, match="conversations"):
            ConversationSample(conversations=[
                ConversationTurn(role="user", content="Hi"),
            ])

    def test_empty_conversations_rejected(self):
        with pytest.raises(ValidationError, match="conversations"):
            ConversationSample(conversations=[])

    def test_first_turn_assistant_rejected(self):
        with pytest.raises(ValidationError, match="First turn"):
            ConversationSample(conversations=[
                ConversationTurn(role="assistant", content="Hi"),
                ConversationTurn(role="user", content="Hello"),
            ])

    def test_last_turn_user_rejected(self):
        with pytest.raises(ValidationError, match="Last turn"):
            ConversationSample(conversations=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="user", content="Hello?"),
            ])

    def test_non_alternating_rejected(self):
        with pytest.raises(ValidationError, match="Turn"):
            ConversationSample(conversations=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="user", content="Hello?"),
                ConversationTurn(role="assistant", content="Hey"),
            ])

    def test_double_assistant_rejected(self):
        with pytest.raises(ValidationError, match="Turn"):
            ConversationSample(conversations=[
                ConversationTurn(role="user", content="Hi"),
                ConversationTurn(role="assistant", content="Hello"),
                ConversationTurn(role="assistant", content="Again"),
                ConversationTurn(role="user", content="What?"),
                ConversationTurn(role="assistant", content="Nothing"),
            ])

    def test_system_not_followed_by_user_rejected(self):
        with pytest.raises(ValidationError, match="Turn"):
            ConversationSample(conversations=[
                ConversationTurn(role="system", content="Be helpful"),
                ConversationTurn(role="assistant", content="Ok"),
            ])

    def test_multi_turn_without_system_valid(self):
        cs = ConversationSample(conversations=[
            ConversationTurn(role="user", content="Q1"),
            ConversationTurn(role="assistant", content="A1"),
            ConversationTurn(role="user", content="Q2"),
            ConversationTurn(role="assistant", content="A2"),
        ])
        assert len(cs.conversations) == 4


# ══════════════════════════════════════════════
# DatasetStatistics
# ══════════════════════════════════════════════

class TestDatasetStatistics:

    def test_valid(self):
        ds = DatasetStatistics(
            total_samples=1000,
            avg_input_tokens=150.5,
            avg_output_tokens=200.3,
            max_input_tokens=512,
            max_output_tokens=1024,
            token_distribution_percentiles={"p50": 100, "p90": 300, "p95": 400, "p99": 500},
        )
        assert ds.total_samples == 1000
        assert ds.token_distribution_percentiles["p99"] == 500

    def test_missing_field_rejected(self):
        with pytest.raises(ValidationError):
            DatasetStatistics(
                total_samples=1000,
                avg_input_tokens=150.5,
                # missing avg_output_tokens and others
            )  # type: ignore[call-arg]

    def test_wrong_type_rejected(self):
        with pytest.raises(ValidationError):
            DatasetStatistics(
                total_samples="not_a_number",  # type: ignore[arg-type]
                avg_input_tokens=150.5,
                avg_output_tokens=200.3,
                max_input_tokens=512,
                max_output_tokens=1024,
                token_distribution_percentiles={},
            )


# ══════════════════════════════════════════════
# DatasetManifest
# ══════════════════════════════════════════════

class TestDatasetManifest:

    @pytest.fixture
    def valid_stats(self) -> DatasetStatistics:
        return DatasetStatistics(
            total_samples=500,
            avg_input_tokens=120.0,
            avg_output_tokens=180.0,
            max_input_tokens=400,
            max_output_tokens=800,
            token_distribution_percentiles={"p50": 100, "p90": 250, "p95": 350, "p99": 400},
        )

    def test_valid_instruction_format(self, valid_stats):
        m = DatasetManifest(
            name="my-dataset",
            version="1.0.0",
            format="instruction",
            source_path="s3://bucket/data.jsonl",
            num_samples=500,
            sha256_checksum="abc123def456",
            created_at=datetime(2025, 1, 1, tzinfo=timezone.utc),
            statistics=valid_stats,
        )
        assert m.format == "instruction"
        assert m.version == "1.0.0"

    def test_valid_conversation_format(self, valid_stats):
        m = DatasetManifest(
            name="chat-data",
            version="2.1.0",
            format="conversation",
            source_path="/data/chat.jsonl",
            num_samples=1000,
            sha256_checksum="deadbeef",
            created_at=datetime.now(tz=timezone.utc),
            statistics=valid_stats,
        )
        assert m.format == "conversation"

    def test_valid_completion_format(self, valid_stats):
        m = DatasetManifest(
            name="completions",
            version="0.1.0",
            format="completion",
            source_path="/data/comp.jsonl",
            num_samples=200,
            sha256_checksum="aabbcc",
            created_at=datetime.now(tz=timezone.utc),
            statistics=valid_stats,
        )
        assert m.format == "completion"

    def test_invalid_format_rejected(self, valid_stats):
        with pytest.raises(ValidationError, match="format"):
            DatasetManifest(
                name="bad",
                version="1.0.0",
                format="qa",  # type: ignore[arg-type]
                source_path="/data",
                num_samples=10,
                sha256_checksum="abc",
                created_at=datetime.now(tz=timezone.utc),
                statistics=valid_stats,
            )

    def test_missing_statistics_rejected(self):
        with pytest.raises(ValidationError):
            DatasetManifest(
                name="bad",
                version="1.0.0",
                format="instruction",
                source_path="/data",
                num_samples=10,
                sha256_checksum="abc",
                created_at=datetime.now(tz=timezone.utc),
            )  # type: ignore[call-arg]

    def test_nested_statistics_validated(self):
        with pytest.raises(ValidationError):
            DatasetManifest(
                name="bad",
                version="1.0.0",
                format="instruction",
                source_path="/data",
                num_samples=10,
                sha256_checksum="abc",
                created_at=datetime.now(tz=timezone.utc),
                statistics={"total_samples": "wrong"},  # type: ignore[arg-type]
            )

    def test_datetime_parsing(self, valid_stats):
        m = DatasetManifest(
            name="test",
            version="1.0.0",
            format="instruction",
            source_path="/data",
            num_samples=1,
            sha256_checksum="x",
            created_at="2025-06-15T10:30:00Z",  # type: ignore[arg-type]
            statistics=valid_stats,
        )
        assert m.created_at.year == 2025
