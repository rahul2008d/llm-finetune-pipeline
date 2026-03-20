"""Unit tests for DataIngester — format detection, loaders, normalisation."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from src.data.ingestion import DataIngester
from src.data.schemas import ConversationSample, ConversationTurn, RawSample


@pytest.fixture
def ingester() -> DataIngester:
    return DataIngester()


# ══════════════════════════════════════════════
# Format detection
# ══════════════════════════════════════════════


class TestDetectFormat:

    def test_jsonl_extension(self):
        assert DataIngester._detect_format("data/train.jsonl") == "jsonl"

    def test_json_extension(self):
        assert DataIngester._detect_format("data/train.json") == "jsonl"

    def test_parquet_extension(self):
        assert DataIngester._detect_format("train.parquet") == "parquet"

    def test_csv_extension(self):
        assert DataIngester._detect_format("data.csv") == "csv"

    def test_s3_uri_jsonl(self):
        assert DataIngester._detect_format("s3://bucket/path/data.jsonl") == "jsonl"

    def test_s3_uri_parquet(self):
        assert DataIngester._detect_format("s3://bucket/key.parquet") == "parquet"

    def test_unknown_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            DataIngester._detect_format("data.txt")

    def test_no_extension_raises(self):
        with pytest.raises(ValueError, match="Cannot detect format"):
            DataIngester._detect_format("mydata")


# ══════════════════════════════════════════════
# JSONL loader
# ══════════════════════════════════════════════


class TestLoadJsonl:

    def test_valid_lines(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        rows = [
            {"instruction": "Say hi", "output": "Hi!"},
            {"instruction": "Say bye", "output": "Bye!"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        result = ingester._load_jsonl(str(p))
        assert len(result) == 2
        assert result[0]["instruction"] == "Say hi"

    def test_blank_lines_skipped(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
        assert len(ingester._load_jsonl(str(p))) == 2

    def test_malformed_lines_skipped(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        p.write_text('{"ok":1}\nNOT JSON\n{"ok":2}\n', encoding="utf-8")
        result = ingester._load_jsonl(str(p))
        assert len(result) == 2

    def test_non_dict_lines_skipped(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        p.write_text('{"ok":1}\n[1,2,3]\n"string"\n', encoding="utf-8")
        result = ingester._load_jsonl(str(p))
        assert len(result) == 1

    def test_empty_file(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        assert ingester._load_jsonl(str(p)) == []


# ══════════════════════════════════════════════
# CSV loader
# ══════════════════════════════════════════════


class TestLoadCsv:

    def test_basic_csv(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "data.csv"
        p.write_text(
            '"instruction","output"\n"Say hi","Hi!"\n"Say bye","Bye!"\n',
            encoding="utf-8",
        )
        result = ingester._load_csv(str(p))
        assert len(result) == 2
        assert result[0]["instruction"] == "Say hi"
        assert result[1]["output"] == "Bye!"

    def test_csv_preserves_empty_string(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "data.csv"
        p.write_text('"col"\n""\n', encoding="utf-8")
        result = ingester._load_csv(str(p))
        assert result[0]["col"] == ""


# ══════════════════════════════════════════════
# Parquet loader
# ══════════════════════════════════════════════


class TestLoadParquet:

    def test_basic_parquet(self, ingester: DataIngester, tmp_path: Path):
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")

        import pyarrow as pa

        table = pa.table(
            {"instruction": ["Say hi", "Say bye"], "output": ["Hi!", "Bye!"]}
        )
        p = tmp_path / "data.parquet"
        pq.write_table(table, str(p))

        result = ingester._load_parquet(str(p))
        assert len(result) == 2
        assert result[0]["instruction"] == "Say hi"


# ══════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════


class TestNormalise:

    def test_instruction_schema(self):
        row = {"instruction": "Do it", "output": "Done"}
        sample = DataIngester._normalise(row, "instruction")
        assert isinstance(sample, RawSample)
        assert sample.instruction == "Do it"

    def test_conversation_schema(self):
        row = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        sample = DataIngester._normalise(row, "conversation")
        assert isinstance(sample, ConversationSample)

    def test_auto_prefers_instruction(self):
        row = {"instruction": "X", "output": "Y"}
        sample = DataIngester._normalise(row, "auto")
        assert isinstance(sample, RawSample)

    def test_auto_falls_back_to_conversation(self):
        row = {
            "conversations": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        }
        sample = DataIngester._normalise(row, "auto")
        assert isinstance(sample, ConversationSample)

    def test_invalid_row_raises(self):
        with pytest.raises(Exception):
            DataIngester._normalise({}, "instruction")


# ══════════════════════════════════════════════
# End-to-end ingest
# ══════════════════════════════════════════════


class TestIngest:

    def test_ingest_jsonl_instruction(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "train.jsonl"
        rows = [
            {"instruction": "Sum it", "output": "42"},
            {"instruction": "Mul it", "output": "7"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

        samples = ingester.ingest(str(p), target_schema="instruction")
        assert len(samples) == 2
        assert all(isinstance(s, RawSample) for s in samples)

    def test_ingest_jsonl_conversation(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "chat.jsonl"
        row = {
            "conversations": [
                {"role": "user", "content": "Hey"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        p.write_text(json.dumps(row) + "\n", encoding="utf-8")

        samples = ingester.ingest(str(p), target_schema="conversation")
        assert len(samples) == 1
        assert isinstance(samples[0], ConversationSample)

    def test_ingest_with_failures_logged(
        self, ingester: DataIngester, tmp_path: Path
    ):
        p = tmp_path / "mixed.jsonl"
        good = {"instruction": "Do", "output": "Done"}
        bad = {"not_a_valid_field": True}
        p.write_text(
            json.dumps(good) + "\n" + json.dumps(bad) + "\n", encoding="utf-8"
        )

        samples = ingester.ingest(str(p), target_schema="instruction")
        assert len(samples) == 1

    def test_ingest_csv(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "data.csv"
        p.write_text(
            '"instruction","output"\n"Say hi","Hi!"\n', encoding="utf-8"
        )
        samples = ingester.ingest(str(p), target_schema="instruction")
        assert len(samples) == 1
        assert isinstance(samples[0], RawSample)

    def test_ingest_explicit_format(self, ingester: DataIngester, tmp_path: Path):
        p = tmp_path / "data.txt"  # unusual extension
        p.write_text(
            json.dumps({"instruction": "X", "output": "Y"}) + "\n",
            encoding="utf-8",
        )
        samples = ingester.ingest(str(p), fmt="jsonl", target_schema="instruction")
        assert len(samples) == 1

    def test_ingest_unsupported_format_raises(self, ingester: DataIngester):
        with pytest.raises(ValueError, match="Unsupported format"):
            ingester.ingest("file.jsonl", fmt="xml")
