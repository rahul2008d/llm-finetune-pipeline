"""Unit tests for data.registry – DatasetRegistry."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.data.registry import DatasetRegistry, _matches, _resolve
from src.data.schemas import (
    DatasetLineage,
    DatasetManifest,
    DatasetStatistics,
    RegisteredDataset,
)


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════

def _stats(**overrides) -> DatasetStatistics:
    defaults = {
        "total_samples": 1000,
        "avg_input_tokens": 120.5,
        "avg_output_tokens": 45.2,
        "max_input_tokens": 512,
        "max_output_tokens": 256,
        "token_distribution_percentiles": {
            "p50": 100.0, "p90": 200.0, "p95": 250.0, "p99": 300.0,
        },
    }
    defaults.update(overrides)
    return DatasetStatistics(**defaults)


def _manifest(
    name: str = "my-dataset",
    version: str = "1.0.0",
    **overrides,
) -> DatasetManifest:
    defaults = {
        "name": name,
        "version": version,
        "format": "instruction",
        "source_path": "data/processed/my-dataset",
        "num_samples": 1000,
        "sha256_checksum": "abc123",
        "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
        "statistics": _stats(),
    }
    defaults.update(overrides)
    return DatasetManifest(**defaults)


def _lineage(**overrides) -> DatasetLineage:
    defaults = {
        "source_uri": "s3://bucket/raw/data.jsonl",
        "source_format": "jsonl",
        "pipeline_config_hash": "sha256abc",
        "pipeline_version": "a1b2c3d",
        "transformations_applied": ["clean", "tokenize", "deduplicate"],
        "parent_dataset_id": None,
        "pii_scan_result": "clean",
        "created_at": datetime(2026, 1, 15, tzinfo=timezone.utc),
    }
    defaults.update(overrides)
    return DatasetLineage(**defaults)


def _mock_s3_empty() -> MagicMock:
    """Return a mock S3 client with an empty registry."""
    mock_s3 = MagicMock()
    no_such_key = type("NoSuchKey", (Exception,), {})
    mock_s3.exceptions.NoSuchKey = no_such_key
    mock_s3.get_object.side_effect = no_such_key("not found")
    mock_s3.put_object.return_value = {}
    return mock_s3


def _mock_s3_with_data(data: dict) -> MagicMock:
    """Return a mock S3 client that returns *data* as the registry."""
    mock_s3 = MagicMock()
    no_such_key = type("NoSuchKey", (Exception,), {})
    mock_s3.exceptions.NoSuchKey = no_such_key

    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps(data, default=str).encode("utf-8")
    mock_s3.get_object.return_value = {
        "Body": body_mock,
        "ETag": '"etag123"',
        "ResponseMetadata": {"HTTPHeaders": {"etag": '"etag123"'}},
    }
    mock_s3.put_object.return_value = {}
    return mock_s3


def _make_registry(mock_s3: MagicMock) -> DatasetRegistry:
    """Create a DatasetRegistry wired to the given mock S3 client."""
    with patch("src.data.registry.boto3") as mock_boto:
        mock_boto.client.return_value = mock_s3
        reg = DatasetRegistry(bucket="test-bucket")
    return reg


# ══════════════════════════════════════════════
# Constructor
# ══════════════════════════════════════════════


class TestConstructor:
    def test_requires_bucket(self) -> None:
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("src.data.registry.boto3"),
            pytest.raises(ValueError, match="Registry bucket"),
        ):
            DatasetRegistry()

    def test_bucket_from_env(self) -> None:
        with (
            patch.dict("os.environ", {"REGISTRY_BUCKET": "env-bucket"}),
            patch("src.data.registry.boto3") as mock_boto,
        ):
            mock_boto.client.return_value = MagicMock()
            reg = DatasetRegistry()
            assert reg._bucket == "env-bucket"

    def test_explicit_bucket_wins(self) -> None:
        with (
            patch.dict("os.environ", {"REGISTRY_BUCKET": "env-bucket"}),
            patch("src.data.registry.boto3") as mock_boto,
        ):
            mock_boto.client.return_value = MagicMock()
            reg = DatasetRegistry(bucket="explicit-bucket")
            assert reg._bucket == "explicit-bucket"


# ══════════════════════════════════════════════
# register
# ══════════════════════════════════════════════


class TestRegister:
    def test_returns_uuid(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        dataset_id = reg.register(
            _manifest(), _lineage(), registered_by="tester",
        )
        # UUID-v4 format
        assert len(dataset_id) == 36
        assert dataset_id.count("-") == 4

    def test_writes_to_s3(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        reg.register(_manifest(), _lineage(), registered_by="tester")

        mock_s3.put_object.assert_called_once()
        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "registry/datasets.json"
        assert call_kwargs["ContentType"] == "application/json"

    def test_stored_entry_has_all_fields(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        dataset_id = reg.register(
            _manifest(), _lineage(), registered_by="alice",
        )

        body = mock_s3.put_object.call_args.kwargs["Body"]
        written = json.loads(body.decode("utf-8"))
        entry = written[dataset_id]

        assert entry["dataset_id"] == dataset_id
        assert entry["manifest"]["name"] == "my-dataset"
        assert entry["lineage"]["source_uri"] == "s3://bucket/raw/data.jsonl"
        assert entry["registered_by"] == "alice"
        assert "registered_at" in entry

    def test_appends_to_existing_registry(self) -> None:
        existing = {
            "old-id": json.loads(
                RegisteredDataset(
                    dataset_id="old-id",
                    manifest=_manifest(),
                    lineage=_lineage(),
                    train_path="data/processed/my-dataset/train",
                    registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                    registered_by="prev",
                ).model_dump_json(),
            ),
        }
        mock_s3 = _mock_s3_with_data(existing)
        reg = _make_registry(mock_s3)

        new_id = reg.register(
            _manifest(name="new-ds"), _lineage(), registered_by="bob",
        )

        body = mock_s3.put_object.call_args.kwargs["Body"]
        written = json.loads(body.decode("utf-8"))
        assert "old-id" in written
        assert new_id in written

    def test_train_path_derived_from_source_path(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        dataset_id = reg.register(
            _manifest(source_path="data/processed/ds"), _lineage(),
        )

        body = mock_s3.put_object.call_args.kwargs["Body"]
        written = json.loads(body.decode("utf-8"))
        entry = written[dataset_id]
        assert entry["train_path"] == "data/processed/ds/train"
        assert entry["validation_path"] == "data/processed/ds/validation"

    def test_default_registered_by_falls_back(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        with patch("src.data.registry.getpass") as mock_gp:
            mock_gp.getuser.return_value = "os-user"
            reg.register(_manifest(), _lineage())

        body = mock_s3.put_object.call_args.kwargs["Body"]
        written = json.loads(body.decode("utf-8"))
        entry = list(written.values())[0]
        assert entry["registered_by"] == "os-user"


# ══════════════════════════════════════════════
# get
# ══════════════════════════════════════════════


class TestGet:
    def test_returns_registered_dataset(self) -> None:
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=_lineage(),
            train_path="data/processed/my-dataset/train",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        data = {"id-1": json.loads(entry.model_dump_json())}
        mock_s3 = _mock_s3_with_data(data)
        reg = _make_registry(mock_s3)

        result = reg.get("id-1")
        assert isinstance(result, RegisteredDataset)
        assert result.dataset_id == "id-1"
        assert result.manifest.name == "my-dataset"
        assert result.train_path == "data/processed/my-dataset/train"

    def test_missing_id_raises_key_error(self) -> None:
        mock_s3 = _mock_s3_with_data({})
        reg = _make_registry(mock_s3)

        with pytest.raises(KeyError, match="Dataset not found"):
            reg.get("nonexistent")

    def test_has_validation_path(self) -> None:
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=_lineage(),
            train_path="data/train",
            validation_path="data/validation",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        data = {"id-1": json.loads(entry.model_dump_json())}
        mock_s3 = _mock_s3_with_data(data)
        reg = _make_registry(mock_s3)

        result = reg.get("id-1")
        assert result.validation_path == "data/validation"


# ══════════════════════════════════════════════
# search
# ══════════════════════════════════════════════


class TestSearch:
    def _seed_registry(self) -> dict:
        entries = {}
        for i, (name, fmt) in enumerate(
            [("ds-a", "instruction"), ("ds-b", "conversation"), ("ds-a", "instruction")],
        ):
            entry = RegisteredDataset(
                dataset_id=f"id-{i}",
                manifest=_manifest(name=name, version=f"{i}.0.0", format=fmt),
                lineage=_lineage(),
                train_path=f"data/{name}/train",
                registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                registered_by="tester",
            )
            entries[f"id-{i}"] = json.loads(entry.model_dump_json())
        return entries

    def test_filter_by_manifest_name(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({"manifest.name": "ds-a"})
        assert len(results) == 2
        assert all(r.manifest.name == "ds-a" for r in results)

    def test_filter_by_manifest_format(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({"manifest.format": "conversation"})
        assert len(results) == 1
        assert results[0].manifest.format == "conversation"

    def test_no_matches_returns_empty(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({"manifest.name": "nonexistent"})
        assert results == []

    def test_multiple_filters_anded(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({
            "manifest.name": "ds-a",
            "manifest.version": "0.0.0",
        })
        assert len(results) == 1
        assert results[0].dataset_id == "id-0"

    def test_empty_filters_returns_all(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({})
        assert len(results) == 3

    def test_filter_by_top_level_field(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({"registered_by": "tester"})
        assert len(results) == 3

    def test_filter_by_lineage_field(self) -> None:
        data = self._seed_registry()
        reg = _make_registry(_mock_s3_with_data(data))

        results = reg.search({"lineage.pii_scan_result": "clean"})
        assert len(results) == 3


# ══════════════════════════════════════════════
# get_lineage
# ══════════════════════════════════════════════


class TestGetLineage:
    def test_returns_lineage(self) -> None:
        lin = _lineage(source_uri="s3://src/raw.jsonl")
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=lin,
            train_path="data/train",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        data = {"id-1": json.loads(entry.model_dump_json())}
        reg = _make_registry(_mock_s3_with_data(data))

        result = reg.get_lineage("id-1")
        assert isinstance(result, DatasetLineage)
        assert result.source_uri == "s3://src/raw.jsonl"
        assert result.pii_scan_result == "clean"
        assert result.transformations_applied == [
            "clean", "tokenize", "deduplicate",
        ]

    def test_missing_raises_key_error(self) -> None:
        reg = _make_registry(_mock_s3_with_data({}))

        with pytest.raises(KeyError, match="Dataset not found"):
            reg.get_lineage("nope")


# ══════════════════════════════════════════════
# S3 optimistic locking
# ══════════════════════════════════════════════


class TestOptimisticLocking:
    def test_first_write_no_etag(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)
        reg.register(_manifest(), _lineage())

        call_kwargs = mock_s3.put_object.call_args.kwargs
        # No expected-etag metadata on first write (etag is None)
        assert "Metadata" not in call_kwargs or call_kwargs.get("Metadata", {}).get("expected-etag") is None

    def test_subsequent_write_includes_etag(self) -> None:
        data = {"old-id": json.loads(
            RegisteredDataset(
                dataset_id="old-id",
                manifest=_manifest(),
                lineage=_lineage(),
                train_path="t",
                registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                registered_by="x",
            ).model_dump_json(),
        )}
        mock_s3 = _mock_s3_with_data(data)
        reg = _make_registry(mock_s3)
        reg.register(_manifest(), _lineage())

        call_kwargs = mock_s3.put_object.call_args.kwargs
        assert call_kwargs["Metadata"]["expected-etag"] == "etag123"

    def test_write_failure_raises_runtime_error(self) -> None:
        mock_s3 = _mock_s3_empty()
        mock_s3.put_object.side_effect = Exception("Conflict")
        reg = _make_registry(mock_s3)

        with pytest.raises(RuntimeError, match="concurrent update"):
            reg.register(_manifest(), _lineage())


# ══════════════════════════════════════════════
# _read_registry
# ══════════════════════════════════════════════


class TestReadRegistry:
    def test_empty_when_no_file(self) -> None:
        mock_s3 = _mock_s3_empty()
        reg = _make_registry(mock_s3)

        data, etag = reg._read_registry()
        assert data == {}
        assert etag is None

    def test_returns_data_and_etag(self) -> None:
        stored = {"k": "v"}
        mock_s3 = _mock_s3_with_data(stored)
        reg = _make_registry(mock_s3)

        data, etag = reg._read_registry()
        assert data == stored
        assert etag is not None


# ══════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════


class TestResolve:
    def test_top_level(self) -> None:
        assert _resolve({"a": 1}, "a") == 1

    def test_nested(self) -> None:
        assert _resolve({"a": {"b": {"c": 3}}}, "a.b.c") == 3

    def test_missing_returns_none(self) -> None:
        assert _resolve({"a": 1}, "b") is None

    def test_deep_missing_returns_none(self) -> None:
        assert _resolve({"a": {"b": 2}}, "a.c") is None


class TestMatches:
    def test_empty_filters_always_match(self) -> None:
        assert _matches({"a": 1}, {}) is True

    def test_matching_filter(self) -> None:
        assert _matches({"a": 1, "b": 2}, {"a": 1}) is True

    def test_non_matching_filter(self) -> None:
        assert _matches({"a": 1}, {"a": 2}) is False

    def test_nested_filter(self) -> None:
        assert _matches({"x": {"y": "z"}}, {"x.y": "z"}) is True


# ══════════════════════════════════════════════
# Model tests
# ══════════════════════════════════════════════


class TestDatasetLineage:
    def test_all_fields(self) -> None:
        lin = _lineage(
            parent_dataset_id="parent-123",
            pii_scan_result="redacted",
        )
        assert lin.source_uri == "s3://bucket/raw/data.jsonl"
        assert lin.source_format == "jsonl"
        assert lin.pipeline_config_hash == "sha256abc"
        assert lin.pipeline_version == "a1b2c3d"
        assert lin.transformations_applied == [
            "clean", "tokenize", "deduplicate",
        ]
        assert lin.parent_dataset_id == "parent-123"
        assert lin.pii_scan_result == "redacted"

    def test_default_parent_is_none(self) -> None:
        lin = _lineage()
        assert lin.parent_dataset_id is None

    def test_default_pii_scan_result(self) -> None:
        lin = DatasetLineage(
            source_uri="s3://b/d",
            source_format="jsonl",
            pipeline_config_hash="h",
            pipeline_version="v",
            transformations_applied=[],
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        assert lin.pii_scan_result == "not_scanned"

    def test_serialization_roundtrip(self) -> None:
        lin = _lineage()
        j = lin.model_dump_json()
        restored = DatasetLineage.model_validate_json(j)
        assert restored.source_uri == lin.source_uri
        assert restored.pipeline_config_hash == lin.pipeline_config_hash


class TestRegisteredDataset:
    def test_all_fields(self) -> None:
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=_lineage(),
            train_path="data/train",
            validation_path="data/val",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        assert entry.dataset_id == "id-1"
        assert entry.train_path == "data/train"
        assert entry.validation_path == "data/val"
        assert entry.registered_by == "alice"

    def test_validation_path_optional(self) -> None:
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=_lineage(),
            train_path="data/train",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        assert entry.validation_path is None

    def test_serialization_roundtrip(self) -> None:
        entry = RegisteredDataset(
            dataset_id="id-1",
            manifest=_manifest(),
            lineage=_lineage(),
            train_path="data/train",
            validation_path="data/val",
            registered_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            registered_by="alice",
        )
        j = entry.model_dump_json()
        restored = RegisteredDataset.model_validate_json(j)
        assert restored.dataset_id == entry.dataset_id
        assert restored.manifest.name == entry.manifest.name
        assert restored.lineage.source_uri == entry.lineage.source_uri
