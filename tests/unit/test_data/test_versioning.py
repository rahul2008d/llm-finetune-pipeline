"""Unit tests for data.versioning – DatasetVersionManager."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.schemas import DatasetDiff, DatasetManifest, DatasetStatistics
from src.data.versioning import DatasetVersionManager


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


# ══════════════════════════════════════════════
# version_dataset
# ══════════════════════════════════════════════


class TestVersionDataset:
    def test_writes_manifest_json(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifest = _manifest(source_path="data/processed/my-dataset")

        # Create dataset dir + .dvc file for the hash extraction
        dataset_dir = tmp_path / "data" / "processed" / "my-dataset"
        dataset_dir.mkdir(parents=True)
        dvc_file = Path(f"{dataset_dir}.dvc")
        dvc_file.write_text("outs:\n- md5: abc456\n  path: my-dataset\n")

        with patch.object(mgr, "_run_dvc"):
            result = mgr.version_dataset(manifest, "initial version")

        manifest_path = (
            tmp_path / "data" / "manifests" / "my-dataset_v1.0.0.json"
        )
        assert manifest_path.exists()
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert loaded["name"] == "my-dataset"
        assert loaded["version"] == "1.0.0"
        assert loaded["num_samples"] == 1000

    def test_calls_dvc_add_and_push(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifest = _manifest(source_path="data/processed/ds")

        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        dvc_file = Path(f"{dataset_dir}.dvc")
        dvc_file.write_text("outs:\n- md5: hash789\n  path: ds\n")

        with patch.object(mgr, "_run_dvc") as mock_dvc:
            mgr.version_dataset(manifest, "v1")

        calls = [c.args[0] for c in mock_dvc.call_args_list]
        assert ["add", str(dataset_dir)] in calls
        assert ["push"] in calls

    def test_returns_dvc_hash(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifest = _manifest(source_path="data/processed/ds")

        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        dvc_file = Path(f"{dataset_dir}.dvc")
        dvc_file.write_text("outs:\n- md5: expectedhash\n  path: ds\n")

        with patch.object(mgr, "_run_dvc"):
            result = mgr.version_dataset(manifest, "msg")

        assert result == "expectedhash"

    def test_creates_manifests_directory(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifest = _manifest(source_path="data/processed/ds")

        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        dvc_file = Path(f"{dataset_dir}.dvc")
        dvc_file.write_text("outs:\n- md5: h\n  path: ds\n")

        with patch.object(mgr, "_run_dvc"):
            mgr.version_dataset(manifest, "msg")

        assert (tmp_path / "data" / "manifests").is_dir()


# ══════════════════════════════════════════════
# checkout_dataset
# ══════════════════════════════════════════════


class TestCheckoutDataset:
    def test_returns_dataset_path(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifest = _manifest(source_path="data/processed/ds")

        # Write manifest
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)
        mp = manifests_dir / "my-dataset_v1.0.0.json"
        mp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        # Create dataset dir with matching checksum
        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "file.txt").write_text("hello")

        # Compute real checksum and update manifest
        real_hash = mgr._compute_directory_sha256(dataset_dir)
        manifest_fixed = _manifest(
            source_path="data/processed/ds", sha256_checksum=real_hash,
        )
        mp.write_text(
            manifest_fixed.model_dump_json(indent=2), encoding="utf-8",
        )

        with patch.object(mgr, "_run_dvc"):
            result = mgr.checkout_dataset("my-dataset", "1.0.0")

        assert result == dataset_dir

    def test_calls_dvc_pull(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)

        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)
        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "f.bin").write_bytes(b"data")

        real_hash = mgr._compute_directory_sha256(dataset_dir)
        manifest = _manifest(
            source_path="data/processed/ds", sha256_checksum=real_hash,
        )
        mp = manifests_dir / "my-dataset_v1.0.0.json"
        mp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        with patch.object(mgr, "_run_dvc") as mock_dvc:
            mgr.checkout_dataset("my-dataset", "1.0.0")

        mock_dvc.assert_called_once_with(["pull"])

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)

        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)
        dataset_dir = tmp_path / "data" / "processed" / "ds"
        dataset_dir.mkdir(parents=True)
        (dataset_dir / "f.bin").write_bytes(b"data")

        manifest = _manifest(
            source_path="data/processed/ds", sha256_checksum="wrong_hash",
        )
        mp = manifests_dir / "my-dataset_v1.0.0.json"
        mp.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

        with (
            patch.object(mgr, "_run_dvc"),
            pytest.raises(ValueError, match="Checksum mismatch"),
        ):
            mgr.checkout_dataset("my-dataset", "1.0.0")

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            mgr.checkout_dataset("no-such-ds", "1.0.0")


# ══════════════════════════════════════════════
# list_versions
# ══════════════════════════════════════════════


class TestListVersions:
    def test_returns_empty_when_no_manifests(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        assert mgr.list_versions("anything") == []

    def test_lists_matching_manifests(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        for v in ("1.0.0", "1.1.0", "2.0.0"):
            m = _manifest(version=v)
            (manifests_dir / f"my-dataset_v{v}.json").write_text(
                m.model_dump_json(indent=2), encoding="utf-8",
            )

        # Also write a manifest for a different dataset
        other = _manifest(name="other-ds", version="1.0.0")
        (manifests_dir / "other-ds_v1.0.0.json").write_text(
            other.model_dump_json(indent=2), encoding="utf-8",
        )

        result = mgr.list_versions("my-dataset")
        assert len(result) == 3
        assert [m.version for m in result] == ["1.0.0", "1.1.0", "2.0.0"]

    def test_does_not_include_other_datasets(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        other = _manifest(name="other-ds", version="3.0.0")
        (manifests_dir / "other-ds_v3.0.0.json").write_text(
            other.model_dump_json(indent=2), encoding="utf-8",
        )

        assert mgr.list_versions("my-dataset") == []

    def test_returns_sorted_by_filename(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        # Write out of order
        for v in ("2.0.0", "1.0.0"):
            m = _manifest(version=v)
            (manifests_dir / f"my-dataset_v{v}.json").write_text(
                m.model_dump_json(indent=2), encoding="utf-8",
            )

        result = mgr.list_versions("my-dataset")
        assert result[0].version == "1.0.0"
        assert result[1].version == "2.0.0"


# ══════════════════════════════════════════════
# diff_versions
# ══════════════════════════════════════════════


class TestDiffVersions:
    def test_samples_added(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        m1 = _manifest(version="1.0.0", num_samples=1000)
        m2 = _manifest(version="2.0.0", num_samples=1500)
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m1.model_dump_json(indent=2), encoding="utf-8",
        )
        (manifests_dir / "my-dataset_v2.0.0.json").write_text(
            m2.model_dump_json(indent=2), encoding="utf-8",
        )

        diff = mgr.diff_versions("my-dataset", "1.0.0", "2.0.0")
        assert diff.samples_added == 500
        assert diff.samples_removed == 0

    def test_samples_removed(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        m1 = _manifest(version="1.0.0", num_samples=2000)
        m2 = _manifest(version="2.0.0", num_samples=1800)
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m1.model_dump_json(indent=2), encoding="utf-8",
        )
        (manifests_dir / "my-dataset_v2.0.0.json").write_text(
            m2.model_dump_json(indent=2), encoding="utf-8",
        )

        diff = mgr.diff_versions("my-dataset", "1.0.0", "2.0.0")
        assert diff.samples_added == 0
        assert diff.samples_removed == 200

    def test_diff_preserves_stats(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        s1 = _stats(avg_input_tokens=100.0)
        s2 = _stats(avg_input_tokens=150.0)
        m1 = _manifest(version="1.0.0", statistics=s1)
        m2 = _manifest(version="2.0.0", statistics=s2)
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m1.model_dump_json(indent=2), encoding="utf-8",
        )
        (manifests_dir / "my-dataset_v2.0.0.json").write_text(
            m2.model_dump_json(indent=2), encoding="utf-8",
        )

        diff = mgr.diff_versions("my-dataset", "1.0.0", "2.0.0")
        assert diff.stats_old.avg_input_tokens == 100.0
        assert diff.stats_new.avg_input_tokens == 150.0

    def test_diff_same_version(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        m = _manifest(version="1.0.0", num_samples=1000)
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m.model_dump_json(indent=2), encoding="utf-8",
        )

        diff = mgr.diff_versions("my-dataset", "1.0.0", "1.0.0")
        assert diff.samples_added == 0
        assert diff.samples_removed == 0

    def test_diff_missing_version_raises(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        m = _manifest(version="1.0.0")
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m.model_dump_json(indent=2), encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            mgr.diff_versions("my-dataset", "1.0.0", "9.9.9")

    def test_diff_returns_dataset_diff_model(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        manifests_dir = tmp_path / "data" / "manifests"
        manifests_dir.mkdir(parents=True)

        m1 = _manifest(version="1.0.0")
        m2 = _manifest(version="2.0.0")
        (manifests_dir / "my-dataset_v1.0.0.json").write_text(
            m1.model_dump_json(indent=2), encoding="utf-8",
        )
        (manifests_dir / "my-dataset_v2.0.0.json").write_text(
            m2.model_dump_json(indent=2), encoding="utf-8",
        )

        diff = mgr.diff_versions("my-dataset", "1.0.0", "2.0.0")
        assert isinstance(diff, DatasetDiff)
        assert diff.name == "my-dataset"
        assert diff.version_old == "1.0.0"
        assert diff.version_new == "2.0.0"


# ══════════════════════════════════════════════
# _run_dvc
# ══════════════════════════════════════════════


class TestRunDvc:
    def test_raises_on_failure(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "dvc error"

        with patch("src.data.versioning.subprocess.run", return_value=mock_result):
            with pytest.raises(RuntimeError, match="DVC command failed"):
                mgr._run_dvc(["status"])

    def test_passes_cwd_as_repo_root(self, tmp_path: Path) -> None:
        mgr = DatasetVersionManager(repo_root=tmp_path)
        mock_result = MagicMock()
        mock_result.returncode = 0

        with patch("src.data.versioning.subprocess.run", return_value=mock_result) as mock_run:
            mgr._run_dvc(["status"])

        _, kwargs = mock_run.call_args
        assert kwargs["cwd"] == str(tmp_path)


# ══════════════════════════════════════════════
# _extract_dvc_hash
# ══════════════════════════════════════════════


class TestExtractDvcHash:
    def test_extracts_md5(self, tmp_path: Path) -> None:
        dvc_file = tmp_path / "dataset.dvc"
        dvc_file.write_text("outs:\n- md5: abc123def\n  path: dataset\n")
        assert DatasetVersionManager._extract_dvc_hash(dvc_file) == "abc123def"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetVersionManager._extract_dvc_hash(tmp_path / "nope.dvc")

    def test_empty_outs_raises(self, tmp_path: Path) -> None:
        dvc_file = tmp_path / "dataset.dvc"
        dvc_file.write_text("outs: []\n")
        with pytest.raises(ValueError, match="No outputs"):
            DatasetVersionManager._extract_dvc_hash(dvc_file)


# ══════════════════════════════════════════════
# _compute_directory_sha256
# ══════════════════════════════════════════════


class TestComputeDirectorySha256:
    def test_deterministic(self, tmp_path: Path) -> None:
        d = tmp_path / "ds"
        d.mkdir()
        (d / "a.txt").write_text("hello")
        (d / "b.txt").write_text("world")

        h1 = DatasetVersionManager._compute_directory_sha256(d)
        h2 = DatasetVersionManager._compute_directory_sha256(d)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path: Path) -> None:
        d1 = tmp_path / "ds1"
        d1.mkdir()
        (d1 / "a.txt").write_text("hello")

        d2 = tmp_path / "ds2"
        d2.mkdir()
        (d2 / "a.txt").write_text("different")

        h1 = DatasetVersionManager._compute_directory_sha256(d1)
        h2 = DatasetVersionManager._compute_directory_sha256(d2)
        assert h1 != h2

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            DatasetVersionManager._compute_directory_sha256(
                tmp_path / "nonexistent",
            )

    def test_includes_filenames_in_hash(self, tmp_path: Path) -> None:
        d1 = tmp_path / "ds1"
        d1.mkdir()
        (d1 / "a.txt").write_text("same")

        d2 = tmp_path / "ds2"
        d2.mkdir()
        (d2 / "b.txt").write_text("same")

        h1 = DatasetVersionManager._compute_directory_sha256(d1)
        h2 = DatasetVersionManager._compute_directory_sha256(d2)
        assert h1 != h2


# ══════════════════════════════════════════════
# DatasetDiff model
# ══════════════════════════════════════════════


class TestDatasetDiff:
    def test_model_fields(self) -> None:
        diff = DatasetDiff(
            name="ds",
            version_old="1.0.0",
            version_new="2.0.0",
            samples_added=100,
            samples_removed=50,
            stats_old=_stats(),
            stats_new=_stats(total_samples=1050),
        )
        assert diff.name == "ds"
        assert diff.samples_added == 100
        assert diff.samples_removed == 50

    def test_serialization_roundtrip(self) -> None:
        diff = DatasetDiff(
            name="ds",
            version_old="1.0.0",
            version_new="2.0.0",
            samples_added=10,
            samples_removed=0,
            stats_old=_stats(),
            stats_new=_stats(),
        )
        j = diff.model_dump_json()
        restored = DatasetDiff.model_validate_json(j)
        assert restored.name == diff.name
        assert restored.version_old == diff.version_old
        assert restored.samples_added == diff.samples_added
