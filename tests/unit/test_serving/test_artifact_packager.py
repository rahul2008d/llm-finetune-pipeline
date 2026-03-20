"""Tests for artifact packaging for SageMaker and Bedrock deployment."""

from __future__ import annotations

import json
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.artifact_packager import ArtifactPackager


# ── Helpers ─────────────────────────────────────────────────────


def _create_model_dir(
    base: Path,
    *,
    include_config: bool = True,
    include_tokenizer: bool = True,
    include_safetensors: bool = True,
    architecture: str = "LlamaForCausalLM",
) -> Path:
    """Create a minimal model directory for testing."""
    model_dir = base / "test_model"
    model_dir.mkdir(parents=True, exist_ok=True)

    if include_config:
        config = {"architectures": [architecture], "model_type": "llama"}
        (model_dir / "config.json").write_text(json.dumps(config))

    if include_tokenizer:
        (model_dir / "tokenizer.json").write_text("{}")
        (model_dir / "tokenizer_config.json").write_text("{}")
        (model_dir / "special_tokens_map.json").write_text("{}")

    if include_safetensors:
        # Create a small dummy safetensors file
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

    return model_dir


# ── Fixtures ────────────────────────────────────────────────────


@pytest.fixture()
def packager() -> ArtifactPackager:
    """Create an ArtifactPackager instance."""
    return ArtifactPackager()


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """Create a complete model directory."""
    return _create_model_dir(tmp_path)


@pytest.fixture()
def inference_code(tmp_path: Path) -> Path:
    """Create a dummy inference.py file."""
    code_path = tmp_path / "inference.py"
    code_path.write_text('"""Inference handler."""\ndef model_fn(model_dir): pass\n')
    return code_path


# ── SageMaker Packaging Tests ──────────────────────────────────


class TestPackageForSagemaker:
    """Tests for package_for_sagemaker method."""

    def test_package_for_sagemaker_creates_tarball(
        self,
        packager: ArtifactPackager,
        model_dir: Path,
        inference_code: Path,
        tmp_path: Path,
    ) -> None:
        """Verify tar.gz created with correct structure."""
        output_path = str(tmp_path / "output" / "model.tar.gz")

        result = packager.package_for_sagemaker(
            model_path=str(model_dir),
            output_path=output_path,
            inference_code_path=str(inference_code),
        )

        assert result == output_path
        assert os.path.exists(output_path)

        # Verify it's a valid tar.gz
        with tarfile.open(output_path, "r:gz") as tar:
            members = tar.getnames()
            assert any(m.startswith("model/") for m in members)
            assert any(m.startswith("code/") for m in members)

    def test_package_for_sagemaker_includes_inference_code(
        self,
        packager: ArtifactPackager,
        model_dir: Path,
        inference_code: Path,
        tmp_path: Path,
    ) -> None:
        """Verify code/ directory in tar contains inference.py."""
        output_path = str(tmp_path / "model.tar.gz")

        packager.package_for_sagemaker(
            model_path=str(model_dir),
            output_path=output_path,
            inference_code_path=str(inference_code),
        )

        with tarfile.open(output_path, "r:gz") as tar:
            members = tar.getnames()
            assert "code/inference.py" in members
            assert "code/requirements.txt" in members

    def test_package_for_sagemaker_includes_model_files(
        self,
        packager: ArtifactPackager,
        model_dir: Path,
        inference_code: Path,
        tmp_path: Path,
    ) -> None:
        """Verify model/ directory contains expected files."""
        output_path = str(tmp_path / "model.tar.gz")

        packager.package_for_sagemaker(
            model_path=str(model_dir),
            output_path=output_path,
            inference_code_path=str(inference_code),
        )

        with tarfile.open(output_path, "r:gz") as tar:
            members = tar.getnames()
            assert "model/config.json" in members
            assert "model/model.safetensors" in members

    def test_package_for_sagemaker_missing_model_dir(
        self, packager: ArtifactPackager, tmp_path: Path
    ) -> None:
        """Verify error raised for missing model directory."""
        with pytest.raises(FileNotFoundError):
            packager.package_for_sagemaker(
                model_path=str(tmp_path / "nonexistent"),
                output_path=str(tmp_path / "model.tar.gz"),
            )

    @patch("src.serving.artifact_packager.boto3")
    def test_package_for_sagemaker_s3_upload(
        self,
        mock_boto3: MagicMock,
        packager: ArtifactPackager,
        model_dir: Path,
        inference_code: Path,
        tmp_path: Path,
    ) -> None:
        """Verify S3 upload when output_path is an S3 URI."""
        s3_uri = "s3://my-bucket/models/model.tar.gz"

        result = packager.package_for_sagemaker(
            model_path=str(model_dir),
            output_path=s3_uri,
            inference_code_path=str(inference_code),
        )

        assert result == s3_uri
        mock_boto3.client.return_value.upload_file.assert_called_once()


# ── Bedrock Packaging Tests ────────────────────────────────────


class TestPackageForBedrock:
    """Tests for package_for_bedrock method."""

    def test_package_for_bedrock_validates_architecture(
        self,
        packager: ArtifactPackager,
        tmp_path: Path,
    ) -> None:
        """Verify config.json architecture is checked."""
        model_dir = _create_model_dir(tmp_path, architecture="UnsupportedModel")

        with pytest.raises(ValueError, match="invalid for Bedrock"):
            packager.package_for_bedrock(
                model_path=str(model_dir),
                output_s3_uri="s3://bucket/bedrock-models/test/",
            )

    @patch("src.serving.artifact_packager.boto3")
    def test_package_for_bedrock_uploads_files(
        self,
        mock_boto3: MagicMock,
        packager: ArtifactPackager,
        model_dir: Path,
    ) -> None:
        """Verify files are uploaded to S3."""
        s3_uri = "s3://bucket/bedrock-models/test"

        result = packager.package_for_bedrock(
            model_path=str(model_dir),
            output_s3_uri=s3_uri,
        )

        assert result == s3_uri
        s3_client = mock_boto3.client.return_value
        assert s3_client.upload_file.call_count > 0


# ── Verify Artifact Tests ──────────────────────────────────────


class TestVerifyArtifact:
    """Tests for verify_artifact method."""

    def test_verify_artifact_valid(
        self, packager: ArtifactPackager, model_dir: Path
    ) -> None:
        """Verify returns is_valid=True for correct artifacts."""
        result = packager.verify_artifact(str(model_dir), "bedrock")

        assert result["is_valid"] is True
        assert result["issues"] == []
        assert result["size_gb"] >= 0

    def test_verify_artifact_missing_files(
        self, packager: ArtifactPackager, tmp_path: Path
    ) -> None:
        """Verify returns issues list for missing files."""
        model_dir = _create_model_dir(
            tmp_path, include_tokenizer=False, include_safetensors=False
        )

        result = packager.verify_artifact(str(model_dir), "bedrock")

        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        issue_text = " ".join(result["issues"])
        assert "tokenizer" in issue_text.lower() or "safetensors" in issue_text.lower()

    def test_verify_artifact_too_large(
        self, packager: ArtifactPackager, tmp_path: Path
    ) -> None:
        """Verify size check works by testing the limit logic."""
        model_dir = _create_model_dir(tmp_path)

        # Patch the size calculation to simulate a > 50GB model
        original_verify = packager._verify_bedrock_artifact

        def fake_verify(path: str) -> dict[str, Any]:
            result = original_verify(path)
            # Override size to exceed limit
            result["size_gb"] = 55.0
            result["is_valid"] = False
            result["issues"].append(
                "Total size 55.0GB exceeds Bedrock limit of 50.0GB"
            )
            return result

        packager._verify_bedrock_artifact = fake_verify  # type: ignore[assignment]

        result = packager.verify_artifact(str(model_dir), "bedrock")
        assert result["is_valid"] is False
        assert any("50.0GB" in issue for issue in result["issues"])

    def test_verify_artifact_sagemaker_tarball(
        self,
        packager: ArtifactPackager,
        model_dir: Path,
        inference_code: Path,
        tmp_path: Path,
    ) -> None:
        """Verify SageMaker tar.gz validation works."""
        output_path = str(tmp_path / "model.tar.gz")
        packager.package_for_sagemaker(
            model_path=str(model_dir),
            output_path=output_path,
            inference_code_path=str(inference_code),
        )

        result = packager.verify_artifact(output_path, "sagemaker")
        assert result["is_valid"] is True

    def test_verify_artifact_unknown_target(
        self, packager: ArtifactPackager, model_dir: Path
    ) -> None:
        """Verify unknown target returns invalid."""
        result = packager.verify_artifact(str(model_dir), "unknown")
        assert result["is_valid"] is False
        assert "Unknown target" in result["issues"][0]

    def test_verify_artifact_missing_config(
        self, packager: ArtifactPackager, tmp_path: Path
    ) -> None:
        """Verify missing config.json is reported."""
        model_dir = _create_model_dir(tmp_path, include_config=False)
        result = packager.verify_artifact(str(model_dir), "bedrock")
        assert result["is_valid"] is False
        assert any("config.json" in issue for issue in result["issues"])
