"""Tests for Bedrock Import Manager."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.bedrock import BedrockImportManager


@pytest.fixture()
def mock_bedrock_client() -> MagicMock:
    """Create a mock Bedrock client."""
    client = MagicMock()
    client.create_model_import_job.return_value = {
        "jobArn": "arn:aws:bedrock:us-east-1:123456789012:model-import-job/test-job"
    }
    client.get_model_import_job.return_value = {
        "status": "Completed",
        "importedModelArn": "arn:aws:bedrock:us-east-1:123456789012:imported-model/test-model",
    }
    client.create_provisioned_model_throughput.return_value = {
        "provisionedModelArn": "arn:aws:bedrock:us-east-1:123456789012:provisioned-model/test"
    }
    client.get_provisioned_model_throughput.return_value = {
        "status": "InService"
    }
    client.list_imported_models.return_value = {
        "modelSummaries": [
            {"modelArn": "arn:test", "modelName": "test-model"}
        ]
    }
    client.list_provisioned_model_throughputs.return_value = {
        "provisionedModelSummaries": []
    }
    return client


@pytest.fixture()
def mock_runtime_client() -> MagicMock:
    """Create a mock bedrock-runtime client."""
    client = MagicMock()
    body = MagicMock()
    body.read.return_value = json.dumps(
        {
            "generation": "Test response",
            "prompt_token_count": 10,
            "generation_token_count": 5,
        }
    ).encode("utf-8")
    client.invoke_model.return_value = {"body": body}
    return client


@pytest.fixture()
def manager(
    mock_bedrock_client: MagicMock, mock_runtime_client: MagicMock
) -> BedrockImportManager:
    """Create a BedrockImportManager with mocked boto3."""
    with patch("src.serving.bedrock.boto3") as mock_boto3:

        def _client_factory(service: str, **kwargs: Any) -> MagicMock:
            if service == "bedrock-runtime":
                return mock_runtime_client
            return mock_bedrock_client

        mock_boto3.client.side_effect = _client_factory
        mgr = BedrockImportManager(region="us-east-1")
    return mgr


class TestImportModel:
    """Tests for import_model."""

    def test_import_model_validates_artifacts(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify _validate_s3_artifacts is called."""
        with patch.object(manager, "_validate_s3_artifacts") as mock_validate:
            result = manager.import_model(
                model_name="test-model",
                model_s3_uri="s3://bucket/models/test/",
                role_arn="arn:aws:iam::role/test",
            )

            mock_validate.assert_called_once_with("s3://bucket/models/test/")
            assert result["status"] == "Completed"
            assert "model_arn" in result

    def test_import_model_polls_until_complete(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify polling happens."""
        with patch.object(manager, "_validate_s3_artifacts"):
            result = manager.import_model(
                model_name="test-model",
                model_s3_uri="s3://bucket/models/test/",
                role_arn="arn:aws:iam::role/test",
            )

            assert result["status"] == "Completed"
            assert result["import_job_arn"] == (
                "arn:aws:bedrock:us-east-1:123456789012:model-import-job/test-job"
            )

    def test_import_model_fails_on_unsupported_architecture(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify error for unsupported architecture."""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "models/config.json", "Size": 100},
                {"Key": "models/model.safetensors", "Size": 1000},
                {"Key": "models/tokenizer.json", "Size": 100},
                {"Key": "models/tokenizer_config.json", "Size": 100},
            ]
        }
        config = {"architectures": ["UnsupportedArch"]}
        body = MagicMock()
        body.read.return_value = json.dumps(config).encode("utf-8")
        mock_s3.get_object.return_value = {"Body": body}

        with patch("src.serving.bedrock.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_s3
            with pytest.raises(ValueError, match="Unsupported architecture"):
                manager._validate_s3_artifacts("s3://bucket/models/")

    def test_import_model_fails_on_job_failure(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify RuntimeError on import failure."""
        manager._client.get_model_import_job.return_value = {
            "status": "Failed",
            "failureMessage": "Bad model format",
        }

        with patch.object(manager, "_validate_s3_artifacts"):
            with pytest.raises(RuntimeError, match="Bedrock import failed"):
                manager.import_model(
                    model_name="test-model",
                    model_s3_uri="s3://bucket/models/test/",
                    role_arn="arn:aws:iam::role/test",
                )


class TestProvisionedThroughput:
    """Tests for create_provisioned_throughput."""

    def test_create_provisioned_throughput(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify API call and wait."""
        result = manager.create_provisioned_throughput(
            model_arn="arn:aws:bedrock:us-east-1:123456789012:model/test",
            throughput_name="test-throughput",
            model_units=2,
        )

        manager._client.create_provisioned_model_throughput.assert_called_once()
        assert "provisioned_model_arn" in result


class TestInvokeModel:
    """Tests for invoke_model."""

    def test_invoke_model_returns_text(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify invoke returns generated text."""
        result = manager.invoke_model(
            provisioned_model_arn="arn:aws:bedrock:us-east-1:123456789012:model/test",
            prompt="Hello!",
        )

        assert result["generated_text"] == "Test response"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 5
        assert "latency_ms" in result


class TestDeleteModel:
    """Tests for delete_model."""

    def test_delete_model_cleans_up_throughput_first(
        self, manager: BedrockImportManager
    ) -> None:
        """Verify provisioned throughput deleted before model."""
        model_arn = "arn:aws:bedrock:us-east-1:123456789012:model/test"
        manager._client.list_provisioned_model_throughputs.return_value = {
            "provisionedModelSummaries": [
                {
                    "modelArn": model_arn,
                    "provisionedModelArn": "arn:provisioned-test",
                }
            ]
        }

        manager.delete_model(model_arn)

        manager._client.delete_provisioned_model_throughput.assert_called_once()
        manager._client.delete_imported_model.assert_called_once_with(
            modelIdentifier=model_arn
        )

        # Verify throughput deleted before model
        calls = manager._client.method_calls
        call_names = [c[0] for c in calls]
        tp_idx = call_names.index("delete_provisioned_model_throughput")
        model_idx = call_names.index("delete_imported_model")
        assert tp_idx < model_idx
