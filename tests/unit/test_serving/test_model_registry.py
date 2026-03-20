"""Tests for SageMaker Model Registry integration."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.model_registry import ModelRegistryManager


# ── Fixtures ────────────────────────────────────────────────────


class _FakeTrainingResult:
    """Minimal stand-in for TrainingResult."""

    run_id = "run-001"
    experiment_name = "test-experiment"
    final_train_loss = 0.42
    final_eval_loss = 0.55
    best_eval_loss = 0.50
    total_steps = 1000
    training_time_seconds = 3600.0
    estimated_cost_usd = 12.50
    adapter_s3_uri = "s3://bucket/adapters/run-001"
    metrics: dict[str, float] = {}


@pytest.fixture()
def mock_sagemaker_client() -> MagicMock:
    """Create a mock SageMaker client."""
    client = MagicMock()
    # Default: describe raises so _ensure_model_package_group creates
    client.exceptions.ClientError = type("ClientError", (Exception,), {})
    client.describe_model_package_group.side_effect = client.exceptions.ClientError(
        "Not found"
    )
    client.create_model_package.return_value = {
        "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"
    }
    # Paginator for list_versions
    paginator = MagicMock()
    paginator.paginate.return_value = [
        {
            "ModelPackageSummaryList": [
                {
                    "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1",
                    "ModelApprovalStatus": "Approved",
                    "CreationTime": "2025-01-01T00:00:00Z",
                },
                {
                    "ModelPackageArn": "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/2",
                    "ModelApprovalStatus": "PendingManualApproval",
                    "CreationTime": "2025-01-02T00:00:00Z",
                },
            ]
        }
    ]
    client.get_paginator.return_value = paginator
    return client


@pytest.fixture()
def registry(mock_sagemaker_client: MagicMock) -> ModelRegistryManager:
    """Create a ModelRegistryManager with mocked boto3."""
    with patch("src.serving.model_registry.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sagemaker_client
        mgr = ModelRegistryManager(region="us-east-1")
    return mgr


@pytest.fixture()
def training_result() -> _FakeTrainingResult:
    """Create a fake training result."""
    return _FakeTrainingResult()


# ── Tests ───────────────────────────────────────────────────────


class TestRegisterModel:
    """Tests for register_model method."""

    def test_register_model_creates_group_and_package(
        self,
        registry: ModelRegistryManager,
        training_result: _FakeTrainingResult,
    ) -> None:
        """Verify both CreateModelPackageGroup and CreateModelPackage are called."""
        registry.register_model(
            model_s3_uri="s3://bucket/model.tar.gz",
            model_package_group_name="test-group",
            training_result=training_result,
        )

        # Group creation attempted (describe raises, so create is called)
        registry._client.create_model_package_group.assert_called_once()
        registry._client.create_model_package.assert_called_once()

    def test_register_model_returns_arn(
        self,
        registry: ModelRegistryManager,
        training_result: _FakeTrainingResult,
    ) -> None:
        """Verify ARN string returned."""
        arn = registry.register_model(
            model_s3_uri="s3://bucket/model.tar.gz",
            model_package_group_name="test-group",
            training_result=training_result,
        )

        assert arn == "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"

    def test_register_model_with_eval_results(
        self,
        registry: ModelRegistryManager,
        training_result: _FakeTrainingResult,
    ) -> None:
        """Verify metrics included when eval_results provided."""
        eval_results = {"accuracy": 0.92, "perplexity": 3.14}

        registry.register_model(
            model_s3_uri="s3://bucket/model.tar.gz",
            model_package_group_name="test-group",
            training_result=training_result,
            eval_results=eval_results,
        )

        call_kwargs = registry._client.create_model_package.call_args[1]
        assert "ModelMetrics" in call_kwargs
        stats_body = call_kwargs["ModelMetrics"]["ModelQuality"]["Statistics"]["Body"]
        parsed = json.loads(stats_body)
        assert parsed["accuracy"] == 0.92
        assert parsed["perplexity"] == 3.14

    def test_register_model_skips_group_creation_if_exists(
        self,
        registry: ModelRegistryManager,
        training_result: _FakeTrainingResult,
    ) -> None:
        """Verify group is not re-created if describe succeeds."""
        registry._client.describe_model_package_group.side_effect = None
        registry._client.describe_model_package_group.return_value = {}

        registry.register_model(
            model_s3_uri="s3://bucket/model.tar.gz",
            model_package_group_name="test-group",
            training_result=training_result,
        )

        registry._client.create_model_package_group.assert_not_called()


class TestApproval:
    """Tests for approve_model and reject_model methods."""

    def test_approve_model(self, registry: ModelRegistryManager) -> None:
        """Verify UpdateModelPackage called with Approved."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"
        registry.approve_model(arn)

        registry._client.update_model_package.assert_called_once_with(
            ModelPackageArn=arn,
            ModelApprovalStatus="Approved",
        )

    def test_reject_model(self, registry: ModelRegistryManager) -> None:
        """Verify rejection reason passed."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"
        reason = "Performance below threshold"
        registry.reject_model(arn, reason)

        registry._client.update_model_package.assert_called_once_with(
            ModelPackageArn=arn,
            ModelApprovalStatus="Rejected",
            ApprovalDescription=reason,
        )


class TestQueries:
    """Tests for get_latest_approved, list_versions, get_model_lineage."""

    def test_get_latest_approved(self, registry: ModelRegistryManager) -> None:
        """Mock list response, verify filtering returns correct result."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"
        registry._client.list_model_packages.return_value = {
            "ModelPackageSummaryList": [
                {
                    "ModelPackageArn": arn,
                    "CreationTime": "2025-01-01T00:00:00Z",
                }
            ]
        }
        registry._client.describe_model_package.return_value = {
            "InferenceSpecification": {
                "Containers": [{"ModelDataUrl": "s3://bucket/model.tar.gz"}]
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "Body": '{"accuracy": 0.95}',
                    }
                }
            },
        }

        result = registry.get_latest_approved("test-group")

        assert result["model_package_arn"] == arn
        assert result["model_data_url"] == "s3://bucket/model.tar.gz"
        assert result["metrics"]["accuracy"] == 0.95

    def test_get_latest_approved_empty(self, registry: ModelRegistryManager) -> None:
        """Verify empty dict returned when no approved model found."""
        registry._client.list_model_packages.return_value = {
            "ModelPackageSummaryList": []
        }

        result = registry.get_latest_approved("test-group")
        assert result == {}

    def test_list_versions(self, registry: ModelRegistryManager) -> None:
        """Verify all versions returned."""
        versions = registry.list_versions("test-group")

        assert len(versions) == 2
        assert versions[0]["status"] == "Approved"
        assert versions[1]["status"] == "PendingManualApproval"

    def test_get_model_lineage(self, registry: ModelRegistryManager) -> None:
        """Verify lineage info returned from metadata and metrics."""
        arn = "arn:aws:sagemaker:us-east-1:123456789012:model-package/test-group/1"
        registry._client.describe_model_package.return_value = {
            "CustomerMetadataProperties": {
                "run_id": "run-001",
                "experiment_name": "test-experiment",
                "final_train_loss": "0.42",
                "best_eval_loss": "0.50",
                "training_time_seconds": "3600.0",
            },
            "ModelMetrics": {
                "ModelQuality": {
                    "Statistics": {
                        "ContentType": "application/json",
                        "Body": '{"accuracy": 0.95}',
                    }
                }
            },
        }

        lineage = registry.get_model_lineage(arn)

        assert lineage["run_id"] == "run-001"
        assert lineage["experiment_name"] == "test-experiment"
        assert lineage["eval_results"]["accuracy"] == 0.95
