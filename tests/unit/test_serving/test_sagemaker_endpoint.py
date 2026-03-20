"""Tests for SageMaker Endpoint Manager."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.endpoint import SageMakerEndpointManager


@pytest.fixture()
def mock_sm_client() -> MagicMock:
    """Create a mock SageMaker client."""
    client = MagicMock()

    # Default describe_endpoint returns InService
    client.describe_endpoint.return_value = {
        "EndpointName": "test-endpoint",
        "EndpointArn": "arn:aws:sagemaker:us-east-1:123456789012:endpoint/test-endpoint",
        "EndpointStatus": "InService",
        "EndpointConfigName": "test-endpoint-config",
        "CreationTime": "2025-01-01T00:00:00Z",
        "LastModifiedTime": "2025-01-01T00:00:00Z",
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "CurrentWeight": 1.0,
                "CurrentInstanceCount": 1,
            }
        ],
    }

    client.describe_endpoint_config.return_value = {
        "EndpointConfigName": "test-endpoint-config",
        "ProductionVariants": [
            {
                "VariantName": "AllTraffic",
                "ModelName": "test-model",
                "InstanceType": "ml.g5.xlarge",
                "InitialInstanceCount": 1,
            }
        ],
    }

    client.create_model.return_value = {}
    client.create_endpoint_config.return_value = {}
    client.create_endpoint.return_value = {}
    client.update_endpoint.return_value = {}

    return client


@pytest.fixture()
def manager(mock_sm_client: MagicMock) -> SageMakerEndpointManager:
    """Create a SageMakerEndpointManager with mocked boto3."""
    with patch("src.serving.endpoint.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sm_client
        mgr = SageMakerEndpointManager(region="us-east-1")
    return mgr


class TestCreateEndpoint:
    """Tests for create_endpoint."""

    def test_create_endpoint_calls_apis_in_order(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify create_model -> create_endpoint_config -> create_endpoint."""
        result = manager.create_endpoint(
            model_data_url="s3://bucket/model.tar.gz",
            endpoint_name="test-ep",
            role_arn="arn:aws:iam::role/test",
        )

        manager._client.create_model.assert_called_once()
        manager._client.create_endpoint_config.assert_called_once()
        manager._client.create_endpoint.assert_called_once()

        # Verify order: model before config before endpoint
        calls = manager._client.method_calls
        call_names = [c[0] for c in calls]
        model_idx = call_names.index("create_model")
        config_idx = call_names.index("create_endpoint_config")
        endpoint_idx = call_names.index("create_endpoint")
        assert model_idx < config_idx < endpoint_idx

    def test_create_endpoint_waits_for_inservice(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify endpoint waits for InService."""
        result = manager.create_endpoint(
            model_data_url="s3://bucket/model.tar.gz",
            endpoint_name="test-ep",
            role_arn="arn:aws:iam::role/test",
        )
        assert result["status"] == "InService"

    def test_create_endpoint_with_data_capture(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify data capture config is passed."""
        manager.create_endpoint(
            model_data_url="s3://bucket/model.tar.gz",
            endpoint_name="test-ep",
            role_arn="arn:aws:iam::role/test",
            data_capture_enabled=True,
            data_capture_s3_uri="s3://bucket/capture/",
            data_capture_sampling_pct=50,
        )

        config_call = manager._client.create_endpoint_config.call_args
        assert "DataCaptureConfig" in config_call[1]
        assert config_call[1]["DataCaptureConfig"]["EnableCapture"] is True


class TestDeleteEndpoint:
    """Tests for delete_endpoint."""

    def test_delete_endpoint_cleans_up(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify all 3 resources deleted."""
        manager.delete_endpoint("test-endpoint", delete_model=True)

        manager._client.delete_endpoint.assert_called_once_with(
            EndpointName="test-endpoint"
        )
        manager._client.delete_endpoint_config.assert_called_once()
        manager._client.delete_model.assert_called_once()


class TestDescribeEndpoint:
    """Tests for describe_endpoint."""

    def test_describe_endpoint_returns_info(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify dict structure."""
        result = manager.describe_endpoint("test-endpoint")

        assert result["status"] == "InService"
        assert "endpoint_arn" in result
        assert "variants" in result
        assert "creation_time" in result
        assert "last_modified" in result


class TestBlueGreenDeploy:
    """Tests for blue_green_deploy."""

    def test_blue_green_deploy_shifts_traffic(
        self, manager: SageMakerEndpointManager
    ) -> None:
        """Verify traffic update calls during blue-green deploy."""
        result = manager.blue_green_deploy(
            endpoint_name="test-endpoint",
            new_model_data_url="s3://bucket/new-model.tar.gz",
            instance_type="ml.g5.xlarge",
            canary_pct=0.1,
            bake_time_minutes=0,  # Skip bake time
            role_arn="arn:aws:iam::role/test",
        )

        assert result["status"] == "completed"
        assert result["rolled_back"] is False
        # Verify traffic was shifted
        assert manager._client.update_endpoint_weights_and_capacities.call_count >= 2

    @patch("src.serving.endpoint.boto3")
    def test_blue_green_rollback_on_alarm(
        self,
        mock_boto3_module: MagicMock,
        manager: SageMakerEndpointManager,
    ) -> None:
        """Verify rollback when alarm fires."""
        # Mock CloudWatch to return alarm in ALARM state
        mock_cw = MagicMock()
        mock_cw.describe_alarms.return_value = {
            "MetricAlarms": [
                {"AlarmName": "test-alarm", "StateValue": "ALARM"}
            ]
        }
        mock_boto3_module.client.return_value = mock_cw

        result = manager.blue_green_deploy(
            endpoint_name="test-endpoint",
            new_model_data_url="s3://bucket/new-model.tar.gz",
            bake_time_minutes=1,
            rollback_alarm_names=["test-alarm"],
            role_arn="arn:aws:iam::role/test",
        )

        assert result["rolled_back"] is True
        assert result["status"] == "rolled_back"


class TestListEndpoints:
    """Tests for list_endpoints."""

    def test_list_endpoints(self, manager: SageMakerEndpointManager) -> None:
        """Verify list returns expected structure."""
        manager._client.list_endpoints.return_value = {
            "Endpoints": [
                {
                    "EndpointName": "ep-1",
                    "EndpointStatus": "InService",
                    "CreationTime": "2025-01-01",
                }
            ]
        }

        result = manager.list_endpoints(name_contains="ep")
        assert len(result) == 1
        assert result[0]["endpoint_name"] == "ep-1"
