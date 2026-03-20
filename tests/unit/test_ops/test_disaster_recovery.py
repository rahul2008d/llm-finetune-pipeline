"""Unit tests for ops.disaster_recovery module."""

from unittest.mock import MagicMock, patch

from src.ops.disaster_recovery import DisasterRecoveryManager


class TestDisasterRecoveryManager:
    """Tests for DisasterRecoveryManager."""

    @patch("boto3.client")
    def test_export_endpoint_config(self, mock_boto3_client: MagicMock) -> None:
        """Mock SageMaker, verify exported JSON structure."""
        mock_sm = MagicMock()
        mock_boto3_client.return_value = mock_sm

        mock_sm.describe_endpoint.return_value = {
            "EndpointName": "test-endpoint",
            "EndpointConfigName": "test-config",
            "EndpointStatus": "InService",
        }
        mock_sm.describe_endpoint_config.return_value = {
            "EndpointConfigName": "test-config",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "test-model",
                    "InstanceType": "ml.g5.xlarge",
                    "InitialInstanceCount": 1,
                }
            ],
            "DataCaptureConfig": {"EnableCapture": False},
        }
        mock_sm.describe_model.return_value = {
            "ModelName": "test-model",
            "PrimaryContainer": {
                "Image": "123456789.dkr.ecr.us-east-1.amazonaws.com/llm:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
            },
            "ExecutionRoleArn": "arn:aws:iam::role/sagemaker-exec",
        }

        dr = DisasterRecoveryManager(region="us-east-1")
        result = dr.export_endpoint_config("test-endpoint")

        assert result["endpoint_name"] == "test-endpoint"
        assert result["endpoint_config_name"] == "test-config"
        assert len(result["production_variants"]) == 1
        assert len(result["model_configs"]) == 1
        assert result["model_configs"][0]["model_name"] == "test-model"
        assert "exported_at" in result
        assert result["region"] == "us-east-1"

    @patch("boto3.client")
    def test_validate_backups(self, mock_boto3_client: MagicMock) -> None:
        """Mock S3, verify check results."""
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        mock_s3.list_objects_v2.side_effect = [
            # model artifacts
            {
                "KeyCount": 1,
                "Contents": [{"Key": "models/v1/", "LastModified": "2024-01-15"}],
            },
            # training configs
            {
                "KeyCount": 1,
                "Contents": [{"Key": "configs/default.yaml", "LastModified": "2024-01-14"}],
            },
            # mlflow data
            {"KeyCount": 0, "Contents": []},
            # endpoint configs
            {
                "KeyCount": 1,
                "Contents": [{"Key": "endpoint-configs/prod.json", "LastModified": "2024-01-15"}],
            },
        ]

        dr = DisasterRecoveryManager(region="us-east-1")
        result = dr.validate_backups()

        assert "all_valid" in result
        assert "checks" in result
        assert len(result["checks"]) == 4
        assert result["checks"][0]["status"] == "valid"
        assert result["checks"][1]["status"] == "valid"
        assert result["checks"][2]["status"] == "missing"
        assert result["checks"][3]["status"] == "valid"
        assert result["all_valid"] is False  # mlflow missing

    @patch("boto3.client")
    def test_full_region_failover(self, mock_boto3_client: MagicMock) -> None:
        """Mock all services, verify failover sequence."""
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        # Source region: describe endpoint + config + model
        mock_client.describe_endpoint.side_effect = [
            # Source region call
            {
                "EndpointName": "prod-endpoint",
                "EndpointConfigName": "prod-config",
                "EndpointStatus": "InService",
            },
            # Target region call (smoke test)
            {
                "EndpointName": "prod-endpoint-us-west-2",
                "EndpointStatus": "InService",
            },
        ]
        mock_client.describe_endpoint_config.return_value = {
            "EndpointConfigName": "prod-config",
            "ProductionVariants": [
                {
                    "VariantName": "AllTraffic",
                    "ModelName": "prod-model",
                    "InstanceType": "ml.g5.xlarge",
                    "InitialInstanceCount": 1,
                }
            ],
        }
        mock_client.describe_model.return_value = {
            "ModelName": "prod-model",
            "PrimaryContainer": {
                "Image": "123.dkr.ecr.us-east-1.amazonaws.com/llm:latest",
                "ModelDataUrl": "s3://bucket/model.tar.gz",
            },
            "ExecutionRoleArn": "arn:aws:iam::role/exec",
        }
        mock_client.create_model.return_value = {}
        mock_client.create_endpoint_config.return_value = {}
        mock_client.create_endpoint.return_value = {}

        dr = DisasterRecoveryManager(region="us-east-1")
        result = dr.full_region_failover(
            source_region="us-east-1",
            target_region="us-west-2",
            endpoint_name="prod-endpoint",
        )

        assert result["status"] == "completed"
        assert result["new_endpoint"] == "prod-endpoint-us-west-2"
        assert result["smoke_test_passed"] is True
        assert "duration_seconds" in result
        assert result["source_region"] == "us-east-1"
        assert result["target_region"] == "us-west-2"
        mock_client.create_model.assert_called_once()
        mock_client.create_endpoint_config.assert_called_once()
        mock_client.create_endpoint.assert_called_once()
