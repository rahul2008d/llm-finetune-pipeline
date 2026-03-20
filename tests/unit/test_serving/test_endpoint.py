"""Unit tests for serving.endpoint module."""

from unittest.mock import MagicMock, patch

from src.serving.endpoint import SageMakerEndpointHandler


class TestSageMakerEndpointHandler:
    """Tests for SageMakerEndpointHandler."""

    def test_init(self) -> None:
        """Test handler initialization."""
        handler = SageMakerEndpointHandler(
            model_path="s3://bucket/model.tar.gz",
            endpoint_name="test-endpoint",
        )
        assert handler.model_path == "s3://bucket/model.tar.gz"
        assert handler.endpoint_name == "test-endpoint"
        assert handler.instance_type == "ml.g5.2xlarge"
        assert handler.instance_count == 1

    @patch("serving.endpoint.boto3")
    def test_delete(self, mock_boto3: MagicMock) -> None:
        """Test endpoint deletion."""
        mock_client = MagicMock()
        mock_boto3.client.return_value = mock_client

        handler = SageMakerEndpointHandler(
            model_path="s3://bucket/model",
            endpoint_name="test-ep",
        )
        handler.delete()

        mock_client.delete_endpoint.assert_called_once_with(EndpointName="test-ep")
