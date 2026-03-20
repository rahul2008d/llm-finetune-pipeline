"""Unit tests for monitoring.alerting module."""

from unittest.mock import MagicMock, patch

from src.monitoring.alerting import AlertManager


class TestAlertManager:
    """Tests for AlertManager."""

    @patch("boto3.client")
    def test_send_alert(self, mock_boto3_client: MagicMock) -> None:
        """Test sending an alert via SNS."""
        mock_client = MagicMock()
        mock_client.publish.return_value = {"MessageId": "test-123"}
        mock_boto3_client.return_value = mock_client

        manager = AlertManager(topic_arn="arn:aws:sns:us-east-1:123:test")
        msg_id = manager.send_alert("Test Alert", "Something happened")

        assert msg_id == "test-123"
        mock_client.publish.assert_called_once()
