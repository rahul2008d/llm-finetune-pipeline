"""Unit tests for monitoring.endpoint_monitor module."""

from unittest.mock import MagicMock, patch

from src.monitoring.endpoint_monitor import EndpointMonitor


class TestEndpointMonitor:
    """Tests for EndpointMonitor."""

    @patch("boto3.client")
    def test_setup_monitoring_creates_alarms(self, mock_boto3_client: MagicMock) -> None:
        """Verify CloudWatch alarm creation."""
        mock_cw = MagicMock()
        mock_boto3_client.return_value = mock_cw
        mock_cw.put_dashboard.return_value = {}

        monitor = EndpointMonitor(region="us-east-1")
        result = monitor.setup_monitoring(
            endpoint_name="test-endpoint",
            alert_sns_topic_arn="arn:aws:sns:us-east-1:123:alerts",
            latency_p99_threshold_ms=5000,
            error_rate_threshold_pct=1.0,
        )

        assert "alarm_arns" in result
        assert len(result["alarm_arns"]) == 4
        assert mock_cw.put_metric_alarm.call_count == 4

        for call in mock_cw.put_metric_alarm.call_args_list:
            assert "AlarmName" in call.kwargs

    @patch("boto3.client")
    def test_setup_monitoring_creates_dashboard(self, mock_boto3_client: MagicMock) -> None:
        """Verify dashboard JSON creation."""
        mock_cw = MagicMock()
        mock_boto3_client.return_value = mock_cw

        monitor = EndpointMonitor(region="us-west-2")
        result = monitor.setup_monitoring(
            endpoint_name="my-endpoint",
            alert_sns_topic_arn="arn:aws:sns:us-west-2:123:alerts",
        )

        assert "dashboard_url" in result
        assert "us-west-2" in result["dashboard_url"]
        assert "LLMFineTune-my-endpoint" in result["dashboard_url"]
        mock_cw.put_dashboard.assert_called()

        # Find the put_dashboard call
        dashboard_calls = [
            c for c in mock_cw.put_dashboard.call_args_list
        ]
        assert len(dashboard_calls) >= 1
        last_call = dashboard_calls[-1]
        assert last_call.kwargs["DashboardName"] == "LLMFineTune-my-endpoint"

    @patch("boto3.client")
    def test_get_monitoring_report(self, mock_boto3_client: MagicMock) -> None:
        """Mock CloudWatch and verify report structure."""
        mock_cw = MagicMock()
        mock_boto3_client.return_value = mock_cw

        mock_cw.get_metric_statistics.side_effect = [
            # Invocations
            {"Datapoints": [{"Sum": 100.0}, {"Sum": 200.0}]},
            # 5xx errors
            {"Datapoints": [{"Sum": 1.0}, {"Sum": 0.0}]},
            # Latency p50 (Average)
            {"Datapoints": [{"Average": 50000.0}, {"Average": 60000.0}]},
            # Latency p99 (Maximum)
            {"Datapoints": [{"Maximum": 200000.0}, {"Maximum": 150000.0}]},
        ]

        monitor = EndpointMonitor(region="us-east-1")
        report = monitor.get_monitoring_report("test-endpoint", hours=24)

        assert report["endpoint_name"] == "test-endpoint"
        assert report["period_hours"] == 24
        assert "uptime_pct" in report
        assert "error_rate" in report
        assert "latency_stats" in report
        assert "invocation_count" in report
        assert report["invocation_count"] == 300
        assert "cost_estimate" in report
        assert report["latency_stats"]["avg_p50_ms"] == 55.0
        assert report["latency_stats"]["max_p99_ms"] == 200.0
