"""Tests for auto-scaling configuration."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.autoscaling import EndpointAutoScaler


@pytest.fixture()
def mock_as_client() -> MagicMock:
    """Create a mock Application Auto Scaling client."""
    client = MagicMock()
    client.describe_scalable_targets.return_value = {
        "ScalableTargets": [
            {
                "ResourceId": "endpoint/test-ep/variant/AllTraffic",
                "MinCapacity": 1,
                "MaxCapacity": 4,
            }
        ]
    }
    client.describe_scaling_policies.return_value = {
        "ScalingPolicies": [
            {
                "PolicyName": "test-policy",
                "ResourceId": "endpoint/test-ep/variant/AllTraffic",
            }
        ]
    }
    client.describe_scaling_activities.return_value = {
        "ScalingActivities": []
    }
    return client


@pytest.fixture()
def autoscaler(mock_as_client: MagicMock) -> EndpointAutoScaler:
    """Create an EndpointAutoScaler with mocked boto3."""
    with patch("src.serving.autoscaling.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_as_client
        scaler = EndpointAutoScaler(region="us-east-1")
    return scaler


class TestConfigureAutoscaling:
    """Tests for configure_autoscaling."""

    def test_configure_autoscaling_registers_target(
        self, autoscaler: EndpointAutoScaler
    ) -> None:
        """Verify register_scalable_target called."""
        autoscaler.configure_autoscaling(
            endpoint_name="test-ep",
            min_instances=1,
            max_instances=4,
        )

        autoscaler._client.register_scalable_target.assert_called_once()
        call_kwargs = autoscaler._client.register_scalable_target.call_args[1]
        assert call_kwargs["MinCapacity"] == 1
        assert call_kwargs["MaxCapacity"] == 4

    def test_configure_autoscaling_creates_policy(
        self, autoscaler: EndpointAutoScaler
    ) -> None:
        """Verify put_scaling_policy called."""
        autoscaler.configure_autoscaling(
            endpoint_name="test-ep",
            target_invocations_per_instance=50,
        )

        autoscaler._client.put_scaling_policy.assert_called_once()
        call_kwargs = autoscaler._client.put_scaling_policy.call_args[1]
        assert call_kwargs["PolicyType"] == "TargetTrackingScaling"
        config = call_kwargs["TargetTrackingScalingPolicyConfiguration"]
        assert config["TargetValue"] == 50.0


class TestScheduledScaling:
    """Tests for configure_scheduled_scaling."""

    def test_scheduled_scaling(self, autoscaler: EndpointAutoScaler) -> None:
        """Verify put_scheduled_action called for each schedule."""
        schedules = [
            {
                "name": "scale-up-morning",
                "schedule_expression": "cron(0 9 * * ? *)",
                "min_capacity": 2,
                "max_capacity": 8,
            },
            {
                "name": "scale-down-night",
                "schedule_expression": "cron(0 22 * * ? *)",
                "min_capacity": 1,
                "max_capacity": 2,
            },
        ]

        autoscaler.configure_scheduled_scaling(
            endpoint_name="test-ep",
            schedules=schedules,
        )

        assert autoscaler._client.put_scheduled_action.call_count == 2


class TestGetScalingStatus:
    """Tests for get_scaling_status."""

    def test_get_scaling_status(self, autoscaler: EndpointAutoScaler) -> None:
        """Verify describe calls and return structure."""
        result = autoscaler.get_scaling_status("test-ep")

        assert "scalable_targets" in result
        assert "scaling_policies" in result
        assert "recent_activities" in result
        autoscaler._client.describe_scalable_targets.assert_called_once()
        autoscaler._client.describe_scaling_policies.assert_called_once()


class TestRemoveAutoscaling:
    """Tests for remove_autoscaling."""

    def test_remove_autoscaling(self, autoscaler: EndpointAutoScaler) -> None:
        """Verify deregister called."""
        autoscaler.remove_autoscaling("test-ep")

        autoscaler._client.deregister_scalable_target.assert_called_once()
        call_kwargs = autoscaler._client.deregister_scalable_target.call_args[1]
        assert "endpoint/test-ep/variant/AllTraffic" in call_kwargs["ResourceId"]
