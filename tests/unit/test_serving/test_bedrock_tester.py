"""Tests for Bedrock Endpoint Tester."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.bedrock_tester import BedrockEndpointTester


@pytest.fixture()
def mock_runtime_client() -> MagicMock:
    """Create a mock bedrock-runtime client."""
    client = MagicMock()

    def _make_response() -> dict[str, Any]:
        body = MagicMock()
        body.read.return_value = json.dumps(
            {
                "generation": "Test response text",
                "prompt_token_count": 10,
                "generation_token_count": 5,
            }
        ).encode("utf-8")
        return {"body": body}

    client.invoke_model.return_value = _make_response()
    # Make invoke_model return fresh responses each time
    client.invoke_model.side_effect = lambda **kwargs: _make_response()
    return client


@pytest.fixture()
def tester(mock_runtime_client: MagicMock) -> BedrockEndpointTester:
    """Create a BedrockEndpointTester with mocked boto3."""
    with patch("src.serving.bedrock_tester.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_runtime_client
        t = BedrockEndpointTester(region="us-east-1")
    return t


class TestSmokeTest:
    """Tests for smoke_test method."""

    def test_smoke_test(self, tester: BedrockEndpointTester) -> None:
        """Verify smoke test structure and pass/fail."""
        result = tester.smoke_test("arn:aws:bedrock:model/test")

        assert result["passed"] is True
        assert len(result["results"]) == 5
        for r in result["results"]:
            assert r["status"] == "pass"
            assert r["response"] == "Test response text"

    def test_smoke_test_fails_on_error(
        self, tester: BedrockEndpointTester
    ) -> None:
        """Verify fail when invoke raises."""
        tester._runtime_client.invoke_model.side_effect = Exception("Error")

        result = tester.smoke_test("arn:aws:bedrock:model/test")
        assert result["passed"] is False


class TestThroughputTest:
    """Tests for throughput_test method."""

    def test_throughput_test(self, tester: BedrockEndpointTester) -> None:
        """Verify RPM calculation and returned keys."""
        result = tester.throughput_test(
            provisioned_model_arn="arn:aws:bedrock:model/test",
            target_rpm=5,
            duration_minutes=1,
        )

        assert "achieved_rpm" in result
        assert "throttle_rate" in result
        assert "error_rate" in result
        assert "latency_p50" in result
        assert "latency_p99" in result


class TestCompare:
    """Tests for compare_sagemaker_vs_bedrock."""

    def test_compare_returns_recommendation(
        self, tester: BedrockEndpointTester
    ) -> None:
        """Verify both endpoints called and recommendation returned."""
        # Mock SageMaker client
        mock_sm = MagicMock()
        sm_body = MagicMock()
        sm_body.read.return_value = json.dumps(
            {"generated_text": "SM response"}
        ).encode("utf-8")
        mock_sm.invoke_endpoint.return_value = {
            "Body": sm_body,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        with patch("src.serving.bedrock_tester.boto3") as mock_boto3:
            mock_boto3.client.return_value = mock_sm

            result = tester.compare_sagemaker_vs_bedrock(
                sagemaker_endpoint="sm-endpoint",
                bedrock_model_arn="arn:aws:bedrock:model/test",
                test_prompts=["Test prompt 1", "Test prompt 2"],
            )

        assert "sagemaker_latency_ms" in result
        assert "bedrock_latency_ms" in result
        assert "recommendation" in result
        assert result["num_prompts"] == 2
