"""Tests for endpoint testing utilities."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.serving.endpoint_tester import EndpointTester


@pytest.fixture()
def mock_runtime_client() -> MagicMock:
    """Create a mock SageMaker Runtime client."""
    client = MagicMock()

    def _make_response(text: str = "Test response") -> dict[str, Any]:
        body = MagicMock()
        body.read.return_value = json.dumps(
            {"generated_text": text}
        ).encode("utf-8")
        return {
            "Body": body,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

    client.invoke_endpoint.return_value = _make_response()
    return client


@pytest.fixture()
def tester(mock_runtime_client: MagicMock) -> EndpointTester:
    """Create a EndpointTester with mocked boto3."""
    with patch("src.serving.endpoint_tester.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_runtime_client
        t = EndpointTester(region="us-east-1")
    return t


class TestSmokeTest:
    """Tests for smoke_test method."""

    def test_smoke_test_passes_on_valid_responses(
        self, tester: EndpointTester
    ) -> None:
        """Verify pass when all responses are valid."""
        result = tester.smoke_test("test-endpoint")

        assert result["passed"] is True
        assert len(result["results"]) == 5
        for r in result["results"]:
            assert r["status"] == "pass"

    def test_smoke_test_fails_on_error(
        self, tester: EndpointTester
    ) -> None:
        """Verify fail when endpoint returns error."""
        tester._runtime_client.invoke_endpoint.side_effect = Exception(
            "500 Internal Server Error"
        )

        result = tester.smoke_test("test-endpoint")

        assert result["passed"] is False
        for r in result["results"]:
            assert "error" in r["status"]


class TestLatencyTest:
    """Tests for latency_test method."""

    def test_latency_test_returns_percentiles(
        self, tester: EndpointTester
    ) -> None:
        """Verify all percentile keys in result."""
        result = tester.latency_test(
            "test-endpoint",
            num_requests=5,
            concurrency=2,
            warmup_requests=1,
        )

        assert "p50_ms" in result
        assert "p90_ms" in result
        assert "p95_ms" in result
        assert "p99_ms" in result
        assert "mean_ms" in result
        assert "throughput_rps" in result
        assert "error_rate" in result
        assert result["total_requests"] == 5


class TestCorrectnessTest:
    """Tests for correctness_test method."""

    def test_correctness_test_pass(self, tester: EndpointTester) -> None:
        """Verify pass when response matches expected pattern."""
        # Mock response containing expected text
        body = MagicMock()
        body.read.return_value = json.dumps(
            {"generated_text": "Machine learning is a subset of AI."}
        ).encode("utf-8")
        tester._runtime_client.invoke_endpoint.return_value = {
            "Body": body,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        test_cases = [
            {
                "prompt": "What is ML?",
                "expected_contains": ["machine learning"],
                "expected_not_contains": ["error"],
            }
        ]

        result = tester.correctness_test("test-endpoint", test_cases)
        assert result["passed"] == 1
        assert result["failed"] == 0

    def test_correctness_test_fail(self, tester: EndpointTester) -> None:
        """Verify fail when response doesn't match."""
        body = MagicMock()
        body.read.return_value = json.dumps(
            {"generated_text": "I don't know."}
        ).encode("utf-8")
        tester._runtime_client.invoke_endpoint.return_value = {
            "Body": body,
            "ResponseMetadata": {"HTTPStatusCode": 200},
        }

        test_cases = [
            {
                "prompt": "What is ML?",
                "expected_contains": ["machine learning"],
            }
        ]

        result = tester.correctness_test("test-endpoint", test_cases)
        assert result["failed"] == 1
        assert result["passed"] == 0
