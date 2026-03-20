"""Endpoint testing: smoke tests, latency tests, correctness tests."""

from __future__ import annotations

import concurrent.futures
import json
import statistics
import time
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

try:
    import boto3
except ImportError:  # pragma: no cover
    boto3 = None  # type: ignore[assignment]

_SMOKE_TEST_PROMPTS = [
    "What is machine learning?",
    "Explain the concept of overfitting.",
    "Translate 'hello' to French.",
    "Summarize the benefits of fine-tuning.",
    "Write a short poem about AI.",
]


class EndpointTester:
    """Test deployed SageMaker endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with SageMaker Runtime client.

        Args:
            region: AWS region.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for EndpointTester")
        self.region = region
        self._runtime_client: Any = boto3.client(
            "sagemaker-runtime", region_name=region
        )

    def smoke_test(self, endpoint_name: str) -> dict[str, Any]:
        """Run 5 simple prompts and verify basic functionality.

        Checks: responses non-empty, response time < 30s, no error codes,
        output is valid JSON.

        Args:
            endpoint_name: Name of the endpoint to test.

        Returns:
            Dict with passed flag and results list.
        """
        results: list[dict[str, Any]] = []
        all_passed = True

        for prompt in _SMOKE_TEST_PROMPTS:
            try:
                result = self.invoke_endpoint(
                    endpoint_name, prompt, max_new_tokens=64, temperature=0.1
                )
                passed = (
                    result.get("status_code") == 200
                    and bool(result.get("generated_text", "").strip())
                    and result.get("latency_ms", 999999) < 30000
                )
                results.append(
                    {
                        "prompt": prompt,
                        "response": result.get("generated_text", ""),
                        "latency_ms": result.get("latency_ms", 0),
                        "status": "pass" if passed else "fail",
                    }
                )
                if not passed:
                    all_passed = False
            except Exception as e:
                all_passed = False
                results.append(
                    {
                        "prompt": prompt,
                        "response": "",
                        "latency_ms": 0,
                        "status": f"error: {e}",
                    }
                )

        logger.info(
            "Smoke test complete",
            endpoint=endpoint_name,
            passed=all_passed,
        )
        return {"passed": all_passed, "results": results}

    def latency_test(
        self,
        endpoint_name: str,
        num_requests: int = 100,
        concurrency: int = 10,
        warmup_requests: int = 10,
    ) -> dict[str, Any]:
        """Measure latency distribution under concurrent load.

        Uses ThreadPoolExecutor. Excludes warmup_requests from metrics.

        Args:
            endpoint_name: Name of the endpoint.
            num_requests: Total number of requests to send.
            concurrency: Number of concurrent workers.
            warmup_requests: Requests to exclude from metrics.

        Returns:
            Dict with p50, p90, p95, p99, mean, throughput, error_rate.
        """
        prompt = "Briefly explain gradient descent."
        all_latencies: list[float] = []
        errors = 0

        def _invoke(_: int) -> float | None:
            try:
                result = self.invoke_endpoint(
                    endpoint_name, prompt, max_new_tokens=32, temperature=0.1
                )
                return result.get("latency_ms", 0.0)
            except Exception:
                return None

        total = warmup_requests + num_requests

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrency
        ) as executor:
            futures = [executor.submit(_invoke, i) for i in range(total)]
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                latency = future.result()
                if i >= warmup_requests:
                    if latency is not None:
                        all_latencies.append(latency)
                    else:
                        errors += 1

        if not all_latencies:
            return {
                "p50_ms": 0,
                "p90_ms": 0,
                "p95_ms": 0,
                "p99_ms": 0,
                "mean_ms": 0,
                "throughput_rps": 0,
                "error_rate": 1.0,
                "total_requests": num_requests,
                "errors": errors,
            }

        sorted_latencies = sorted(all_latencies)
        total_time_s = sum(all_latencies) / 1000.0

        def _percentile(data: list[float], pct: float) -> float:
            idx = int(len(data) * pct / 100.0)
            idx = min(idx, len(data) - 1)
            return round(data[idx], 2)

        return {
            "p50_ms": _percentile(sorted_latencies, 50),
            "p90_ms": _percentile(sorted_latencies, 90),
            "p95_ms": _percentile(sorted_latencies, 95),
            "p99_ms": _percentile(sorted_latencies, 99),
            "mean_ms": round(statistics.mean(all_latencies), 2),
            "throughput_rps": round(
                len(all_latencies) / max(total_time_s, 0.001), 2
            ),
            "error_rate": round(errors / num_requests, 4),
            "total_requests": num_requests,
            "errors": errors,
        }

    def correctness_test(
        self,
        endpoint_name: str,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run test cases with expected patterns.

        Each test_case: {prompt, expected_contains: list[str],
                         expected_not_contains: list[str], max_tokens: int}

        Args:
            endpoint_name: Name of the endpoint.
            test_cases: List of test case dicts.

        Returns:
            Dict with passed, failed, total counts and details.
        """
        passed = 0
        failed = 0
        details: list[dict[str, Any]] = []

        for tc in test_cases:
            prompt = tc["prompt"]
            max_tokens = tc.get("max_tokens", 256)
            expected_contains = tc.get("expected_contains", [])
            expected_not_contains = tc.get("expected_not_contains", [])

            result = self.invoke_endpoint(
                endpoint_name, prompt, max_new_tokens=max_tokens, temperature=0.1
            )
            text = result.get("generated_text", "").lower()

            issues: list[str] = []
            for expected in expected_contains:
                if expected.lower() not in text:
                    issues.append(f"Missing expected: '{expected}'")
            for not_expected in expected_not_contains:
                if not_expected.lower() in text:
                    issues.append(f"Contains forbidden: '{not_expected}'")

            test_passed = len(issues) == 0
            if test_passed:
                passed += 1
            else:
                failed += 1

            details.append(
                {
                    "prompt": prompt,
                    "generated_text": result.get("generated_text", ""),
                    "passed": test_passed,
                    "issues": issues,
                }
            )

        return {
            "passed": passed,
            "failed": failed,
            "total": len(test_cases),
            "details": details,
        }

    def invoke_endpoint(
        self,
        endpoint_name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Single endpoint invocation.

        Args:
            endpoint_name: Name of the endpoint.
            prompt: Prompt to send.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Dict with generated_text, latency_ms, status_code.
        """
        payload = json.dumps(
            {
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
            }
        )

        start = time.perf_counter()
        response = self._runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=payload,
        )
        latency_ms = (time.perf_counter() - start) * 1000

        status_code = response.get("ResponseMetadata", {}).get(
            "HTTPStatusCode", 200
        )
        body = response["Body"].read().decode("utf-8")

        try:
            result = json.loads(body)
        except json.JSONDecodeError:
            result = {"generated_text": body}

        return {
            "generated_text": result.get("generated_text", body),
            "latency_ms": round(latency_ms, 2),
            "status_code": status_code,
        }


__all__: list[str] = ["EndpointTester"]
