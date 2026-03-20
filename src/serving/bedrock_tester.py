"""Testing utilities for Bedrock deployed models."""

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


class BedrockEndpointTester:
    """Test Bedrock provisioned throughput endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock-runtime client.

        Args:
            region: AWS region.
        """
        if boto3 is None:
            raise ImportError("boto3 is required for BedrockEndpointTester")
        self.region = region
        self._runtime_client: Any = boto3.client(
            "bedrock-runtime", region_name=region
        )

    def smoke_test(self, provisioned_model_arn: str) -> dict[str, Any]:
        """Run basic smoke tests against Bedrock model.

        Args:
            provisioned_model_arn: ARN of the provisioned model.

        Returns:
            Dict with passed flag and results list.
        """
        results: list[dict[str, Any]] = []
        all_passed = True

        for prompt in _SMOKE_TEST_PROMPTS:
            try:
                result = self._invoke(
                    provisioned_model_arn, prompt, max_tokens=64
                )
                passed = (
                    bool(result.get("generated_text", "").strip())
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

        logger.info("Smoke test complete", passed=all_passed)
        return {"passed": all_passed, "results": results}

    def throughput_test(
        self,
        provisioned_model_arn: str,
        target_rpm: int = 60,
        duration_minutes: int = 5,
    ) -> dict[str, Any]:
        """Test sustained throughput.

        Args:
            provisioned_model_arn: ARN of the provisioned model.
            target_rpm: Target requests per minute.
            duration_minutes: Test duration in minutes.

        Returns:
            Dict with achieved_rpm, throttle_rate, error_rate, latencies.
        """
        prompt = "Briefly explain gradient descent."
        interval = 60.0 / target_rpm
        total_requests = target_rpm * duration_minutes

        latencies: list[float] = []
        errors = 0
        throttles = 0

        for i in range(total_requests):
            start = time.perf_counter()
            try:
                result = self._invoke(
                    provisioned_model_arn, prompt, max_tokens=32
                )
                latencies.append(result.get("latency_ms", 0.0))
            except Exception as e:
                errors += 1
                if "ThrottlingException" in str(e):
                    throttles += 1

            elapsed = time.perf_counter() - start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        achieved_rpm = (
            len(latencies) / duration_minutes if duration_minutes > 0 else 0
        )
        sorted_lats = sorted(latencies) if latencies else [0]

        def _pct(data: list[float], p: float) -> float:
            idx = min(int(len(data) * p / 100), len(data) - 1)
            return round(data[idx], 2)

        return {
            "achieved_rpm": round(achieved_rpm, 2),
            "throttle_rate": round(
                throttles / max(total_requests, 1), 4
            ),
            "error_rate": round(errors / max(total_requests, 1), 4),
            "latency_p50": _pct(sorted_lats, 50),
            "latency_p99": _pct(sorted_lats, 99),
        }

    def guardrail_integration_test(
        self,
        provisioned_model_arn: str,
        guardrail_id: str,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Test guardrail behavior end-to-end.

        Send prompts that should be filtered and prompts that should pass.

        Args:
            provisioned_model_arn: ARN of the provisioned model.
            guardrail_id: Guardrail ID to apply.
            test_cases: List of {prompt, should_be_blocked} dicts.

        Returns:
            Dict with passed, failed, details.
        """
        passed = 0
        failed = 0
        details: list[dict[str, Any]] = []

        for tc in test_cases:
            prompt = tc["prompt"]
            should_block = tc.get("should_be_blocked", False)

            try:
                body = json.dumps(
                    {
                        "prompt": prompt,
                        "max_gen_len": 64,
                        "temperature": 0.1,
                    }
                )
                response = self._runtime_client.invoke_model(
                    modelId=provisioned_model_arn,
                    body=body,
                    contentType="application/json",
                    accept="application/json",
                    guardrailIdentifier=guardrail_id,
                    guardrailVersion="DRAFT",
                )

                result = json.loads(response["body"].read())
                was_blocked = "amazon-bedrock-guardrailAction" in str(response)
                test_passed = was_blocked == should_block

            except Exception as e:
                # Blocked requests may raise exceptions
                was_blocked = "blocked" in str(e).lower()
                test_passed = was_blocked == should_block
                result = {}

            if test_passed:
                passed += 1
            else:
                failed += 1

            details.append(
                {
                    "prompt": prompt,
                    "should_be_blocked": should_block,
                    "was_blocked": was_blocked,
                    "passed": test_passed,
                }
            )

        return {"passed": passed, "failed": failed, "details": details}

    def compare_sagemaker_vs_bedrock(
        self,
        sagemaker_endpoint: str,
        bedrock_model_arn: str,
        test_prompts: list[str],
    ) -> dict[str, Any]:
        """Compare identical prompts across both serving options.

        Args:
            sagemaker_endpoint: SageMaker endpoint name.
            bedrock_model_arn: Bedrock provisioned model ARN.
            test_prompts: List of prompts to test.

        Returns:
            Dict with latency comparison and recommendation.
        """
        sm_client = boto3.client(
            "sagemaker-runtime", region_name=self.region
        )

        sm_latencies: list[float] = []
        br_latencies: list[float] = []

        for prompt in test_prompts:
            # SageMaker
            payload = json.dumps(
                {"prompt": prompt, "max_new_tokens": 64, "temperature": 0.1}
            )
            start = time.perf_counter()
            try:
                sm_client.invoke_endpoint(
                    EndpointName=sagemaker_endpoint,
                    ContentType="application/json",
                    Body=payload,
                )
                sm_latencies.append(
                    (time.perf_counter() - start) * 1000
                )
            except Exception:
                sm_latencies.append(-1)

            # Bedrock
            result = self._invoke(bedrock_model_arn, prompt, max_tokens=64)
            br_latencies.append(result.get("latency_ms", -1))

        valid_sm = [l for l in sm_latencies if l > 0]
        valid_br = [l for l in br_latencies if l > 0]

        sm_mean = round(statistics.mean(valid_sm), 2) if valid_sm else -1
        br_mean = round(statistics.mean(valid_br), 2) if valid_br else -1

        if sm_mean > 0 and br_mean > 0:
            recommendation = (
                "bedrock" if br_mean < sm_mean else "sagemaker"
            )
        else:
            recommendation = "insufficient_data"

        return {
            "sagemaker_latency_ms": sm_mean,
            "bedrock_latency_ms": br_mean,
            "recommendation": recommendation,
            "num_prompts": len(test_prompts),
        }

    # ── Private helpers ─────────────────────────────────────────

    def _invoke(
        self,
        model_arn: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Single model invocation.

        Args:
            model_arn: Model ARN.
            prompt: Prompt string.
            max_tokens: Max generation tokens.
            temperature: Sampling temperature.

        Returns:
            Dict with generated_text and latency_ms.
        """
        body = json.dumps(
            {
                "prompt": prompt,
                "max_gen_len": max_tokens,
                "temperature": temperature,
            }
        )

        start = time.perf_counter()
        response = self._runtime_client.invoke_model(
            modelId=model_arn,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        latency_ms = (time.perf_counter() - start) * 1000

        result = json.loads(response["body"].read())
        return {
            "generated_text": result.get("generation", ""),
            "latency_ms": round(latency_ms, 2),
        }


__all__: list[str] = ["BedrockEndpointTester"]
