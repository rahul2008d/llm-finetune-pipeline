# Phase 10-11: SageMaker Endpoint & Bedrock Deployment

## Context for Copilot

Training, evaluation, and model registry are complete. Now I need to deploy models to SageMaker endpoints and Bedrock. The following files already exist — check them first and enhance rather than replace.

**Existing serving files:**
- `src/serving/endpoint.py` — may have `SageMakerEndpointHandler`
- `src/serving/bedrock.py` — may have `BedrockImporter`
- `src/serving/inference.py` — SageMaker inference handler (from Phase 9)
- `src/serving/model_registry.py` — `ModelRegistryManager` (from Phase 9)
- `src/serving/artifact_packager.py` — `ArtifactPackager` (from Phase 9)

**Existing config/CLI:**
- `src/cli.py` — has `deploy sagemaker` and `deploy bedrock` commands
- `configs/deployment/` — may have deployment YAML files

**Rules:** Same as previous phases — absolute imports, type hints, docstrings, structlog, tests for everything.

---

## Prompt 37 — SageMaker Endpoint Manager

Check `src/serving/endpoint.py`. If it has a stub `SageMakerEndpointHandler`, replace the class body. If it's well-built, add missing methods. The class should be named `SageMakerEndpointManager`.

```python
"""SageMaker real-time endpoint management with blue-green deployment."""
```

### Class: `SageMakerEndpointManager`

```python
class SageMakerEndpointManager:
    """Create, update, and manage SageMaker real-time endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 SageMaker client."""

    def create_endpoint(
        self,
        model_package_arn: str | None = None,
        model_data_url: str | None = None,
        endpoint_name: str = "",
        instance_type: str = "ml.g5.xlarge",
        initial_instance_count: int = 1,
        role_arn: str = "",
        container_image: str | None = None,
        data_capture_enabled: bool = False,
        data_capture_s3_uri: str = "",
        data_capture_sampling_pct: int = 100,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Create a new SageMaker endpoint.

        1. Create Model (from model_package_arn or model_data_url + container)
        2. Create EndpointConfig with:
           - ProductionVariant
           - DataCaptureConfig (if enabled)
        3. Create Endpoint
        4. Wait for InService status (with timeout of 15 minutes)
        5. Return dict with: endpoint_name, endpoint_arn, status, url
        """

    def update_endpoint_traffic(
        self,
        endpoint_name: str,
        target_variant: str,
        target_weight: int,
    ) -> None:
        """Shift traffic between variants for canary/A-B deployment."""

    def blue_green_deploy(
        self,
        endpoint_name: str,
        new_model_data_url: str,
        new_model_package_arn: str | None = None,
        instance_type: str = "ml.g5.xlarge",
        canary_pct: float = 0.1,
        bake_time_minutes: int = 30,
        rollback_alarm_names: list[str] | None = None,
        role_arn: str = "",
        container_image: str | None = None,
    ) -> dict[str, Any]:
        """Blue-green deployment with canary traffic shifting.

        1. Create new model variant with new model
        2. Update endpoint config to add new variant at canary_pct weight
        3. Apply the update to the endpoint
        4. Monitor for bake_time_minutes:
           - Poll CloudWatch alarms if rollback_alarm_names provided
           - If any alarm fires -> rollback (shift all traffic to old variant)
        5. If stable -> shift 100% to new variant, remove old variant
        6. Return: {status, timeline, old_variant, new_variant, rolled_back}
        """

    def delete_endpoint(self, endpoint_name: str, delete_model: bool = True) -> None:
        """Delete endpoint, endpoint config, and optionally the model."""

    def describe_endpoint(self, endpoint_name: str) -> dict[str, Any]:
        """Full endpoint status, variants, traffic distribution.

        Return: {status, endpoint_arn, variants: [{name, instance_type, weight, status}],
                 creation_time, last_modified}
        """

    def list_endpoints(self, name_contains: str = "") -> list[dict[str, Any]]:
        """List endpoints, optionally filtered by name prefix."""
```

### Tests: `tests/unit/test_serving/test_sagemaker_endpoint.py`

- `test_create_endpoint_calls_apis_in_order` — mock boto3, verify create_model -> create_endpoint_config -> create_endpoint
- `test_create_endpoint_waits_for_inservice` — mock waiter
- `test_delete_endpoint_cleans_up` — verify all 3 resources deleted
- `test_describe_endpoint_returns_info` — verify dict structure
- `test_blue_green_deploy_shifts_traffic` — verify traffic update calls
- `test_blue_green_rollback_on_alarm` — mock alarm in ALARM state, verify rollback

---

## Prompt 38 — Auto-Scaling Configuration

Create NEW file `src/serving/autoscaling.py`:

```python
"""Auto-scaling configuration for SageMaker endpoints."""
```

### Class: `EndpointAutoScaler`

```python
class EndpointAutoScaler:
    """Configure auto-scaling policies for SageMaker endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with Application Auto Scaling client."""

    def configure_autoscaling(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        min_instances: int = 1,
        max_instances: int = 4,
        target_invocations_per_instance: int = 50,
        scale_in_cooldown: int = 300,
        scale_out_cooldown: int = 60,
    ) -> None:
        """Configure target tracking scaling policy.

        1. Register scalable target with Application Auto Scaling
        2. Create target tracking policy on InvocationsPerInstance metric
        """

    def configure_scheduled_scaling(
        self,
        endpoint_name: str,
        variant_name: str = "AllTraffic",
        schedules: list[dict[str, Any]] | None = None,
    ) -> None:
        """Configure scheduled scaling actions.

        Example schedule: scale to 4 instances 9AM-5PM, 1 instance overnight.
        Each schedule dict: {name, schedule_expression (cron), min_capacity, max_capacity}
        """

    def get_scaling_status(self, endpoint_name: str) -> dict[str, Any]:
        """Get current scaling status: instance count, pending activities, recent events."""

    def remove_autoscaling(self, endpoint_name: str, variant_name: str = "AllTraffic") -> None:
        """Remove all scaling policies and deregister the scalable target."""
```

### Tests: `tests/unit/test_serving/test_autoscaling.py`

- `test_configure_autoscaling_registers_target` — verify register_scalable_target called
- `test_configure_autoscaling_creates_policy` — verify put_scaling_policy called
- `test_scheduled_scaling` — verify put_scheduled_action called
- `test_get_scaling_status` — verify describe calls
- `test_remove_autoscaling` — verify deregister called

---

## Prompt 39 — Endpoint Testing & Smoke Tests

Create NEW file `src/serving/endpoint_tester.py`:

```python
"""Endpoint testing: smoke tests, latency tests, correctness tests."""
```

### Class: `EndpointTester`

```python
class EndpointTester:
    """Test deployed SageMaker endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with SageMaker Runtime client."""

    def smoke_test(self, endpoint_name: str) -> dict[str, Any]:
        """Run 5 simple prompts and verify basic functionality.

        Checks: responses non-empty, response time < 30s, no error codes,
        output is valid JSON.
        Return: {passed: bool, results: list[{prompt, response, latency_ms, status}]}
        """

    def latency_test(
        self,
        endpoint_name: str,
        num_requests: int = 100,
        concurrency: int = 10,
        warmup_requests: int = 10,
    ) -> dict[str, Any]:
        """Measure latency distribution under concurrent load.

        Use concurrent.futures ThreadPoolExecutor.
        Exclude warmup_requests from metrics.
        Return: {p50_ms, p90_ms, p95_ms, p99_ms, mean_ms, throughput_rps,
                 error_rate, total_requests, errors}
        """

    def correctness_test(
        self,
        endpoint_name: str,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run test cases with expected patterns.

        Each test_case: {prompt, expected_contains: list[str],
                         expected_not_contains: list[str], max_tokens: int}
        Return: {passed: int, failed: int, total: int, details: list[...]}
        """

    def invoke_endpoint(
        self,
        endpoint_name: str,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """Single endpoint invocation.

        Return: {generated_text, latency_ms, status_code}
        """
```

### Tests: `tests/unit/test_serving/test_endpoint_tester.py`

- `test_smoke_test_passes_on_valid_responses` — mock invoke, verify pass
- `test_smoke_test_fails_on_error` — mock 500 response, verify fail
- `test_latency_test_returns_percentiles` — verify all percentile keys
- `test_correctness_test_pass` — mock matching response
- `test_correctness_test_fail` — mock non-matching response

---

## Prompt 40 — Deployment Config Files

Create deployment configuration YAML files:

### `configs/deployment/sagemaker_dev.yaml`
```yaml
endpoint_name: llm-ft-dev-v1
instance_type: ml.g5.xlarge
initial_instance_count: 1
role_arn: "${ssm:/llm-finetune/sagemaker-endpoint-role-arn}"
autoscaling:
  enabled: false
data_capture:
  enabled: false
smoke_test:
  required: false
tags:
  Environment: dev
  Project: llm-finetune
```

### `configs/deployment/sagemaker_staging.yaml`
```yaml
endpoint_name: llm-ft-staging-v1
instance_type: ml.g5.xlarge
initial_instance_count: 1
role_arn: "${ssm:/llm-finetune/sagemaker-endpoint-role-arn}"
autoscaling:
  enabled: true
  min_instances: 1
  max_instances: 2
  target_invocations: 30
data_capture:
  enabled: true
  sampling_percentage: 100
  s3_uri: "s3://llm-finetune-staging-model-artifacts/data-capture/"
smoke_test:
  required: true
tags:
  Environment: staging
  Project: llm-finetune
```

### `configs/deployment/sagemaker_prod.yaml`
```yaml
endpoint_name: llm-ft-prod-v1
instance_type: ml.g5.2xlarge
initial_instance_count: 2
role_arn: "${ssm:/llm-finetune/sagemaker-endpoint-role-arn}"
autoscaling:
  enabled: true
  min_instances: 2
  max_instances: 8
  target_invocations: 50
  scale_in_cooldown: 300
  scale_out_cooldown: 60
data_capture:
  enabled: true
  sampling_percentage: 10
  s3_uri: "s3://llm-finetune-prod-model-artifacts/data-capture/"
blue_green:
  enabled: true
  canary_pct: 0.1
  bake_time_minutes: 30
  rollback_alarm_names:
    - "llm-finetune-prod-endpoint-5xx-rate"
    - "llm-finetune-prod-endpoint-latency-p99"
smoke_test:
  required: true
latency_test:
  required: true
  threshold_p99_ms: 5000
tags:
  Environment: production
  Project: llm-finetune
```

### `configs/deployment/bedrock.yaml`
```yaml
model_name: my-custom-llm-v1
model_source_s3_uri: "s3://llm-finetune-prod-model-artifacts/bedrock-models/model-v1/"
role_arn: "${ssm:/llm-finetune/bedrock-import-role-arn}"
provisioned_throughput:
  model_units: 1
  commitment: NO_COMMITMENT  # or ONE_MONTH, SIX_MONTH
tags:
  Environment: production
  Project: llm-finetune
```

---

## Prompt 41 — Bedrock Import Manager

Check `src/serving/bedrock.py`. If it has a stub `BedrockImporter`, enhance it to be production-grade. The class should be named `BedrockImportManager`.

```python
"""AWS Bedrock Custom Model Import and provisioned throughput management."""
```

### Class: `BedrockImportManager`

```python
class BedrockImportManager:
    """Import custom models into Bedrock and manage provisioned throughput."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock and bedrock-runtime clients."""

    def import_model(
        self,
        model_name: str,
        model_s3_uri: str,
        role_arn: str,
        tags: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Import a model into Bedrock.

        1. Validate model artifacts at S3 URI:
           - config.json exists and has valid architecture
           - tokenizer files exist
           - safetensors files exist and sum < 50GB
           - Supported: LlamaForCausalLM, MistralForCausalLM, PhiForCausalLM
        2. Call create_model_import_job
        3. Poll get_model_import_job until Complete or Failed (timeout 60 min)
        4. On failure: fetch error details, log, raise
        5. Return: {model_arn, import_job_arn, status}
        """

    def create_provisioned_throughput(
        self,
        model_arn: str,
        throughput_name: str,
        model_units: int = 1,
        commitment: str = "NO_COMMITMENT",
    ) -> dict[str, Any]:
        """Create provisioned throughput for the imported model.

        Wait for Active status (timeout 30 min).
        Return: {provisioned_model_arn, status}
        """

    def invoke_model(
        self,
        provisioned_model_arn: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> dict[str, Any]:
        """Invoke the model via bedrock-runtime.

        Return: {generated_text, input_tokens, output_tokens, latency_ms}
        """

    def delete_model(self, model_arn: str) -> None:
        """Delete provisioned throughput first (if any), then delete the model."""

    def list_custom_models(self) -> list[dict[str, Any]]:
        """List all custom imported models."""
```

### Tests: `tests/unit/test_serving/test_bedrock_import.py`

- `test_import_model_validates_artifacts` — mock S3, verify validation
- `test_import_model_polls_until_complete` — mock polling
- `test_create_provisioned_throughput` — verify API call
- `test_invoke_model_returns_text` — mock bedrock-runtime
- `test_delete_model_cleans_up_throughput_first` — verify order
- `test_import_model_fails_on_unsupported_architecture` — verify error

---

## Prompt 42 — Bedrock Guardrails

Create NEW file `src/serving/bedrock_guardrails.py`:

```python
"""Bedrock Guardrails configuration and management."""
```

### Class: `GuardrailsManager`

```python
class GuardrailsManager:
    """Create and manage Bedrock Guardrails for content safety."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock client."""

    def create_guardrail(
        self,
        name: str,
        description: str = "",
        content_filters: dict[str, str] | None = None,
        denied_topics: list[dict[str, str]] | None = None,
        word_filters: list[str] | None = None,
        pii_entity_types: list[str] | None = None,
        pii_action: str = "ANONYMIZE",
    ) -> str:
        """Create a Bedrock Guardrail.

        content_filters: {hate, insults, sexual, violence, misconduct}
                         values: NONE, LOW, MEDIUM, HIGH
        denied_topics: [{name, definition, examples, type}]
        word_filters: list of blocked words/phrases
        pii_entity_types: [EMAIL, PHONE, SSN, CREDIT_CARD, etc.]
        pii_action: ANONYMIZE or BLOCK

        Return guardrail_id.
        """

    def test_guardrail(
        self,
        guardrail_id: str,
        test_prompts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Test guardrail with sample prompts.

        Each test_prompt: {text, should_be_blocked: bool}
        Return: {passed, failed, total, false_positives, false_negatives, details}
        """

    def delete_guardrail(self, guardrail_id: str) -> None:
        """Delete a guardrail."""

    def list_guardrails(self) -> list[dict[str, Any]]:
        """List all guardrails."""
```

### Tests: `tests/unit/test_serving/test_bedrock_guardrails.py`

- `test_create_guardrail_with_content_filters` — verify API call structure
- `test_create_guardrail_with_pii` — verify PII config passed
- `test_test_guardrail_reports_results` — verify pass/fail counting
- `test_delete_guardrail` — verify API call

---

## Prompt 43 — Bedrock Endpoint Tester

Create NEW file `src/serving/bedrock_tester.py`:

```python
"""Testing utilities for Bedrock deployed models."""
```

### Class: `BedrockEndpointTester`

```python
class BedrockEndpointTester:
    """Test Bedrock provisioned throughput endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with bedrock-runtime client."""

    def smoke_test(self, provisioned_model_arn: str) -> dict[str, Any]:
        """Run basic smoke tests against Bedrock model.

        Same pattern as SageMaker EndpointTester.smoke_test.
        """

    def throughput_test(
        self,
        provisioned_model_arn: str,
        target_rpm: int = 60,
        duration_minutes: int = 5,
    ) -> dict[str, Any]:
        """Test sustained throughput.

        Return: {achieved_rpm, throttle_rate, error_rate, latency_p50, latency_p99}
        """

    def guardrail_integration_test(
        self,
        provisioned_model_arn: str,
        guardrail_id: str,
        test_cases: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Test guardrail behavior end-to-end.

        Send prompts that should be filtered and prompts that should pass.
        Return: {passed, failed, details}
        """

    def compare_sagemaker_vs_bedrock(
        self,
        sagemaker_endpoint: str,
        bedrock_model_arn: str,
        test_prompts: list[str],
    ) -> dict[str, Any]:
        """Compare identical prompts across both serving options.

        Return: {sagemaker_latency, bedrock_latency, output_similarity, recommendation}
        """
```

### Tests: `tests/unit/test_serving/test_bedrock_tester.py`

- `test_smoke_test` — mock invoke, verify structure
- `test_throughput_test` — verify RPM calculation
- `test_compare_returns_recommendation` — verify both endpoints called

---

## Summary of Phase 10-11 files

### New files to CREATE:
1. `src/serving/autoscaling.py`
2. `src/serving/endpoint_tester.py`
3. `src/serving/bedrock_guardrails.py`
4. `src/serving/bedrock_tester.py`
5. `configs/deployment/sagemaker_dev.yaml`
6. `configs/deployment/sagemaker_staging.yaml`
7. `configs/deployment/sagemaker_prod.yaml`
8. `configs/deployment/bedrock.yaml`
9. `tests/unit/test_serving/test_sagemaker_endpoint.py`
10. `tests/unit/test_serving/test_autoscaling.py`
11. `tests/unit/test_serving/test_endpoint_tester.py`
12. `tests/unit/test_serving/test_bedrock_import.py`
13. `tests/unit/test_serving/test_bedrock_guardrails.py`
14. `tests/unit/test_serving/test_bedrock_tester.py`

### Files to ENHANCE (check first, then add):
1. `src/serving/endpoint.py` — enhance to `SageMakerEndpointManager`
2. `src/serving/bedrock.py` — enhance to `BedrockImportManager`
