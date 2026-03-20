# Phase 12-14: CI/CD, Monitoring, Guardrails & Operations

## Context for Copilot

All training, evaluation, and deployment code is complete. Now I need CI/CD pipelines, production monitoring, and operational runbooks.

**Existing files:**
- `.github/workflows/ci.yml` — may exist with basic lint/test
- `src/monitoring/cloudwatch.py` — `CloudWatchMetrics`, `TrainingMetricsPublisher`
- `src/monitoring/alerting.py` — `AlertManager`
- `src/monitoring/drift.py` — `DriftDetector`
- `src/serving/endpoint_tester.py` — `EndpointTester`
- `src/serving/bedrock_guardrails.py` — `GuardrailsManager`

**Rules:** Same as previous phases.

---

## Prompt 44 — GitHub Actions: Lint & Test

Check if `.github/workflows/ci.yml` exists. If it does, enhance it. If not, create it.

Create `.github/workflows/ci.yml`:

```yaml
name: CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  lint-and-type-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - name: Install uv
        uses: astral-sh/setup-uv@v5
      - name: Install dependencies
        run: uv sync --extra dev --extra data
      - name: Lint
        run: uv run ruff check src/ tests/
      - name: Format check
        run: uv run ruff format --check src/ tests/
      - name: Type check
        run: uv run mypy src/
      - name: Secret scan
        run: uv run detect-secrets scan src/

  unit-tests:
    runs-on: ubuntu-latest
    needs: lint-and-type-check
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev --extra data
      - name: Run unit tests
        run: uv run pytest tests/unit/ -v --cov=src --cov-report=xml --timeout=300
      - name: Upload coverage
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report
          path: coverage.xml
      - name: Check coverage threshold
        run: |
          coverage=$(uv run python -c "import xml.etree.ElementTree as ET; \
            t=ET.parse('coverage.xml'); \
            print(float(t.getroot().get('line-rate',0))*100)")
          echo "Coverage: ${coverage}%"
          uv run python -c "assert float('${coverage}') >= 70, f'Coverage {${coverage}}% below 70%'"

  integration-tests:
    runs-on: ubuntu-latest
    needs: unit-tests
    services:
      localstack:
        image: localstack/localstack:latest
        ports:
          - "4566:4566"
    env:
      AWS_ENDPOINT_URL: http://localhost:4566
      AWS_DEFAULT_REGION: us-east-1
      AWS_ACCESS_KEY_ID: test
      AWS_SECRET_ACCESS_KEY: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev --extra data
      - name: Run integration tests
        run: uv run pytest tests/integration/ -v --timeout=300

  security-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - name: Dependency audit
        run: uv run pip-audit 2>/dev/null || echo "pip-audit not available, skipping"
      - name: Security lint
        run: uv run bandit -r src/ -ll 2>/dev/null || echo "bandit not available, skipping"
```

---

## Prompt 45 — GitHub Actions: Docker Build & Push

Create `.github/workflows/docker.yml`:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [main]
    paths:
      - "docker/**"
      - "src/**"
      - "requirements*.txt"
      - "pyproject.toml"
  workflow_dispatch: {}

permissions:
  id-token: write
  contents: read

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [train, serve]
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}

      - name: Login to Amazon ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push
        uses: docker/build-push-action@v6
        with:
          context: .
          file: docker/Dockerfile.${{ matrix.target }}
          push: true
          tags: |
            ${{ steps.ecr-login.outputs.registry }}/llm-finetune-${{ matrix.target }}:${{ github.sha }}
            ${{ steps.ecr-login.outputs.registry }}/llm-finetune-${{ matrix.target }}:latest
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64
```

---

## Prompt 46 — GitHub Actions: Training Pipeline

Create `.github/workflows/train.yml`:

```yaml
name: Training Pipeline

on:
  workflow_dispatch:
    inputs:
      config_file:
        description: "Training config path"
        required: true
        default: "configs/training/qlora_llama3_8b.yaml"
      dataset_path:
        description: "S3 URI or local path to dataset"
        required: true
      run_evaluation:
        description: "Run eval after training"
        type: boolean
        default: true
      deploy_to:
        description: "Deploy target"
        type: choice
        options:
          - none
          - dev
          - staging

permissions:
  id-token: write
  contents: read

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra dev
      - name: Validate config
        run: |
          uv run python -c "
          from src.config.training import TrainingJobConfig
          import yaml
          with open('${{ inputs.config_file }}') as f:
              cfg = yaml.safe_load(f)
          TrainingJobConfig.model_validate(cfg)
          print('Config validation passed')
          "
      - name: Estimate cost
        run: |
          echo "Estimated cost: check config for instance type and epochs"
          echo "Config: ${{ inputs.config_file }}"

  train:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Submit SageMaker training job
        run: |
          uv run llm-ft train sagemaker \
            --config ${{ inputs.config_file }} \
            --yes
        timeout-minutes: 1440  # 24 hours

  evaluate:
    needs: train
    if: ${{ inputs.run_evaluation }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync --extra eval
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Run evaluation
        run: |
          uv run llm-ft evaluate \
            --model-path ${{ needs.train.outputs.model_path || 's3://placeholder' }} \
            --benchmark mmlu,hellaswag,arc_challenge \
            --output-dir ./results/ \
            --yes

  deploy:
    needs: evaluate
    if: ${{ inputs.deploy_to != 'none' }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Deploy
        run: |
          uv run llm-ft deploy sagemaker \
            --model ${{ needs.train.outputs.model_s3_uri || 's3://placeholder' }} \
            --endpoint-name llm-ft-${{ inputs.deploy_to }}-v1 \
            --instance ml.g5.xlarge \
            --yes
      - name: Smoke test
        run: |
          uv run python -c "
          from src.serving.endpoint_tester import EndpointTester
          tester = EndpointTester()
          result = tester.smoke_test('llm-ft-${{ inputs.deploy_to }}-v1')
          assert result['passed'], f'Smoke test failed: {result}'
          print('Smoke test passed')
          "
```

---

## Prompt 47 — GitHub Actions: Production Deployment

Create `.github/workflows/deploy-prod.yml`:

```yaml
name: Production Deployment

on:
  workflow_dispatch:
    inputs:
      model_s3_uri:
        description: "S3 URI of approved model artifacts"
        required: true
      deploy_target:
        type: choice
        options:
          - sagemaker
          - bedrock
          - both
      skip_bake:
        type: boolean
        default: false
        description: "Skip bake time (emergency only)"

concurrency:
  group: production-deploy
  cancel-in-progress: false

permissions:
  id-token: write
  contents: read

jobs:
  pre-deploy-checks:
    runs-on: ubuntu-latest
    environment: production  # requires manual approval
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Verify model artifacts
        run: |
          uv run python -c "
          from src.serving.artifact_packager import ArtifactPackager
          packager = ArtifactPackager()
          result = packager.verify_artifact('${{ inputs.model_s3_uri }}', 'sagemaker')
          assert result['is_valid'], f'Artifact verification failed: {result[\"issues\"]}'
          print(f'Model verified: {result[\"size_gb\"]}GB')
          "

  deploy-sagemaker:
    needs: pre-deploy-checks
    if: contains(fromJSON('["sagemaker", "both"]'), inputs.deploy_target)
    environment: production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Blue-green deploy
        run: |
          uv run python -c "
          from src.serving.endpoint import SageMakerEndpointManager
          mgr = SageMakerEndpointManager()
          result = mgr.blue_green_deploy(
              endpoint_name='llm-ft-prod-v1',
              new_model_data_url='${{ inputs.model_s3_uri }}',
              canary_pct=0.1,
              bake_time_minutes=${{ inputs.skip_bake && 0 || 30 }},
              rollback_alarm_names=['llm-finetune-prod-endpoint-5xx-rate'],
          )
          print(f'Deployment result: {result}')
          assert result.get('status') != 'rolled_back', 'Deployment rolled back!'
          "
      - name: Smoke test production
        run: |
          uv run python -c "
          from src.serving.endpoint_tester import EndpointTester
          tester = EndpointTester()
          result = tester.smoke_test('llm-ft-prod-v1')
          assert result['passed'], f'Production smoke test failed: {result}'
          "

  deploy-bedrock:
    needs: pre-deploy-checks
    if: contains(fromJSON('["bedrock", "both"]'), inputs.deploy_target)
    environment: production
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - name: Configure AWS
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_OIDC_ROLE_ARN }}
          aws-region: ${{ vars.AWS_REGION || 'us-east-1' }}
      - name: Import to Bedrock
        run: |
          uv run python -c "
          from src.serving.bedrock import BedrockImportManager
          mgr = BedrockImportManager()
          result = mgr.import_model(
              model_name='llm-ft-prod',
              model_s3_uri='${{ inputs.model_s3_uri }}',
              role_arn='arn:aws:iam::role/bedrock-import',
          )
          print(f'Import result: {result}')
          "

  post-deploy:
    needs: [deploy-sagemaker, deploy-bedrock]
    if: always()
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Tag release
        run: |
          git tag "deployed/prod/$(date +%Y%m%d-%H%M%S)"
          git push --tags
```

---

## Prompt 48 — Production Monitoring Stack

Create NEW file `src/monitoring/endpoint_monitor.py`:

```python
"""Production endpoint monitoring: SageMaker Model Monitor + CloudWatch alarms."""
```

### Class: `EndpointMonitor`

```python
class EndpointMonitor:
    """Configure and manage production monitoring for deployed endpoints."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 clients."""

    def setup_monitoring(
        self,
        endpoint_name: str,
        alert_sns_topic_arn: str,
        latency_p99_threshold_ms: int = 5000,
        error_rate_threshold_pct: float = 1.0,
        enable_data_quality: bool = True,
    ) -> dict[str, Any]:
        """Configure full monitoring stack for an endpoint.

        1. Create CloudWatch alarms:
           - Zero invocations for 15 min -> warning
           - P99 latency > threshold -> critical
           - 5xx rate > threshold -> critical
           - Unhealthy instance count > 0 -> critical
        2. Create CloudWatch Dashboard:
           - Invocation count + latency (p50/p90/p99)
           - Error rates
           - Instance health
        3. Return: {alarm_arns, dashboard_url}
        """

    def get_monitoring_report(
        self,
        endpoint_name: str,
        hours: int = 24,
    ) -> dict[str, Any]:
        """Generate monitoring report for last N hours.

        Return: {uptime_pct, error_rate, latency_stats, invocation_count, cost_estimate}
        """
```

### Tests: `tests/unit/test_monitoring/test_endpoint_monitor.py`

- `test_setup_monitoring_creates_alarms` — verify CloudWatch alarm creation
- `test_setup_monitoring_creates_dashboard` — verify dashboard JSON
- `test_get_monitoring_report` — mock CloudWatch, verify report structure

---

## Prompt 49 — Output Drift Detection

Enhance EXISTING `src/monitoring/drift.py` by ADDING new methods to `DriftDetector`. Do NOT remove existing methods.

### New methods to ADD:

```python
    def check_output_drift(
        self,
        recent_outputs: list[dict[str, Any]],
        baseline_stats: dict[str, float],
    ) -> dict[str, Any]:
        """Detect drift in model output quality.

        Compute output statistics:
        - Average response length (tokens)
        - Vocabulary richness
        - Repetition rate
        - Refusal rate (responses matching refusal patterns)

        Compare against baseline using KS test.
        Return: {drifted: bool, metrics: dict, details: list[str]}
        """

    def check_input_drift(
        self,
        recent_inputs: list[dict[str, Any]],
        training_data_stats: dict[str, float],
    ) -> dict[str, Any]:
        """Detect drift in input distribution.

        Compare: topic distribution, length distribution, language distribution.
        Return: {drifted: bool, metrics: dict, details: list[str]}
        """
```

### Tests: add to `tests/unit/test_monitoring/test_drift.py`

- `test_check_output_drift_no_drift` — similar distributions, verify drifted=False
- `test_check_output_drift_detected` — very different distributions, verify drifted=True
- `test_check_input_drift` — verify input comparison works

---

## Prompt 50 — Alerting & Incident Response

Enhance EXISTING `src/monitoring/alerting.py` by ADDING methods. Do NOT remove existing methods.

### New methods to ADD to `AlertManager`:

```python
    def send_deployment_event(
        self,
        event_type: str,  # "started", "canary", "complete", "rollback"
        endpoint_name: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send deployment lifecycle notification."""

    def send_drift_alert(
        self,
        endpoint_name: str,
        drift_report: dict[str, Any],
    ) -> None:
        """Send drift detection alert with recommended actions."""

    def send_cost_alert(
        self,
        current_cost: float,
        budget_limit: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Send cost threshold alert."""
```

Create `configs/monitoring/alerts.yaml`:

```yaml
# Alert configuration
channels:
  dev:
    sns_topic_arn: "${ssm:/llm-finetune/alerts-dev-topic-arn}"
  staging:
    sns_topic_arn: "${ssm:/llm-finetune/alerts-staging-topic-arn}"
  prod:
    sns_topic_arn: "${ssm:/llm-finetune/alerts-prod-topic-arn}"

thresholds:
  latency_p99_ms: 5000
  error_rate_pct: 1.0
  cost_warning_pct: 80
  cost_critical_pct: 100
  drift_sensitivity: 0.05

dedup_window_minutes: 5
```

---

## Prompt 51 — Operational Runbook

Create `docs/runbook.md`:

A comprehensive operational runbook with these sections. Write REAL content, not placeholders:

### Common Operations
- How to launch a new training job (exact CLI commands with example configs)
- How to monitor a running training job (CloudWatch dashboard URLs, log commands)
- How to evaluate a trained model (CLI commands + expected output)
- How to deploy to dev/staging/prod (step-by-step with safety checks)
- How to roll back a production deployment (exact commands)
- How to scale up/down endpoints (autoscaling commands)
- How to re-run data pipeline (commands + validation)

### Incident Response Playbooks
For each incident type, provide: symptoms, root cause checklist, investigation steps, remediation commands, prevention measures.
- Endpoint returning 5xx errors
- Training job failure (OOM, data issue, spot interruption)
- Model producing degraded outputs (drift detected)
- Cost overrun (budget alarm triggered)
- Security incident (data exposure)

### Architecture Decision Records
- ADR-001: QLoRA vs full fine-tuning (decision, rationale, consequences)
- ADR-002: SageMaker vs self-managed training
- ADR-003: Bedrock vs SageMaker endpoints
- ADR-004: DVC vs S3 versioning for datasets
- ADR-005: MLflow vs SageMaker Experiments

### Contact & Escalation
- Template for team contacts and on-call rotation
- Escalation matrix by severity

---

## Prompt 52 — Disaster Recovery

Create `src/ops/disaster_recovery.py` and `docs/disaster_recovery.md`:

### `src/ops/__init__.py` — create empty

### `src/ops/disaster_recovery.py`

```python
"""Disaster recovery: backup, export, import, and failover operations."""
```

### Class: `DisasterRecoveryManager`

```python
class DisasterRecoveryManager:
    """Manage backups, exports, and cross-region failover."""

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize with boto3 clients."""

    def export_endpoint_config(self, endpoint_name: str) -> dict[str, Any]:
        """Export full endpoint configuration as JSON.

        Include: model, instance config, autoscaling, data capture settings.
        Save to S3 as backup.
        """

    def import_endpoint_config(
        self,
        config: dict[str, Any],
        target_region: str,
    ) -> str:
        """Recreate endpoint in target region from exported config.

        Copy model artifacts to target region if needed.
        Return new endpoint name.
        """

    def validate_backups(self) -> dict[str, Any]:
        """Verify all backup targets are current and restorable.

        Check: model artifacts in S3, training configs in git, MLflow data,
        endpoint configs exported.
        Return: {all_valid: bool, checks: list[{name, status, last_backup}]}
        """

    def full_region_failover(
        self,
        source_region: str,
        target_region: str,
        endpoint_name: str,
    ) -> dict[str, Any]:
        """Execute full region failover.

        1. Verify model artifacts in target region
        2. Deploy endpoint in target region
        3. Run smoke tests
        4. Return: {status, new_endpoint, smoke_test_passed, duration_seconds}
        """
```

### `docs/disaster_recovery.md`

Write comprehensive DR documentation:
- RPO/RTO targets (RPO: 1 hour for models, 24h for data. RTO: 4 hours)
- What is backed up and where (S3 cross-region, git, MLflow)
- Failover procedure (step-by-step)
- DR drill instructions (quarterly schedule)
- Recovery verification checklist

### Tests: `tests/unit/test_ops/test_disaster_recovery.py`

- `test_export_endpoint_config` — mock SageMaker, verify JSON structure
- `test_validate_backups` — mock S3, verify check results
- `test_full_region_failover` — mock all services, verify sequence

---

## Summary of Phase 12-14 files

### New files to CREATE:
1. `.github/workflows/ci.yml` (or enhance existing)
2. `.github/workflows/docker.yml`
3. `.github/workflows/train.yml`
4. `.github/workflows/deploy-prod.yml`
5. `src/monitoring/endpoint_monitor.py`
6. `src/ops/__init__.py`
7. `src/ops/disaster_recovery.py`
8. `configs/monitoring/alerts.yaml`
9. `docs/runbook.md`
10. `docs/disaster_recovery.md`
11. `tests/unit/test_monitoring/test_endpoint_monitor.py`
12. `tests/unit/test_ops/__init__.py`
13. `tests/unit/test_ops/test_disaster_recovery.py`

### Existing files to ENHANCE:
1. `src/monitoring/drift.py` — add output/input drift methods
2. `src/monitoring/alerting.py` — add deployment/drift/cost alert methods
