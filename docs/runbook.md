# LLM Fine-Tuning Pipeline — Operational Runbook

## Table of Contents

1. [Common Operations](#common-operations)
2. [Incident Response Playbooks](#incident-response-playbooks)
3. [Architecture Decision Records](#architecture-decision-records)
4. [Contact & Escalation](#contact--escalation)

---

## Common Operations

### Launch a New Training Job

**Prerequisites:** AWS credentials configured, training config validated.

```bash
# Validate config before submitting
llm-ft validate --config configs/training/qlora_llama3_8b.yaml

# Estimate cost
llm-ft cost-estimate --config configs/training/qlora_llama3_8b.yaml

# Submit SageMaker training job
llm-ft train sagemaker \
  --config configs/training/qlora_llama3_8b.yaml \
  --yes

# Or launch locally for debugging (single GPU)
llm-ft train local \
  --config configs/training/smoke_test_cpu.yaml
```

**Example configs:**
- `configs/training/qlora_llama3_8b.yaml` — QLoRA on Llama 3 8B (recommended starting point)
- `configs/training/qlora_llama3_70b.yaml` — QLoRA on Llama 3 70B (multi-GPU)
- `configs/training/qlora_mistral_7b.yaml` — QLoRA on Mistral 7B
- `configs/training/dora_llama3_8b.yaml` — DoRA on Llama 3 8B
- `configs/training/smoke_test_cpu.yaml` — CPU smoke test (CI/local validation)

### Monitor a Running Training Job

```bash
# Check SageMaker training job status
aws sagemaker describe-training-job --training-job-name <job-name>

# Stream training logs
aws logs tail /aws/sagemaker/TrainingJobs --follow \
  --filter-pattern "<job-name>"

# View CloudWatch dashboard (auto-created during training)
# URL format: https://<region>.console.aws.amazon.com/cloudwatch/home
#   ?region=<region>#dashboards:name=LLMFineTuning-<experiment-name>

# Check MLflow experiment
mlflow ui --port 5000
# Then browse to http://localhost:5000
```

**Key metrics to watch:**
- `TrainLoss` — should decrease steadily
- `EvalLoss` — should decrease; watch for divergence from train loss (overfitting)
- `LearningRate` — should follow the configured schedule
- `GPUMemoryUtilization` — should stay below 95%
- `GradientNorm` — spikes indicate instability

### Evaluate a Trained Model

```bash
# Run standard benchmarks
llm-ft evaluate \
  --model-path s3://your-bucket/models/qlora-llama3-8b-v1/ \
  --benchmark mmlu,hellaswag,arc_challenge \
  --output-dir ./results/

# View results
cat results/eval_results.json
cat results/eval_results.md
```

**Expected output structure:**
```json
{
  "model_path": "s3://...",
  "benchmarks": {
    "mmlu": {"accuracy": 0.65, "details": {...}},
    "hellaswag": {"accuracy": 0.78, "details": {...}}
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Deploy to Dev/Staging/Prod

**Dev deployment:**
```bash
llm-ft deploy sagemaker \
  --model s3://your-bucket/models/qlora-llama3-8b-v1/ \
  --endpoint-name llm-ft-dev-v1 \
  --instance ml.g5.xlarge \
  --config configs/deployment/sagemaker_dev.yaml \
  --yes

# Smoke test
python -c "
from src.serving.endpoint_tester import EndpointTester
tester = EndpointTester()
result = tester.smoke_test('llm-ft-dev-v1')
print(result)
"
```

**Staging deployment:**
```bash
llm-ft deploy sagemaker \
  --model s3://your-bucket/models/qlora-llama3-8b-v1/ \
  --endpoint-name llm-ft-staging-v1 \
  --instance ml.g5.xlarge \
  --config configs/deployment/sagemaker_staging.yaml \
  --yes
```

**Production deployment (blue-green):**
```bash
# REQUIRES manual approval in GitHub Actions
# Use the deploy-prod.yml workflow via GitHub UI
# Or manually:
python -c "
from src.serving.endpoint import SageMakerEndpointManager
mgr = SageMakerEndpointManager()
result = mgr.blue_green_deploy(
    endpoint_name='llm-ft-prod-v1',
    new_model_data_url='s3://your-bucket/models/qlora-llama3-8b-v1/model.tar.gz',
    canary_pct=0.1,
    bake_time_minutes=30,
    rollback_alarm_names=['llm-finetune-prod-endpoint-5xx-rate'],
)
print(result)
"
```

**Safety checklist before prod deploy:**
1. Model passes all benchmark evaluations
2. Model registered in SageMaker Model Registry with approval
3. Staging smoke tests pass
4. No active incidents on the current prod endpoint
5. Oncall engineer is aware and available

### Roll Back a Production Deployment

```bash
# Automatic rollback happens if CloudWatch alarms fire during bake period

# Manual rollback — restore previous endpoint config:
python -c "
from src.serving.endpoint import SageMakerEndpointManager
mgr = SageMakerEndpointManager()

# List endpoints to find the previous model
endpoints = mgr.list_endpoints(name_contains='llm-ft-prod')
print(endpoints)

# Update traffic to 100% on the old variant
mgr.update_endpoint_traffic(
    endpoint_name='llm-ft-prod-v1',
    variant_weights={'old-variant': 1.0},
)
"

# Verify rollback
python -c "
from src.serving.endpoint_tester import EndpointTester
tester = EndpointTester()
result = tester.smoke_test('llm-ft-prod-v1')
assert result['passed'], f'Rollback verification failed: {result}'
print('Rollback verified successfully')
"
```

### Scale Up/Down Endpoints

```bash
# Configure autoscaling
python -c "
from src.serving.autoscaling import EndpointAutoScaler
scaler = EndpointAutoScaler()

# Set target tracking (scales based on invocations per instance)
scaler.configure_target_tracking(
    endpoint_name='llm-ft-prod-v1',
    variant_name='AllTraffic',
    min_capacity=1,
    max_capacity=4,
    target_value=100.0,  # target invocations per instance
)

# Or set scheduled scaling
scaler.configure_scheduled_scaling(
    endpoint_name='llm-ft-prod-v1',
    variant_name='AllTraffic',
    schedule='cron(0 9 * * ? *)',  # scale up at 9am UTC
    min_capacity=2,
    max_capacity=8,
)
"

# Manual scale (immediate)
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name llm-ft-prod-v1 \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":3}]'
```

### Re-Run Data Pipeline

```bash
# Full pipeline run
llm-ft data prepare \
  --config configs/data/preparation.yaml \
  --output-dir data/prepared/

# Validate output
python -c "
from src.data.pipeline import DataPipeline
pipeline = DataPipeline()
stats = pipeline.validate('data/prepared/')
print(stats)
"

# Check for PII
python -c "
from src.data.pii_scanner import PIIScanner
scanner = PIIScanner()
results = scanner.scan('data/prepared/')
print(f'PII findings: {results}')
"
```

---

## Incident Response Playbooks

### Endpoint Returning 5xx Errors

**Symptoms:**
- CloudWatch alarm `<endpoint-name>-5xx-rate` fires
- Users report API failures
- Error rate visible on CloudWatch dashboard

**Root cause checklist:**
- [ ] Model container crashed (OOM, bad model artifact)
- [ ] Instance health issues
- [ ] Model input format changed (breaking contract)
- [ ] AWS service issue (SageMaker regional outage)

**Investigation steps:**
```bash
# 1. Check endpoint status
aws sagemaker describe-endpoint --endpoint-name llm-ft-prod-v1 | jq '.EndpointStatus'

# 2. Check CloudWatch logs for errors
aws logs filter-log-events \
  --log-group-name /aws/sagemaker/Endpoints/llm-ft-prod-v1 \
  --start-time $(date -d '1 hour ago' +%s000) \
  --filter-pattern "ERROR"

# 3. Check instance metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name MemoryUtilization \
  --dimensions Name=EndpointName,Value=llm-ft-prod-v1 \
  --start-time $(date -d '1 hour ago' -u +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 60 --statistics Maximum

# 4. Test with known-good input
python -c "
from src.serving.endpoint_tester import EndpointTester
tester = EndpointTester()
result = tester.smoke_test('llm-ft-prod-v1')
print(result)
"
```

**Remediation:**
```bash
# If model container is crashing — rollback to previous model
python -c "
from src.serving.endpoint import SageMakerEndpointManager
mgr = SageMakerEndpointManager()
mgr.update_endpoint_traffic('llm-ft-prod-v1', {'old-variant': 1.0})
"

# If instance-level issue — scale up to add healthy instances
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name llm-ft-prod-v1 \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":3}]'
```

**Prevention:**
- Always run smoke tests after deployment
- Set up bake time with automatic rollback on alarm
- Monitor memory and GPU utilization trends

### Training Job Failure

**Symptoms:**
- SageMaker training job status is `Failed`
- CloudWatch training loss metrics stop updating
- SNS alert received for training failure

**Root cause checklist:**
- [ ] Out of Memory (OOM) — model too large for instance
- [ ] Data issue — corrupt data, wrong format, missing files
- [ ] Spot interruption — spot instance reclaimed
- [ ] Configuration error — invalid hyperparameters
- [ ] Disk full — insufficient volume size

**Investigation steps:**
```bash
# 1. Get failure reason
aws sagemaker describe-training-job --training-job-name <job-name> | jq '.FailureReason'

# 2. Check CloudWatch logs
aws logs filter-log-events \
  --log-group-name /aws/sagemaker/TrainingJobs \
  --log-stream-name-prefix <job-name> \
  --filter-pattern "Error|OOM|CUDA|killed"

# 3. Check last metrics
aws cloudwatch get-metric-statistics \
  --namespace LLMFineTuning/<experiment-name> \
  --metric-name TrainLoss \
  --start-time $(date -d '24 hours ago' -u +%Y-%m-%dT%H:%M:%S) \
  --end-time $(date -u +%Y-%m-%dT%H:%M:%S) \
  --period 300 --statistics Average
```

**Remediation by cause:**

*OOM:*
```bash
# Reduce batch size or enable gradient checkpointing
# Edit config: per_device_train_batch_size, gradient_accumulation_steps
# Or switch to a larger instance type
```

*Spot interruption:*
```bash
# Re-submit with checkpoint resume
llm-ft train sagemaker \
  --config configs/training/qlora_llama3_8b.yaml \
  --resume-from-checkpoint s3://bucket/checkpoints/last/ \
  --yes
```

*Data issue:*
```bash
# Validate data
llm-ft data validate --path s3://bucket/data/train.jsonl
# Re-run data pipeline if needed
llm-ft data prepare --config configs/data/preparation.yaml
```

**Prevention:**
- Use gradient checkpointing for large models
- Enable SageMaker managed spot with checkpointing
- Validate data before submitting training jobs
- Set appropriate volume sizes (at least 3x model size)

### Model Producing Degraded Outputs (Drift Detected)

**Symptoms:**
- Drift alert from monitoring system
- User complaints about response quality
- Increased refusal rate or repetitive outputs

**Root cause checklist:**
- [ ] Input distribution shift — users sending different types of queries
- [ ] Model degradation — checkpoint corruption
- [ ] Data contamination in recent fine-tuning
- [ ] Serving infrastructure issue (quantization errors)

**Investigation steps:**
```bash
# 1. Run drift analysis
python -c "
from src.monitoring.drift import DriftDetector
detector = DriftDetector()

# Check output drift
outputs = [...]  # collect recent outputs
baseline = {'avg_response_length': 50.0, 'vocab_richness': 0.85, 'refusal_rate': 0.02}
result = detector.check_output_drift(outputs, baseline)
print(result)
"

# 2. Check model metrics on CloudWatch dashboard
# Navigate to: https://<region>.console.aws.amazon.com/cloudwatch/home
#   ?region=<region>#dashboards:name=LLMFineTune-<endpoint-name>

# 3. Compare against benchmark results
llm-ft evaluate \
  --model-path s3://bucket/models/current/ \
  --benchmark mmlu \
  --output-dir ./results/drift-investigation/
```

**Remediation:**
```bash
# If model artifact issue — redeploy known-good model
python -c "
from src.serving.endpoint import SageMakerEndpointManager
mgr = SageMakerEndpointManager()
mgr.blue_green_deploy(
    endpoint_name='llm-ft-prod-v1',
    new_model_data_url='s3://bucket/models/last-known-good/model.tar.gz',
    canary_pct=0.1,
    bake_time_minutes=15,
)
"

# If input drift — update training data and retrain
llm-ft data prepare --config configs/data/preparation.yaml
llm-ft train sagemaker --config configs/training/qlora_llama3_8b.yaml --yes
```

**Prevention:**
- Schedule regular drift checks (daily)
- Maintain baseline statistics from evaluation
- Keep previous model versions for quick rollback

### Cost Overrun (Budget Alarm Triggered)

**Symptoms:**
- Cost alert from AlertManager
- AWS Budget alarm notification
- Unexpected charges on AWS billing dashboard

**Root cause checklist:**
- [ ] Forgotten running endpoints (dev/staging left on)
- [ ] Autoscaling scaled too aggressively
- [ ] Training job running longer than expected
- [ ] Data transfer costs (cross-region)

**Investigation steps:**
```bash
# 1. List all active endpoints
aws sagemaker list-endpoints --status-equals InService | jq '.Endpoints[].EndpointName'

# 2. Check instance counts
for ep in $(aws sagemaker list-endpoints --status-equals InService | jq -r '.Endpoints[].EndpointName'); do
  echo "$ep:"
  aws sagemaker describe-endpoint --endpoint-name "$ep" | \
    jq '.ProductionVariants[] | {VariantName, CurrentInstanceCount}'
done

# 3. Check running training jobs
aws sagemaker list-training-jobs --status-equals InProgress | jq '.TrainingJobSummaries[]'

# 4. Check AWS Cost Explorer (last 7 days)
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics BlendedCost \
  --filter '{"Dimensions":{"Key":"SERVICE","Values":["Amazon SageMaker"]}}'
```

**Remediation:**
```bash
# Delete unused endpoints
python -c "
from src.serving.endpoint import SageMakerEndpointManager
mgr = SageMakerEndpointManager()
mgr.delete_endpoint('llm-ft-dev-v1')  # or whichever is unused
"

# Scale down over-provisioned endpoints
aws sagemaker update-endpoint-weights-and-capacities \
  --endpoint-name llm-ft-prod-v1 \
  --desired-weights-and-capacities '[{"VariantName":"AllTraffic","DesiredInstanceCount":1}]'

# Stop running training jobs if not needed
aws sagemaker stop-training-job --training-job-name <job-name>
```

**Prevention:**
- Set AWS Budgets with alerts at 80% and 100% thresholds
- Use scheduled scaling to reduce capacity during off-hours
- Tag all resources for cost allocation
- Review costs weekly

### Security Incident (Data Exposure)

**Symptoms:**
- Security scan alert (exposed credentials, PII in logs)
- Unauthorized access attempt detected
- Data found in unexpected location

**Root cause checklist:**
- [ ] Credentials committed to repository
- [ ] S3 bucket misconfigured (public access)
- [ ] PII in training data leaked to logs
- [ ] IAM role over-permissioned

**Investigation steps:**
```bash
# 1. Scan for exposed secrets
detect-secrets scan src/ tests/ configs/

# 2. Check S3 bucket policies
aws s3api get-bucket-policy --bucket <bucket-name>
aws s3api get-public-access-block --bucket <bucket-name>

# 3. Review CloudTrail for unauthorized access
aws cloudtrail lookup-events \
  --lookup-attributes AttributeKey=EventName,AttributeValue=GetObject \
  --start-time $(date -d '24 hours ago' -u +%Y-%m-%dT%H:%M:%S)

# 4. Check IAM role permissions
aws iam get-role-policy --role-name <role-name> --policy-name <policy-name>
```

**Remediation:**
```bash
# If credentials exposed:
# 1. Rotate all affected credentials immediately
# 2. Revoke the exposed credentials
aws iam delete-access-key --user-name <user> --access-key-id <key-id>

# If S3 misconfigured:
aws s3api put-public-access-block --bucket <bucket> \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true

# If PII in training data:
# Run PII scanner and redact
python -c "
from src.data.pii_scanner import PIIScanner
scanner = PIIScanner()
scanner.scan_and_redact('data/prepared/')
"
```

**Prevention:**
- Use pre-commit hooks for secret scanning
- Enable S3 Block Public Access at account level
- Run PII scanner as part of data pipeline
- Follow least-privilege IAM policies
- Enable CloudTrail logging on all buckets

---

## Architecture Decision Records

### ADR-001: QLoRA vs Full Fine-Tuning

**Status:** Accepted

**Context:**
We need to fine-tune large language models (7B–70B parameters) on domain-specific data. Full fine-tuning requires updating all parameters, demanding significant GPU memory and compute. QLoRA (Quantized Low-Rank Adaptation) quantizes the base model to 4-bit and trains small adapter matrices.

**Decision:**
Use QLoRA as the default fine-tuning method, with full fine-tuning available as a configuration option for smaller models.

**Rationale:**
- QLoRA reduces memory requirements by ~75%, enabling 70B model training on 4x A100 GPUs (vs 16x for full fine-tune)
- Training cost drops proportionally — a typical QLoRA run costs $50–100 vs $500–1000 for full fine-tuning
- Multiple studies show QLoRA achieves within 1–2% of full fine-tuning on most benchmarks
- Adapter weights are small (~100MB vs ~140GB), simplifying storage and deployment
- Supports rapid experimentation: train and evaluate in hours, not days

**Consequences:**
- Slightly lower peak performance compared to full fine-tuning on some tasks
- Added dependency on bitsandbytes and PEFT libraries
- Need to manage adapter merging for certain deployment targets (e.g., Bedrock)
- DoRA variant available as an alternative that may close the performance gap

### ADR-002: SageMaker vs Self-Managed Training

**Status:** Accepted

**Context:**
Training infrastructure options include: (a) self-managed GPU instances on EC2 with custom orchestration, (b) AWS SageMaker managed training, (c) third-party platforms (Anyscale, Lambda Labs, etc.).

**Decision:**
Use AWS SageMaker as the primary training platform, with local training support for development and debugging.

**Rationale:**
- Managed infrastructure: SageMaker handles instance provisioning, health monitoring, and cleanup
- Spot instance support: 60–80% cost savings with automatic checkpoint resume
- Built-in experiment tracking integration (SageMaker Experiments + MLflow)
- Native integration with S3 for data and artifact management
- Consistent Docker-based execution ensures reproducibility
- No need to manage GPU driver updates, CUDA toolkit versions, etc.

**Consequences:**
- Vendor lock-in to AWS for training (mitigated by standard Docker containers)
- SageMaker overhead adds ~5 minutes startup time per job
- Need SageMaker-compatible Docker images
- Cost premium vs raw EC2 (~10–15%) offset by reduced operational burden

### ADR-003: Bedrock vs SageMaker Endpoints

**Status:** Accepted

**Context:**
Deployment options for fine-tuned models include SageMaker real-time endpoints and AWS Bedrock custom model import.

**Decision:**
Support both SageMaker endpoints and Bedrock, with the choice driven by use case requirements.

**Rationale:**
- **SageMaker endpoints:** Full control over instance type, scaling, and inference code. Best for latency-sensitive applications and custom preprocessing.
- **Bedrock:** Serverless, no infrastructure management, built-in guardrails and content safety. Best for applications needing managed safety features.
- Supporting both provides flexibility without significant additional complexity

**Consequences:**
- Need to maintain two deployment paths and test both
- Artifact packaging differs (SageMaker: model.tar.gz with inference code; Bedrock: merged model weights)
- Monitoring and scaling configurations differ between platforms
- Team needs familiarity with both services

### ADR-004: DVC vs S3 Versioning for Datasets

**Status:** Accepted

**Context:**
Training datasets need versioning, reproducibility, and efficient storage. Options: (a) DVC (Data Version Control) with remote storage, (b) S3 native versioning, (c) Delta Lake / Iceberg.

**Decision:**
Use S3 versioning with structured naming conventions, without introducing DVC as a dependency.

**Rationale:**
- S3 versioning is built into the platform — no additional tooling needed
- Structured paths (`s3://bucket/data/v1.0/`, `s3://bucket/data/v1.1/`) provide clear versioning
- Training configs reference exact S3 paths, ensuring reproducibility
- DVC adds complexity (Git LFS-like workflow, `.dvc` files) without proportional benefit for our dataset sizes
- Data pipeline produces deterministic outputs from config — config is versioned in Git

**Consequences:**
- No automatic de-duplication of data across versions (mitigated by S3 Intelligent-Tiering)
- Reproducibility relies on convention (S3 paths in config) rather than tooling enforcement
- Large dataset downloads always come from S3 (no local cache like DVC)

### ADR-005: MLflow vs SageMaker Experiments

**Status:** Accepted

**Context:**
Experiment tracking options include MLflow (open source), SageMaker Experiments (AWS-native), Weights & Biases, and Neptune.

**Decision:**
Use MLflow as the primary experiment tracker, with SageMaker Experiments as a secondary integration.

**Rationale:**
- MLflow is open source and platform-agnostic — no vendor lock-in
- Rich UI for comparing runs, visualizing metrics, and managing model registry
- Strong community and ecosystem support
- SageMaker SDK automatically logs to SageMaker Experiments alongside MLflow
- Local development uses the same tracking API as remote training
- Model Registry in MLflow provides lifecycle management (staging, production, archived)

**Consequences:**
- Need to run/host MLflow Tracking Server (or use managed offering)
- Duplicate tracking in both MLflow and SageMaker Experiments (acceptable overhead)
- Team needs to learn MLflow API and conventions
- Storage costs for MLflow artifacts in S3

---

## Contact & Escalation

### Team Contacts

| Role | Name | Contact | On-Call Schedule |
|------|------|---------|------------------|
| ML Engineering Lead | \<TBD\> | \<email/slack\> | Mon–Fri business hours |
| MLOps Engineer | \<TBD\> | \<email/slack\> | Rotating weekly |
| Platform Engineer | \<TBD\> | \<email/slack\> | Rotating weekly |
| Data Engineer | \<TBD\> | \<email/slack\> | As needed |

### On-Call Rotation

- Primary on-call: Rotates weekly among MLOps and Platform engineers
- Secondary on-call: ML Engineering Lead
- Rotation schedule managed in PagerDuty / Opsgenie

### Escalation Matrix

| Severity | Response Time | Who | Action |
|----------|--------------|-----|--------|
| **SEV-1 (Critical)** | 15 minutes | Primary on-call → ML Lead → Eng Director | Production endpoint down, data breach. Page immediately. |
| **SEV-2 (High)** | 1 hour | Primary on-call → ML Lead | Degraded performance, failed deployment, high error rate. |
| **SEV-3 (Medium)** | 4 hours | Primary on-call | Training failures, drift detected, non-critical alerts. |
| **SEV-4 (Low)** | Next business day | Ticket assignee | Documentation updates, config changes, feature requests. |

### Communication Channels

- **Slack:** `#ml-platform-alerts` (automated), `#ml-platform-ops` (team), `#ml-platform-incidents` (during incidents)
- **PagerDuty:** ML Platform service
- **Status page:** Internal status dashboard for ML platform health
