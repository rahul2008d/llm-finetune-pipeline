# LLM Fine-Tuning Pipeline

Production-grade QLoRA/DoRA fine-tuning pipeline for large language models on AWS SageMaker and Bedrock.

Built with Python, HuggingFace TRL, PEFT, and AWS CDK.

---

## What this does

Takes a base LLM (Llama 3, Mistral, Phi-3), fine-tunes it on your data using parameter-efficient methods (QLoRA/DoRA), and deploys it to SageMaker endpoints or Bedrock — with full observability, evaluation, and rollback capability.

```
data → validate → tokenize → train (QLoRA) → evaluate → register → deploy → monitor
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐     ┌──────────────┐
│  Data Prep  │────▶│   Training   │────▶│  Evaluation   │────▶│  Deployment  │
│             │     │              │     │               │     │              │
│ HF datasets │     │ SageMaker    │     │ lm-eval       │     │ SageMaker    │
│ PII scan    │     │ ml.g5.2xlarge│     │ custom metrics│     │ Bedrock      │
│ DVC version │     │ QLoRA 4-bit  │     │ A/B compare   │     │ blue-green   │
└─────────────┘     └──────────────┘     └───────────────┘     └──────────────┘
        │                   │                    │                      │
        └───────────────────┴────────────────────┴──────────────────────┘
                                     │
                              ┌──────┴───────┐
                              │  Monitoring  │
                              │  MLflow      │
                              │  CloudWatch  │
                              │  Drift det.  │
                              └──────────────┘
```

## Quick start

### Prerequisites

- Python 3.10–3.11
- [uv](https://github.com/astral-sh/uv) package manager
- AWS CLI configured with credentials
- HuggingFace account (token required for gated models like Llama 3)

### Install

```bash
git clone https://github.com/rahul2008d/llm-finetune-pipeline.git
cd llm-finetune-pipeline
uv sync --extra dev --extra data
```

For GPU training dependencies (Linux only):

```bash
uv sync --extra gpu
```

### Local smoke test (CPU, free)

```bash
# Prepare 50 samples from Alpaca
uv run llm-ft data prepare --config configs/data/preparation.yaml --yes

# Train GPT-2 + LoRA on CPU (~30 seconds)
uv run llm-ft train local --config configs/training/smoke_test_production.yaml --yes

# Evaluate
uv run llm-ft evaluate \
  --model-path ./outputs/smoke_test_production/ \
  --benchmark mmlu \
  --output-dir ./results/ \
  --yes
```

### SageMaker training (GPU, ~$3)

```bash
# 1. Deploy infrastructure
cd infra/cdk && cdk deploy --all --context env=dev --require-approval broadening

# 2. Upload data to S3
aws s3 sync ./data/prepared/smoke_test/ \
  s3://llm-finetune-dev-training-data/datasets/smoke_test/ \
  --sse aws:kms --sse-kms-key-id <your-kms-key-arn>

# 3. Launch training
uv run llm-ft train sagemaker --config configs/training/sagemaker_test.yaml --yes

# 4. Tear down when done (stops ~$4/day idle cost)
cd infra/cdk && cdk destroy --all --context env=dev --force
```

## Project structure

```
llm-finetune-pipeline/
├── src/
│   ├── cli.py                    # Typer CLI entry point
│   ├── config/
│   │   ├── training.py           # Pydantic v2 config models (TrainingJobConfig)
│   │   └── settings.py           # Environment settings
│   ├── data/
│   │   ├── loader.py             # HuggingFace, JSONL, S3 dataset loading
│   │   ├── pipeline.py           # Full data pipeline orchestrator
│   │   ├── pii_scanner.py        # Presidio-based PII detection
│   │   └── validation.py         # Schema and quality checks
│   ├── training/
│   │   ├── trainer.py            # FineTuneTrainer (production training loop)
│   │   ├── model_loader.py       # Quantized model loading + LoRA application
│   │   ├── sagemaker_launcher.py # SageMaker job submission + polling
│   │   ├── callbacks.py          # Training callbacks (memory, cost, gradient)
│   │   └── train_entry.py        # SageMaker container entry point
│   ├── evaluation/
│   │   ├── evaluator.py          # Model evaluation framework
│   │   ├── metrics.py            # Perplexity, ROUGE, BLEU, F1, diversity, toxicity
│   │   ├── comparator.py         # A/B model comparison with bootstrap CI
│   │   └── report.py             # Markdown/JSON evaluation reports
│   ├── serving/
│   │   ├── endpoint.py           # SageMaker endpoint manager (blue-green deploy)
│   │   ├── bedrock.py            # Bedrock custom model import
│   │   ├── inference.py          # SageMaker inference handler
│   │   ├── model_registry.py     # SageMaker Model Registry integration
│   │   └── autoscaling.py        # Endpoint auto-scaling policies
│   ├── monitoring/
│   │   ├── mlflow_tracker.py     # MLflow experiment tracking
│   │   ├── cloudwatch.py         # CloudWatch metrics publisher
│   │   ├── drift.py              # Input/output drift detection
│   │   ├── alerting.py           # SNS alerting (deployment, drift, cost)
│   │   └── model_card.py         # HuggingFace-format model card generator
│   ├── ops/
│   │   └── disaster_recovery.py  # Cross-region failover
│   └── utils/
│       ├── logging.py            # structlog JSON logging
│       ├── s3.py                 # S3 client wrapper
│       └── retry.py              # Exponential backoff with jitter
├── infra/cdk/                    # AWS CDK Python infrastructure
│   ├── app.py                    # CDK app entry point
│   ├── stacks/                   # NetworkStack, StorageStack, IamStack, SageMakerStack
│   ├── constructs/               # TrainingVpc, SecureBucket, PipelineMonitoring
│   └── config/                   # Environment configs (dev/staging/prod)
├── configs/
│   ├── training/                 # Training YAML configs (QLoRA, DoRA, HPO)
│   ├── deployment/               # Endpoint configs (dev/staging/prod/bedrock)
│   ├── evaluation/               # Evaluation configs
│   └── templates/                # Prompt templates (Alpaca, ChatML, Llama3)
├── docker/
│   ├── Dockerfile.train          # Training container
│   └── Dockerfile.serve          # Serving container
├── .github/workflows/
│   ├── ci.yml                    # Lint, type check, unit tests, integration tests
│   ├── docker.yml                # Docker build and push to ECR
│   ├── train.yml                 # SageMaker training pipeline
│   └── deploy-prod.yml           # Production deployment with blue-green
├── docs/
│   ├── runbook.md                # Operational runbook
│   └── disaster_recovery.md      # DR procedures
└── tests/
    ├── unit/                     # 1017 unit tests
    ├── integration/              # LocalStack integration tests
    └── e2e/                      # End-to-end pipeline tests
```

## CLI reference

All commands follow the pattern `llm-ft <group> <command> --config <yaml>`.

| Command | Description |
|---|---|
| `llm-ft data validate` | Run quality checks on a dataset |
| `llm-ft data prepare` | Download, clean, tokenize, and upload dataset |
| `llm-ft train local` | Fine-tune locally (CPU/GPU) |
| `llm-ft train sagemaker` | Submit SageMaker training job |
| `llm-ft train hpo` | Launch hyperparameter optimization |
| `llm-ft evaluate` | Run benchmarks and custom evaluations |
| `llm-ft merge` | Merge LoRA adapter into base model |
| `llm-ft deploy sagemaker` | Deploy to SageMaker endpoint |
| `llm-ft deploy bedrock` | Import model into Bedrock |
| `llm-ft monitor status` | Check endpoint health and metrics |

Every command accepts `--yes` to skip confirmation and `--config` for YAML configuration.

## Configuration

All training configuration is managed through `TrainingJobConfig` — a Pydantic v2 model with full validation. See `src/config/training.py` for the schema.

Example QLoRA config for Llama 3.1 8B:

```yaml
experiment_name: "llama3-8b-qlora"

model:
  model_name_or_path: "meta-llama/Meta-Llama-3.1-8B-Instruct"
  torch_dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096

quantization:
  method: "qlora"
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_compute_dtype: "bfloat16"
  bnb_4bit_use_double_quant: true

lora:
  r: 64
  lora_alpha: 128
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  use_dora: false

training:
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 0.0002
  optim: "paged_adamw_8bit"
  bf16: true
  gradient_checkpointing: true

sagemaker:
  instance_type: "ml.g5.2xlarge"
  instance_count: 1
  role_arn: "arn:aws:iam::123456789:role/sagemaker-training"

dataset_path: "s3://bucket/datasets/my-data/"
output_s3_uri: "s3://bucket/training-outputs/"
```

## Infrastructure

Four CDK stacks deployed per environment:

| Stack | Resources | Idle cost |
|---|---|---|
| `dev-network` | VPC, 3 subnet tiers, NAT gateway, 11 VPC endpoints, security groups, flow logs | ~$3.50/day |
| `dev-storage` | 3 KMS-encrypted S3 buckets, 2 ECR repos (immutable tags, scan on push) | ~$0.03/day |
| `dev-iam` | 4 least-privilege IAM roles (training, endpoint, bedrock, pipeline) with permission boundaries | $0 |
| `dev-sagemaker` | SageMaker domain, budget alarms, CloudTrail, CloudWatch dashboard, EventBridge rules | ~$0.50/day |

Deploy with `cdk deploy --all --context env=dev`. Destroy with `cdk destroy --all --context env=dev --force`.

S3 buckets enforce KMS encryption on all uploads. Use `--sse aws:kms --sse-kms-key-id <key-arn>` with `aws s3` commands.

## Testing

```bash
# Unit tests (1017 tests)
uv run pytest tests/unit/ -q

# Full suite with coverage
uv run pytest tests/ --override-ini="addopts="

# Phase-specific tests
uv run pytest tests/unit/test_training/ -q
uv run pytest tests/unit/test_evaluation/ -q
uv run pytest tests/unit/test_serving/ -q
```

## Security

- All S3 buckets enforce KMS encryption (deny unencrypted uploads, deny TLS < 1.2)
- ECR repositories have immutable tags and scan-on-push enabled
- IAM roles follow least privilege with permission boundaries
- VPC endpoints keep SageMaker traffic off the public internet
- PII scanning via Presidio before data enters the training pipeline
- Secrets (HF token) stored in AWS Secrets Manager, referenced via `resolve:secretsmanager`
- No credentials in code — IAM roles for SageMaker, CLI uses AWS credential chain

## Cost estimates

| Operation | Instance | Duration | Cost |
|---|---|---|---|
| Smoke test (50 samples, GPT-2, CPU) | local | 30s | $0 |
| QLoRA test (50 samples, Llama 3 8B) | ml.g5.2xlarge | ~30 min | ~$0.76 |
| Full training (10K samples, 3 epochs) | ml.g5.2xlarge | ~2 hrs | ~$3.04 |
| Idle infrastructure (dev) | — | per day | ~$4.00 |

Destroy CDK stacks when not training to avoid idle costs.

## Tech stack

| Component | Technology |
|---|---|
| Language | Python 3.10–3.11 |
| Package manager | uv |
| Training | HuggingFace TRL + PEFT + bitsandbytes |
| Quantization | QLoRA (4-bit NF4) / DoRA |
| Orchestration | AWS SageMaker Training Jobs |
| Serving | SageMaker Endpoints / AWS Bedrock |
| Infrastructure | AWS CDK (Python) |
| Experiment tracking | MLflow |
| Monitoring | CloudWatch + SNS |
| CI/CD | GitHub Actions |
| Data versioning | DVC |
| Config validation | Pydantic v2 |
| Logging | structlog (JSON) |
| Testing | pytest (1017 tests) |

## License

MIT
