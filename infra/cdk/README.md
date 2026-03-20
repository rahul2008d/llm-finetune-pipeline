# LLM Fine-Tuning Infrastructure (CDK Python)

AWS CDK Python project for the LLM fine-tuning pipeline infrastructure.

## Structure

```
infra/cdk/
├── app.py                    # CDK app entry point
├── cdk.json                  # CDK app config
├── pyproject.toml            # CDK project dependencies
├── Makefile                  # CDK-specific targets
├── stacks/                   # Stack definitions
│   ├── network_stack.py      # VPC, subnets, endpoints, flow logs
│   ├── storage_stack.py      # S3, KMS, ECR
│   ├── iam_stack.py          # 4 least-privilege IAM roles
│   └── sagemaker_stack.py    # Domain, budgets, alarms, trail, events
├── constructs/               # Reusable constructs
│   ├── training_vpc.py       # VPC with all endpoints
│   ├── secure_bucket.py      # Encrypted S3 bucket pattern
│   ├── least_privilege_role.py  # Configurable IAM role builder
│   └── pipeline_monitoring.py   # Alarms, dashboard, SNS
├── config/                   # Environment configuration
│   ├── constants.py          # Project name, CIDR, endpoint list
│   └── environments.py       # Typed dataclass configs per env
└── tests/                    # CDK assertion tests
    ├── conftest.py            # Shared fixtures
    ├── test_network_stack.py
    ├── test_storage_stack.py
    ├── test_iam_stack.py
    ├── test_sagemaker_stack.py
    └── test_security_invariants.py
```

## Setup

```bash
cd infra/cdk
pip install -e '.[dev]'
```

## Commands

```bash
make synth ENV=dev       # Synthesize CloudFormation
make diff ENV=dev        # Show diff against deployed stack
make deploy ENV=dev      # Deploy all stacks
make test                # Run CDK tests
make quality             # Lint + typecheck + test
```
