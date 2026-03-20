#!/usr/bin/env bash
# Submit a SageMaker training job
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Configuration
CONFIG_FILE="${1:-$PROJECT_ROOT/configs/training/default.yaml}"
JOB_NAME="llm-finetune-$(date +%Y%m%d-%H%M%S)"
INSTANCE_TYPE="${SAGEMAKER_INSTANCE_TYPE:-ml.g5.2xlarge}"
INSTANCE_COUNT="${SAGEMAKER_INSTANCE_COUNT:-1}"

echo "=== SageMaker Training Job Submission ==="
echo "Job Name:      $JOB_NAME"
echo "Config:        $CONFIG_FILE"
echo "Instance Type: $INSTANCE_TYPE"
echo "Instance Count: $INSTANCE_COUNT"
echo ""

# Package source code
echo "Packaging source code..."
PACKAGE_DIR=$(mktemp -d)
cp -r "$PROJECT_ROOT/src" "$PACKAGE_DIR/"
cp -r "$PROJECT_ROOT/configs" "$PACKAGE_DIR/"
cp "$PROJECT_ROOT/pyproject.toml" "$PACKAGE_DIR/"

tar -czf "/tmp/${JOB_NAME}-source.tar.gz" -C "$PACKAGE_DIR" .
rm -rf "$PACKAGE_DIR"

# Upload to S3
echo "Uploading source to S3..."
aws s3 cp "/tmp/${JOB_NAME}-source.tar.gz" \
    "s3://${S3_BUCKET_ARTIFACTS}/training-jobs/${JOB_NAME}/source.tar.gz"

# Submit training job
echo "Submitting training job..."
aws sagemaker create-training-job \
    --training-job-name "$JOB_NAME" \
    --role-arn "$SAGEMAKER_ROLE_ARN" \
    --algorithm-specification \
        "TrainingImage=763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com/huggingface-pytorch-training:2.1.0-transformers4.36.0-gpu-py310-cu121-ubuntu20.04,TrainingInputMode=File" \
    --resource-config \
        "InstanceType=${INSTANCE_TYPE},InstanceCount=${INSTANCE_COUNT},VolumeSizeInGB=100" \
    --input-data-config \
        "[{\"ChannelName\":\"training\",\"DataSource\":{\"S3DataSource\":{\"S3DataType\":\"S3Prefix\",\"S3Uri\":\"s3://${S3_BUCKET_DATA}/datasets/\",\"S3DataDistributionType\":\"FullyReplicated\"}}}]" \
    --output-data-config \
        "S3OutputPath=s3://${S3_BUCKET_MODELS}/training-output/" \
    --stopping-condition "MaxRuntimeInSeconds=86400" \
    --region "$AWS_REGION"

echo ""
echo "=== Job Submitted ==="
echo "Monitor: aws sagemaker describe-training-job --training-job-name $JOB_NAME"
