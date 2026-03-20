#!/usr/bin/env bash
# Build and push training/serving Docker images to Amazon ECR.
# Usage: ./scripts/build_and_push_ecr.sh [--region us-east-1] [--repo-prefix llm-finetune]
set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────
AWS_REGION="${AWS_REGION:-us-east-1}"
REPO_PREFIX="${REPO_PREFIX:-llm-finetune-pipeline}"
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
GIT_SHA=$(git rev-parse --short HEAD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ── Parse flags ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region)  AWS_REGION="$2"; shift 2 ;;
        --repo-prefix) REPO_PREFIX="$2"; shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

TRAIN_REPO="${REPO_PREFIX}-train"
SERVE_REPO="${REPO_PREFIX}-serve"

echo "=== ECR Build & Push ==="
echo "Account:  ${ACCOUNT_ID}"
echo "Region:   ${AWS_REGION}"
echo "Registry: ${ECR_REGISTRY}"
echo "Git SHA:  ${GIT_SHA}"
echo ""

# ── Step 1: Authenticate to ECR ─────────────────────────────────────────────
echo "Authenticating to ECR..."
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Also authenticate to the DLC ECR registry (for base images)
aws ecr get-login-password --region "${AWS_REGION}" \
    | docker login --username AWS --password-stdin "763104351884.dkr.ecr.${AWS_REGION}.amazonaws.com"

# ── Step 2: Ensure ECR repositories exist ────────────────────────────────────
for repo in "${TRAIN_REPO}" "${SERVE_REPO}"; do
    if ! aws ecr describe-repositories --repository-names "${repo}" --region "${AWS_REGION}" >/dev/null 2>&1; then
        echo "Creating ECR repository: ${repo}"
        aws ecr create-repository \
            --repository-name "${repo}" \
            --region "${AWS_REGION}" \
            --image-scanning-configuration scanOnPush=true \
            --encryption-configuration encryptionType=AES256
    fi
done

# ── Step 3: Build images with BuildKit ───────────────────────────────────────
export DOCKER_BUILDKIT=1

echo ""
echo "Building training image..."
docker build \
    --platform linux/amd64 \
    --build-arg AWS_REGION="${AWS_REGION}" \
    -f "${PROJECT_ROOT}/docker/Dockerfile.train" \
    -t "${TRAIN_REPO}:${GIT_SHA}" \
    -t "${TRAIN_REPO}:latest" \
    "${PROJECT_ROOT}"

echo ""
echo "Building serving image..."
docker build \
    --platform linux/amd64 \
    --build-arg AWS_REGION="${AWS_REGION}" \
    -f "${PROJECT_ROOT}/docker/Dockerfile.serve" \
    -t "${SERVE_REPO}:${GIT_SHA}" \
    -t "${SERVE_REPO}:latest" \
    "${PROJECT_ROOT}"

# ── Step 4: Tag and push to ECR ─────────────────────────────────────────────
echo ""
echo "Tagging and pushing images..."

for repo in "${TRAIN_REPO}" "${SERVE_REPO}"; do
    for tag in "${GIT_SHA}" "latest"; do
        docker tag "${repo}:${tag}" "${ECR_REGISTRY}/${repo}:${tag}"
        echo "  Pushing ${ECR_REGISTRY}/${repo}:${tag}"
        docker push "${ECR_REGISTRY}/${repo}:${tag}"
    done
done

echo ""
echo "=== Done ==="
echo "Train: ${ECR_REGISTRY}/${TRAIN_REPO}:${GIT_SHA}"
echo "Serve: ${ECR_REGISTRY}/${SERVE_REPO}:${GIT_SHA}"
