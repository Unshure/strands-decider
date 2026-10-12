#!/usr/bin/env bash
# Build the SageMaker image and push it to ECR in this account (README.md, 1. Image).
#
#   deploy/sagemaker/build_and_push.sh
#   REGION=us-east-1 TAG=v1-2610 deploy/sagemaker/build_and_push.sh
#
# REGION (default: the CLI's region, else us-west-2), REPO (default: strands-decider-sagemaker),
# TAG (default: latest), DEVICE (cuda or cpu, default cuda), CHECKPOINT and CHECKPOINT_REVISION
# (passed to the Dockerfile).
# Prints the image URI on the last line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGION="${REGION:-$(aws configure get region || true)}"
REGION="${REGION:-us-west-2}"
REPO="${REPO:-strands-decider-sagemaker}"
TAG="${TAG:-latest}"
CHECKPOINT="${CHECKPOINT:-StrandsAgents/strands-decider-2B-qwen3.5-v1-2610}"
CHECKPOINT_REVISION="${CHECKPOINT_REVISION:-main}"
DEVICE="${DEVICE:-cuda}"

ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" \
       --image-scanning-configuration scanOnPush=true >/dev/null
aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$REGISTRY"

# SageMaker hosts run x86_64; on an Apple-silicon Mac this build runs under emulation.
docker buildx build --platform linux/amd64 \
  -f "$ROOT/deploy/sagemaker/Dockerfile" \
  --build-arg CHECKPOINT="$CHECKPOINT" \
  --build-arg CHECKPOINT_REVISION="$CHECKPOINT_REVISION" \
  --build-arg DEVICE="$DEVICE" \
  -t "$IMAGE" --push "$ROOT"

echo "$IMAGE"
