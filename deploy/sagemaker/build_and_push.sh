#!/usr/bin/env bash
# Build the SageMaker image and push it to ECR in this account (README.md, 1. Image).
#
#   deploy/sagemaker/build_and_push.sh
#   REGION=us-east-1 DEVICE=cpu TAG=cpu deploy/sagemaker/build_and_push.sh
#
# REGION (default: the CLI's region, else us-west-2), REPO (default: strands-decider-sagemaker),
# TAG (default: latest), DEVICE (cuda or cpu, default cuda). The checkpoint is set in the
# Dockerfile. Prints the image URI on the last line.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REGION="${REGION:-$(aws configure get region || echo us-west-2)}"
REPO="${REPO:-strands-decider-sagemaker}"
REGISTRY="$(aws sts get-caller-identity --query Account --output text).dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG:-latest}"

aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
  || aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null
aws ecr get-login-password --region "$REGION" | docker login -u AWS --password-stdin "$REGISTRY"

# SageMaker hosts run x86_64; on an Apple-silicon Mac this build runs under emulation.
docker buildx build --platform linux/amd64 -f "$ROOT/deploy/sagemaker/Dockerfile" \
  --build-arg DEVICE="${DEVICE:-cuda}" -t "$IMAGE" --push "$ROOT"
echo "$IMAGE"
