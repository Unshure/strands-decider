# SageMaker inference image for strands-decider, with the weights inside (README.md).
# Build from the repository root:
#   docker buildx build --platform linux/amd64 -f deploy/sagemaker/Dockerfile -t strands-decider-sagemaker .
FROM pytorch/pytorch:2.7.1-cuda12.6-cudnn9-runtime

# Triton compiles a small C launcher at run time (training/README.md, Setup).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf

WORKDIR /opt/strands-decider
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
# No .git in the build context, so setuptools-scm takes the version from here.
ARG PACKAGE_VERSION=0.0.0+sagemaker
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} pip install ".[cuda]"

ARG CHECKPOINT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
ARG CHECKPOINT_REVISION=main
COPY deploy/sagemaker/prefetch.py ./
# The folder keeps the repo id, so responses name the model by its basename (server.create_app).
RUN python prefetch.py "${CHECKPOINT}" "${CHECKPOINT_REVISION}" "/opt/models/${CHECKPOINT}"

ENV DECIDER_CHECKPOINT=/opt/models/${CHECKPOINT} \
    DECIDER_DEVICE=cuda \
    HF_HUB_OFFLINE=1

EXPOSE 8080
ENTRYPOINT ["python", "-m", "strands_decider.sagemaker_app"]
