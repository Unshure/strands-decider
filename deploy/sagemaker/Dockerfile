# SageMaker inference image for strands-decider, with the weights inside (README.md).
# Build from the repository root:
#   docker buildx build --platform linux/amd64 -f deploy/sagemaker/Dockerfile -t strands-decider-sagemaker .
# DEVICE=cuda (the default) is for GPU instances. DEVICE=cpu leaves out flash-linear-attention,
# whose Triton kernels need a GPU, for CPU instances; the CUDA build of torch also runs on CPU.
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
ARG DEVICE=cuda
RUN SETUPTOOLS_SCM_PRETEND_VERSION=${PACKAGE_VERSION} pip install ".[${DEVICE}]"

ARG CHECKPOINT=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
ARG CHECKPOINT_REVISION=main
COPY deploy/sagemaker/prefetch.py ./
# The folder keeps the repo id, so responses name the model by its basename (server.create_app).
RUN python prefetch.py "${CHECKPOINT}" "${CHECKPOINT_REVISION}" "/opt/models/${CHECKPOINT}"

# DECIDER_ARGS adds `strands-decider serve` options, such as "--max-batch 8 --strict-window";
# a SageMaker model's Environment can set it.
ENV DECIDER_CHECKPOINT=/opt/models/${CHECKPOINT} \
    DECIDER_DEVICE=${DEVICE} \
    DECIDER_ARGS="" \
    HF_HUB_OFFLINE=1

EXPOSE 8080
# SageMaker runs `docker run IMAGE serve`; that trailing `serve` becomes $0 and is ignored.
ENTRYPOINT ["sh", "-c", "exec strands-decider serve \"$DECIDER_CHECKPOINT\" --host 0.0.0.0 --port 8080 --device \"$DECIDER_DEVICE\" $DECIDER_ARGS"]
