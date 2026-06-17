# Cloud-agnostic training image. Runs anywhere with NVIDIA GPUs + the container
# runtime: Azure ML command jobs, a raw Azure NC/ND GPU VM, RunPod, Lambda, etc.
#
#   docker build -t chess-logic-gpt:latest .
#   docker run --gpus all --rm \
#       -e HF_TOKEN=hf_xxx -e PHASE=sft \
#       -v $PWD/data:/workspace/data \
#       chess-logic-gpt:latest
#
# Data is mounted/attached at runtime (not baked in) -- see .dockerignore.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRACKIO_PROJECT=chess-logic-gpt

RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install deps first (cached) using only the package metadata + source.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install -e ".[training]"

# Then the rest of the project (scripts, configs).
COPY . .

CMD ["bash", "scripts/cloud_train.sh"]
