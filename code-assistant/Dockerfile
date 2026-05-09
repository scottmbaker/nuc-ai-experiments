# OpenVINO Model Server image, GPU build, with a quantized Qwen3-Coder
# baked in. Two stages:
#   1. fetcher — downloads the pre-quantized OpenVINO IR from the
#                OpenVINO HuggingFace org
#   2. runtime — OVMS GPU base image + the model + LLM server config
#
# The fetcher stage is cached by Buildkit on MODEL_ID. Runtime stage
# rebuilds are fast (just copies + config).
#
# CAUTION: the OVMS GPU image targets Intel Compute Runtime versions that
# may not match Panther Lake. If `--target_device GPU` fails to enumerate
# the iGPU, rebuild the runtime stage on top of ubuntu:24.04 with the
# kobuk-team/intel-graphics PPA and install OVMS as a user binary.

ARG OVMS_IMAGE=openvino/model_server:latest-gpu

# ── Stage 1: model fetcher ─────────────────────────────────────────────────
FROM ubuntu:24.04 AS fetcher

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir huggingface_hub

ARG MODEL_ID=OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov
ARG MODEL_DIR=/models/qwen3-coder-30b-a3b-int4

RUN hf download "${MODEL_ID}" --local-dir "${MODEL_DIR}" && \
    du -sh "${MODEL_DIR}"


# ── Stage 2: OVMS runtime ──────────────────────────────────────────────────
FROM ${OVMS_IMAGE}

ARG MODEL_DIR=/models/qwen3-coder-30b-a3b-int4
ARG MODEL_NAME=qwen3-coder

USER root

# OVMS mediapipe layout: model files + graph.pbtxt at the same level,
# referenced by ovms-config.json's mediapipe_config_list.base_path.
COPY --from=fetcher ${MODEL_DIR} /models/${MODEL_NAME}/
COPY graph.pbtxt /models/${MODEL_NAME}/graph.pbtxt
COPY ovms-config.json /workspace/config.json

EXPOSE 8000

CMD ["--config_path", "/workspace/config.json", \
     "--rest_port", "8000", \
     "--port", "9000", \
     "--log_level", "INFO"]
