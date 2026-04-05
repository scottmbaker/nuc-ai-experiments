FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and Intel GPU compute runtime for Panther Lake
RUN apt-get update -qq && \
    apt-get install -y -qq wget curl software-properties-common python3 python3-pip python3-venv && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -qq && \
    apt-get install -y -qq intel-opencl-icd libze-intel-gpu1 libze1 && \
    rm -rf /var/lib/apt/lists/*

# Install OpenVINO and Python packages via pip (no bundled GPU plugin to conflict)
RUN pip3 install --break-system-packages --no-cache-dir \
    openvino==2026.0.0 \
    openvino-genai \
    "optimum[openvino]" \
    diffusers \
    transformers \
    accelerate \
    fastapi \
    uvicorn[standard] \
    torch

# Install Intel NPU driver (user-space components)
RUN wget -q https://github.com/intel/linux-npu-driver/releases/download/v1.30.0/linux-npu-driver-v1.30.0.20260311-22963593310-ubuntu2404.tar.gz && \
    tar -xf linux-npu-driver-v1.30.0*.tar.gz && \
    (dpkg -i *.deb || (apt-get update -qq && apt-get install -f -y -qq)) && \
    rm -f *.deb *.tar.gz && \
    rm -rf /var/lib/apt/lists/*

# Download or export model
ARG MODEL_ID=SimianLuo/LCM_Dreamshaper_v7
ARG MODEL_DIR=/models/lcm-dreamshaper
ARG PIPELINE_CLASS=OVLatentConsistencyModelPipeline
ARG EXPORT_TASK=stable-diffusion
ARG DOWNLOAD_METHOD=export
ARG DEFAULT_STEPS=4
ARG DEFAULT_GUIDANCE=1.0

RUN if [ "$DOWNLOAD_METHOD" = "export" ]; then \
      optimum-cli export openvino --model ${MODEL_ID} --weight-format int8 --task ${EXPORT_TASK} ${MODEL_DIR}; \
    else \
      huggingface-cli download ${MODEL_ID} --local-dir ${MODEL_DIR}; \
    fi && \
    echo "${PIPELINE_CLASS}" > ${MODEL_DIR}/pipeline_class.txt && \
    echo "${DEFAULT_STEPS}" > ${MODEL_DIR}/default_steps.txt && \
    echo "${DEFAULT_GUIDANCE}" > ${MODEL_DIR}/default_guidance.txt

# Copy application files
COPY imagegen.py /app/imagegen.py
COPY static /app/static

WORKDIR /app
EXPOSE 8080
ENTRYPOINT ["python3", "/app/imagegen.py"]
CMD ["--model", "/models/lcm-dreamshaper"]
