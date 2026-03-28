FROM openvino/ubuntu24_dev:latest

USER root

# Install system dependencies
RUN mkdir -p /var/lib/apt/lists/partial && \
    apt-get update -qq && \
    apt-get install -y -qq wget && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --break-system-packages --no-cache-dir \
    openvino-genai \
    "optimum[openvino]" \
    datasets

# Install Intel NPU driver (user-space components)
RUN wget -q https://github.com/intel/linux-npu-driver/releases/download/v1.30.0/linux-npu-driver-v1.30.0.20260311-22963593310-ubuntu2404.tar.gz && \
    tar -xf linux-npu-driver-v1.30.0*.tar.gz && \
    (dpkg -i *.deb || (apt-get update -qq && apt-get install -f -y -qq)) && \
    rm -f *.deb *.tar.gz && \
    rm -rf /var/lib/apt/lists/*

# Download and export model
# Use a pre-exported NPU-optimized model from Hugging Face
# Alternative: export your own with:
#   optimum-cli export openvino -m Qwen/Qwen2-0.5B-Instruct \
#     --weight-format int4 --sym --ratio 1.0 --group-size 128 /models/qwen2-npu
ARG MODEL_ID=OpenVINO/TinyLlama-1.1B-Chat-v1.0-int4-ov
ARG MODEL_DIR=/models/tinyllama-npu
RUN huggingface-cli download ${MODEL_ID} --local-dir ${MODEL_DIR}

# Copy chatbot script (last -- changes most often)
COPY npu-chatbot.py /app/npu-chatbot.py

WORKDIR /app
ENTRYPOINT ["python3", "/app/npu-chatbot.py"]
CMD ["--model", "/models/tinyllama-npu", "--device", "NPU"]
