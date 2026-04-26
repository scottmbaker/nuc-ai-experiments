## Stage 1: export models to OpenVINO IR ─────────────────────────────────────
FROM ubuntu:24.04 AS model-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ffmpeg wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        ultralytics torch openvino nncf "fast-alpr[onnx]"

COPY export_models.py /tmp/export_models.py
RUN python3 /tmp/export_models.py

# Download test videos (all Pexels license — free, no attribution required)
RUN wget -q -O /models/nyc-street.mp4 \
        "https://videos.pexels.com/video-files/3773324/3773324-hd_1280_720_25fps.mp4" && \
    wget -q -O /models/parking-lot.mp4 \
        "https://videos.pexels.com/video-files/5579757/5579757-hd_1920_1080_30fps.mp4" && \
    wget -q -O /models/dashcam-highway.mp4 \
        "https://videos.pexels.com/video-files/5915075/5915075-hd_1920_1080_30fps.mp4" && \
    wget -q -O /models/london-traffic.mp4 \
        "https://videos.pexels.com/video-files/3206967/3206967-hd_1920_1080_30fps.mp4" && \
    wget -q -O /models/park-people.mp4 \
        "https://videos.pexels.com/video-files/5353604/5353604-hd_1920_1080_30fps.mp4" && \
    wget -q -O /models/intersection.mp4 \
        "https://videos.pexels.com/video-files/4791734/4791734-hd_1920_1080_30fps.mp4"


## Stage 2: final runtime image ──────────────────────────────────────────────
FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip \
        wget curl ffmpeg \
        software-properties-common gpg-agent && \
    rm -rf /var/lib/apt/lists/*

# Intel GPU compute runtime
RUN wget -qO- https://repositories.intel.com/gpu/intel-graphics.key | \
        gpg --dearmor -o /usr/share/keyrings/intel-graphics.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
        https://repositories.intel.com/gpu/ubuntu noble unified" \
        > /etc/apt/sources.list.d/intel-gpu.list && \
    apt-get update && apt-get install -y --no-install-recommends \
        intel-opencl-icd libze-intel-gpu1 libze1 && \
    rm -rf /var/lib/apt/lists/*

# Intel NPU driver
RUN wget -q https://github.com/intel/linux-npu-driver/releases/download/v1.30.0/linux-npu-driver-v1.30.0.20260311-22963593310-ubuntu2404.tar.gz && \
    tar -xf linux-npu-driver-v1.30.0*.tar.gz && \
    (dpkg -i *.deb || (apt-get update -qq && apt-get install -f -y -qq)) && \
    rm -f *.deb *.tar.gz && \
    rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip3 install --break-system-packages \
        openvino \
        opencv-python-headless \
        numpy \
        fastapi \
        "uvicorn[standard]" \
        pyyaml \
        requests \
        "fast-alpr[onnx-openvino]"

# Copy models from builder
COPY --from=model-builder /models /models
COPY --from=model-builder /root/.cache /root/.cache

# Application
WORKDIR /app
COPY securitycam.py /app/
COPY config.yaml /app/
COPY static/ /app/static/

RUN mkdir -p /app/clips /app/data /app/videos && \
    ln -s /models/nyc-street.mp4 /app/videos/nyc-street.mp4 && \
    ln -s /models/parking-lot.mp4 /app/videos/parking-lot.mp4 && \
    ln -s /models/dashcam-highway.mp4 /app/videos/dashcam-highway.mp4 && \
    ln -s /models/london-traffic.mp4 /app/videos/london-traffic.mp4 && \
    ln -s /models/park-people.mp4 /app/videos/park-people.mp4 && \
    ln -s /models/intersection.mp4 /app/videos/intersection.mp4

EXPOSE 8080
ENTRYPOINT ["python3", "/app/securitycam.py"]
CMD ["--test"]
