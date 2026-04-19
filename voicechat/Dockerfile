
# ── Stage 1: SpeechT5 model export ──────────────────────────────────────────
# Exports microsoft/speecht5_tts + speecht5_hifigan vocoder to OpenVINO IR.
# Also extracts a speaker embedding from the CMU Arctic x-vectors dataset.
FROM ubuntu:24.04 AS tts-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -y -qq python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages --no-cache-dir \
    "optimum[openvino]" \
    openvino-tokenizers \
    "transformers>=4.45" \
    "torch>=2.1,<2.9" \
    "datasets<4" \
    "numpy<2" \
    sentencepiece \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Export SpeechT5 TTS + HiFi-GAN vocoder to OpenVINO IR
RUN optimum-cli export openvino \
    --model microsoft/speecht5_tts \
    --model-kwargs '{"vocoder":"microsoft/speecht5_hifigan"}' \
    /models/speecht5

# Convert the SpeechT5 tokenizer to OpenVINO format
RUN python3 -c "\
from transformers import SpeechT5Processor; \
import openvino_tokenizers; \
import openvino as ov; \
proc = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts'); \
ov_tokenizer = openvino_tokenizers.convert_tokenizer(proc.tokenizer); \
ov.save_model(ov_tokenizer, '/models/speecht5/openvino_tokenizer.xml'); \
print('Tokenizer exported')"

# Extract a speaker x-vector embedding from the CMU Arctic dataset
# Speaker 7306 is the commonly-used American female voice
COPY extract_speaker_embedding.py /tmp/extract_speaker_embedding.py
RUN python3 /tmp/extract_speaker_embedding.py 7306 /models/speecht5/speaker.bin


# ── Stage 1b: Extract openWakeWord ONNX models ─────────────────────────────
FROM ubuntu:24.04 AS wakeword-builder
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -qq && apt-get install -y -qq python3 python3-pip && rm -rf /var/lib/apt/lists/*
RUN pip3 install --break-system-packages --no-cache-dir openwakeword
COPY extract_wakeword_models.py /tmp/extract_wakeword_models.py
RUN python3 /tmp/extract_wakeword_models.py /models/wakeword


# ── Stage 2: Main application image ──────────────────────────────────────────
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# System deps + Intel GPU compute runtime (Panther Lake)
RUN apt-get update -qq && \
    apt-get install -y -qq \
        wget curl git software-properties-common \
        python3 python3-pip \
        ffmpeg \
        espeak-ng && \
    add-apt-repository -y ppa:kobuk-team/intel-graphics && \
    apt-get update -qq && \
    apt-get install -y -qq intel-opencl-icd libze-intel-gpu1 libze1 && \
    rm -rf /var/lib/apt/lists/*

# Python packages
RUN pip3 install --break-system-packages --no-cache-dir \
    openvino-genai \
    openvino \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    numpy \
    soundfile \
    huggingface_hub

# Intel NPU driver (user-space components)
RUN wget -q https://github.com/intel/linux-npu-driver/releases/download/v1.30.0/linux-npu-driver-v1.30.0.20260311-22963593310-ubuntu2404.tar.gz && \
    tar -xf linux-npu-driver-v1.30.0*.tar.gz && \
    (dpkg -i *.deb || (apt-get update -qq && apt-get install -f -y -qq)) && \
    rm -f *.deb *.tar.gz && \
    rm -rf /var/lib/apt/lists/*

# ── Copy SpeechT5 from builder ──────────────────────────────────────────────
COPY --from=tts-builder /models/speecht5 /models/speecht5

# ── Whisper ASR model (pre-exported FP16 OpenVINO from HuggingFace) ──────────
ARG WHISPER=base
RUN hf download OpenVINO/whisper-${WHISPER}-fp16-ov \
    --local-dir /models/whisper-${WHISPER}

# ── LLM model (OpenVINO IR, baked into image) ─────────────────────────────────
ARG MODEL_ID=OpenVINO/Qwen3-8B-int8-ov
ARG MODEL_DIR=/models/qwen3-8b
RUN hf download ${MODEL_ID} --local-dir ${MODEL_DIR}

# ── Copy wake word ONNX models (after model downloads — changes here don't bust download cache)
COPY --from=wakeword-builder /models/wakeword /models/wakeword
COPY wakewords/ /models/wakeword/

# Copy application files last — changes here don't bust the model download cache
COPY voicechat.py /app/voicechat.py
COPY wakeword.py /app/wakeword.py
COPY models.conf /app/models.conf
COPY voices.conf /app/voices.conf
COPY static /app/static

WORKDIR /app
EXPOSE 8080

ENTRYPOINT ["python3", "/app/voicechat.py"]
CMD ["--model", "/models/qwen3-8b", \
     "--whisper", "/models/whisper-base", \
     "--tts-dir", "/models/speecht5", \
     "--speaker", "ljspeech", \
     "--tts-device", "CPU", \
     "--device", "GPU"]
