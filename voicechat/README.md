# Voice Chat

Voice-driven chatbot running fully on-device on an Intel NUC (Panther Lake or similar), splitting inference across the GPU, NPU, and CPU.

## How it works

Each component is assigned to the device where it runs best:

| Component | Device | Model |
|-----------|--------|-------|
| ASR (speech-to-text) | GPU | Whisper (tiny / base / small) FP16 |
| LLM (language model) | GPU | Qwen3-8B INT8 (default; see `models.conf` for others) |
| TTS (text-to-speech) | CPU | Microsoft SpeechT5 |
| Wake word detection | NPU | openWakeWord classifiers |

The pipeline:

```
[Browser mic] → WebM/Opus → ffmpeg → float32 PCM 16kHz
  → [Whisper/GPU] → text
  → [LLM/GPU] → streamed tokens → clause boundary detection
  → [SpeechT5/CPU] → WAV per clause → [Browser AudioContext] → gapless playback

[Browser mic] → continuous 16kHz int16 PCM chunks
  → [melspectrogram/CPU] → [embedding/NPU] → [classifier/NPU]
  → wake word detected → start recording
```

TTS fires per clause inside the LLM streamer, so the user hears the first sentence while the model is still generating.

## Requirements

- Intel NUC (or similar) with integrated GPU and NPU
- k3s with Intel GPU plugin (`gpu.intel.com/xe`) and NPU plugin (`npu.intel.com/accel`)
- Helm
- buildkitd + nerdctl (for building images)
- At least 20GB free disk space for the build
- A privileged LXC container (required for overlayfs snapshotter — see [LXC-NOTES.md](../LXC-NOTES.md))

## Quick start

```bash
# Copy files to the node
rsync -avz voicechat/ user@<node>:~/voicechat/

# On the node:
cd ~/voicechat
make build MODEL=qwen3-8b
make deploy MODEL=qwen3-8b DEVICES=npu,gpu
make logs
```

Once the pod is ready, open `http://<node-ip>:30082` in a browser.

## Build options

```
make build [MODEL=qwen3-8b] [WHISPER=base] [SPEAKER=ljspeech] [TTS_DEVICE=CPU]
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3-8b` | LLM to bake into the image (see `make models`) |
| `WHISPER` | `base` | Whisper variant: `tiny`, `base`, or `small` |
| `SPEAKER` | `ljspeech` | TTS voice (see `make voices`) |
| `TTS_DEVICE` | `CPU` | TTS inference device (`CPU` only — GPU fails on SpeechT5 dynamic shapes) |

List available models and voices:

```bash
make models
make voices
```

The LLM and Whisper models are downloaded from HuggingFace at build time and baked into the image. A full build takes 20–40 minutes depending on model size and network speed.

## Deploy options

```
make deploy [MODEL=qwen3-8b] [DEVICES=npu,gpu] [WHISPER=base] [SPEAKER=ljspeech]
```

`DEVICES` controls which hardware resource requests appear in the pod spec. Valid values: `npu,gpu`, `gpu`, `npu`, or empty (CPU only, for testing).

The pod is exposed on NodePort **30082**.

## Using the web UI

- **Push-to-talk (PTT):** Hold the microphone button while speaking, release to send.
- **Wake word:** Say the active wake word (e.g. "Hey Athena") — recording starts automatically and stops after 1.5 seconds of silence.
- **Think toggle:** Enables Qwen3's extended reasoning mode (`<think>` blocks). Think tokens appear in italics in the UI and are never spoken aloud.
- **Wake word selector:** Appears in the header if more than one wake word is loaded. Switches the active classifier without restarting the pod.
- **Reset conversation:** Clears the LLM chat history (server-side context window).

### Browser mic on HTTP

The browser blocks microphone access on non-localhost HTTP origins. To fix:

- **Chrome:** Go to `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, add `http://<node-ip>:30082`, relaunch.
- **Firefox:** Go to `about:config`, set `media.devices.insecure.enabled` to `true`.

## Wake word training

Train a custom wake word and bake it into the image:

```bash
# Build the training container (only needed once)
make build-trainer

# Train a wake word (takes ~10 minutes for 1500 samples / 15000 steps)
make train-wakeword WAKEWORD_PHRASE="hey iris"

# For better accuracy (slower):
make train-wakeword WAKEWORD_PHRASE="hey iris" WAKEWORD_SAMPLES=8000 WAKEWORD_STEPS=20000

# Copy the output into the image and rebuild
cp /tmp/wakeword_output/hey_iris.onnx wakewords/
make build
```

Bundled wake words (in `wakewords/`): `Hey_Athena`, `hey_iris`, `hey_mycroft`.

## Other make targets

```
make logs            Stream pod logs
make status          Show pod status and resource allocation
make undeploy        Remove the pod
make nuke            Kill orphaned containerd state (use when pod is stuck)
make clean           Remove all voicechat images from containerd
make build-all       Build images for every model in models.conf
```

## Models

All models run on GPU via `openvino_genai.LLMPipeline`. See `models.conf` for the full list.

| Name | Size | Notes |
|------|------|-------|
| `tinyllama` | ~1B INT8 | Extremely fast, lowest quality |
| `qwen3-1.7b` | ~1.7B INT8 | Fast and small |
| `phi3-mini` | ~3.8B INT8 | Phi-3-mini 128k context |
| `phi3.5-mini` | ~3.8B INT8 | |
| `qwen3-4b` | ~4B INT8 | |
| `phi4-mini` | ~4B INT4 | No INT8 available |
| `qwen3-8b` | ~8B INT8 | Good balance of speed and quality; **default** |
| `mistral-7b` | ~7B INT8 | |
| `qwen25-14b` | ~14B INT8 | |
| `qwen3-14b` | ~14B INT8 | Best quality; has thinking mode |

Whisper variants:

| Variant | Notes |
|---------|-------|
| `tiny` | Fastest (~0.5s), lower accuracy |
| `base` | Default — good balance |
| `small` | Best accuracy, ~2–3s latency |

## HTTP API

All endpoints are served by the FastAPI application on port 8080.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/status` | GET | Readiness check — returns `{"ready": true}` once all engines are loaded |
| `/model-info` | GET | Engine details: LLM name, device, Whisper variant, TTS voice, available wake words |
| `/settings` | GET | Current runtime settings: `thinking_enabled`, `wakeword` (active classifier name) |
| `/settings` | POST | Update settings: `{"thinking_enabled": true}` and/or `{"wakeword": "hey_iris"}` |
| `/voice-chat` | WebSocket | Full-duplex voice conversation (see below) |
| `/transcribe` | POST | Multipart audio file → `{"text": "..."}` (ASR only) |
| `/chat` | POST | `{"message": "..."}` → SSE token stream (LLM only, no ASR/TTS) |
| `/speak` | POST | `{"text": "..."}` → WAV audio (TTS only) |
| `/reset` | POST | Clear LLM conversation history |

### `/status` response

```json
{"ready": true}
{"ready": false, "status": "Loading Whisper..."}
```

### `/model-info` response

```json
{
  "llm": "qwen3-8b",
  "device": "GPU",
  "whisper": "whisper-base",
  "whisper_device": "GPU",
  "voice": "ljspeech",
  "wakeword": "Hey_Athena",
  "available_wakewords": ["Hey_Athena", "hey_iris", "hey_mycroft"]
}
```

## WebSocket protocol (`/voice-chat`)

The web UI communicates over a single persistent WebSocket. Binary frames carry audio; JSON frames carry control messages and events.

### Client → server

| Message | Description |
|---------|-------------|
| `<binary>` | Raw audio recording (WebM/Opus from MediaRecorder). Triggers the full ASR → LLM → TTS pipeline. |
| `{"type": "ww_audio", "data": "<base64 int16 PCM>"}` | Continuous 80ms mic chunks for wake word detection. Sent at 16kHz int16. |
| `{"type": "reset"}` | Clear conversation history. |

### Server → client

| Message | Description |
|---------|-------------|
| `{"type": "status", "stage": "transcribing\|thinking\|speaking\|idle"}` | Pipeline stage changes. |
| `{"type": "transcript", "text": "..."}` | ASR result — the user's transcribed speech. |
| `{"type": "llm_token", "text": "...", "thinking": true\|false}` | Streamed LLM token. `thinking: true` means the token is inside a `<think>` block. |
| `{"type": "llm_done", "text": "..."}` | LLM generation complete. `text` is the full response. |
| `{"type": "audio_chunk", "data": "<base64 WAV>"}` | One TTS audio chunk (one clause). Play in sequence for gapless output. |
| `{"type": "audio_done"}` | All TTS chunks for this response have been sent. |
| `{"type": "wakeword", "score": 0.85}` | Wake word detected with given confidence score. |
| `{"type": "reset_done"}` | Conversation history cleared. |
| `{"type": "error", "message": "..."}` | Pipeline error. |

### Audio format notes

- **Inbound audio** (PTT recording): WebM/Opus container, decoded server-side with ffmpeg to float32 PCM at 16kHz mono.
- **Wake word audio** (`ww_audio`): raw int16 PCM at 16kHz, base64-encoded. No container wrapper.
- **Outbound audio** (`audio_chunk`): WAV container, 16kHz mono int16, base64-encoded. Each chunk covers one clause (sentence fragment), scheduled for gapless playback via `AudioContext.createBufferSource`.

## File structure

```
voicechat/
  voicechat.py                  # FastAPI application: ASR + LLM + TTS pipeline
  wakeword.py                   # WakeWordDetector: mel (CPU) + embedding (NPU) + classifier (NPU)
  models.conf                   # LLM model registry (NAME|HF_MODEL_ID|MODEL_DIR)
  voices.conf                   # TTS voice registry
  Dockerfile                    # 3-stage: tts-builder, wakeword-builder, runtime
  Dockerfile.train-wakeword     # Wake word training container
  train_wakeword_container.py   # Training script (runs inside training container)
  extract_speaker_embedding.py  # CMU Arctic x-vector extraction for SpeechT5
  extract_wakeword_models.py    # Extracts melspectrogram/embedding ONNX from pip package
  wakewords/                    # Wake word classifier ONNX files (baked into image)
  static/index.html             # Web UI
  chart/                        # Helm chart
  Makefile
```
