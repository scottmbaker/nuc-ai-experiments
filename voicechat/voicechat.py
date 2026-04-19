#!/usr/bin/env python3
"""
Voice chatbot for Intel Panther Lake NUC.

  ASR:       Whisper on GPU  (OpenVINO GenAI WhisperPipeline, FP16)
  LLM:       GPU             (OpenVINO GenAI LLMPipeline)
  TTS:       CPU             (OpenVINO GenAI Text2SpeechPipeline, SpeechT5)
  Wake word: NPU             (openWakeWord ONNX via WakeWordDetector)

Usage:
    python3 voicechat.py \
        --model /models/qwen3-14b \
        --whisper /models/whisper-base \
        --tts-dir /models/speecht5 \
        --device GPU
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import wave

import numpy as np
import openvino as ov
import openvino_genai
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from pathlib import Path
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("voicechat")

# Flush TTS on sentence-ending punctuation or clause separators (comma, semicolon),
# when followed by whitespace. This drives low-latency streaming: the first clause
# is spoken while the LLM is still generating the rest.
CLAUSE_END_RE = re.compile(r'(?<=[.!?,;])\s+')

# Maximum tokens to buffer before forcing a TTS flush, regardless of punctuation.
# Guards against very long run-on sentences that would delay first audio.
TTS_MAX_TOKENS = 25

# Whisper hallucinates these phrases on silence or near-silence input.
# After stripping punctuation and lowercasing, any transcript matching one of
# these exactly is dropped rather than sent to the LLM.
WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe",
    "thank you",
    "thanks",
    "you",
}

# RMS energy threshold below which audio is treated as silence and ASR is skipped.
# Float32 audio is normalised to [-1, 1]; 0.01 ≈ -40 dBFS.
SILENCE_RMS_THRESHOLD = 0.01

# Written on every status change so external tools (e.g. a status bar widget)
# can monitor loading progress without polling the HTTP API.
STATUS_FILE = "/tmp/voicechat-status.json"

# Global application context. All engines and shared state live here.
# Populated by load_initial() running in a background thread at startup.
ctx = {
    "llm_pipe":        None,
    "whisper_pipe":    None,
    "tts_pipe":        None,
    "device":          None,   # LLM device (resolved at load time)
    "whisper_device":  None,
    "tts_device":      None,
    "speaker":         None,
    "speaker_embedding": None,
    "wakeword":        None,
    "status":          "starting",
    "devices":         [],
    "sample_rate":     16000,  # SpeechT5 output sample rate
    "ready_event":     threading.Event(),
    "thinking_enabled": False,
}

# Serializes LLM generate() calls. Only one inference request at a time.
gen_lock = threading.Lock()


def write_status(status, device=None):
    ctx["status"] = status
    if device is not None:
        ctx["device"] = device
    data = {
        "status": status,
        "device": ctx["device"],
        "whisper_device": ctx["whisper_device"],
        "devices": ctx["devices"],
    }
    try:
        with open(STATUS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="Voice chatbot on Intel NPU/GPU/CPU")
    parser.add_argument("--model", required=True,
                        help="Path to LLM model directory (OpenVINO IR, quantized)")
    parser.add_argument("--device", default="GPU", choices=["NPU", "CPU", "GPU"],
                        help="LLM inference device (default: GPU)")
    parser.add_argument("--whisper", required=True,
                        help="Path to Whisper model directory (OpenVINO FP16)")
    parser.add_argument("--whisper-device", default="GPU", choices=["GPU", "CPU"],
                        help="Whisper inference device (default: GPU)")
    parser.add_argument("--tts-dir", required=True,
                        help="Path to TTS model directory (SpeechT5 OpenVINO IR)")
    parser.add_argument("--speaker", default="ljspeech",
                        help="TTS voice name (informational; actual voice comes from speaker.bin)")
    parser.add_argument("--tts-device", default="CPU", choices=["GPU", "CPU"],
                        help="TTS inference device (default: CPU — GPU fails on SpeechT5 dynamic shapes)")
    parser.add_argument("--max-prompt-len", type=int, default=2048,
                        help="Max prompt length tokens for NPU (default: 2048)")
    parser.add_argument("--min-response-len", type=int, default=256,
                        help="Reserved response tokens for NPU (default: 256)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max tokens per LLM response (default: 512)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--system", type=str,
                        default=(
                            "You are a helpful, concise voice assistant. "
                            "Give brief, direct answers suitable for spoken conversation. "
                            "Do not use markdown, bullet points, or numbered lists."
                        ))
    parser.add_argument("--wakeword-dir", default="/models/wakeword",
                        help="Path to wake word ONNX model directory")
    parser.add_argument("--wakeword", default="Hey_Athena",
                        help="Wake word model name to activate on startup (default: Hey_Athena)")
    parser.add_argument("--wakeword-threshold", type=float, default=0.5,
                        help="Wake word detection threshold (default: 0.5)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable NPU blob caching")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def build_pipeline_kwargs(model_path, device, args):
    """
    Build extra kwargs for LLMPipeline construction.

    NPU requires static shapes, so max_prompt_len and min_response_len must be
    specified at compile time. Blobs are cached in the model directory to avoid
    recompilation on subsequent runs (compilation can take several minutes).
    """
    kwargs = {}
    if device == "NPU":
        kwargs["max_prompt_len"] = args.max_prompt_len
        kwargs["min_response_len"] = args.min_response_len
        if not args.no_cache:
            blob_dir = os.path.join(model_path, "npu_blobs")
            os.makedirs(blob_dir, exist_ok=True)
            kwargs["blob_path"] = blob_dir
    return kwargs


def effective_system_prompt(args):
    """
    Return the system prompt, prefixed with /no_think when thinking is disabled.

    /no_think is a Qwen3 chat-template directive that suppresses <think> block
    generation. Without it, thinking-capable models emit verbose reasoning tokens
    before every response, which adds latency and is not suitable for voice.
    """
    if ctx["thinking_enabled"]:
        return args.system
    return "/no_think " + args.system


def build_gen_config(args):
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.repetition_penalty = args.repetition_penalty
    config.do_sample = True
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    return config


def detect_devices():
    """Return a list of available OpenVINO compute devices (always includes CPU)."""
    core = ov.Core()
    devices = ["CPU"]
    hw = set(core.available_devices)
    if "GPU" in hw:
        devices.append("GPU")
    if "NPU" in hw:
        devices.append("NPU")
    return devices


def load_initial(args):
    """
    Background thread: load all inference engines in sequence.

    Loading order matters: Whisper and TTS load quickly; LLM (especially on NPU
    first run) can take several minutes. Wake word is loaded before LLM so the
    NPU is available immediately after server ready.
    """
    try:
        write_status("detecting devices")
        ctx["devices"] = detect_devices()
        log.info(f"Available devices: {ctx['devices']}")

        # 1. Whisper ASR — GPU (FP16 OpenVINO export from HuggingFace)
        whisper_dev = args.whisper_device if args.whisper_device in ctx["devices"] else "CPU"
        ctx["whisper_device"] = whisper_dev
        write_status(f"loading Whisper on {whisper_dev}")
        log.info(f"Loading Whisper from {args.whisper} on {whisper_dev}...")
        t0 = time.time()
        ctx["whisper_pipe"] = openvino_genai.WhisperPipeline(args.whisper, whisper_dev)
        log.info(f"Whisper ready in {time.time() - t0:.1f}s")

        # 2. SpeechT5 TTS — CPU
        # GPU fails with "Argument shapes are inconsistent" on SpeechT5's autoregressive
        # decoder due to dynamic shapes. CPU is very responsive despite the placement.
        write_status("loading TTS")
        tts_dev = args.tts_device if args.tts_device in ctx["devices"] else "CPU"
        log.info(f"Loading SpeechT5 from {args.tts_dir} on {tts_dev}...")
        t0 = time.time()
        ctx["tts_pipe"] = openvino_genai.Text2SpeechPipeline(args.tts_dir, tts_dev)

        # Speaker embedding: 512-dim x-vector from CMU Arctic dataset.
        # Encodes voice characteristics. Extracted at image build time from speaker 7306
        # (American female). See extract_speaker_embedding.py.
        spk_path = os.path.join(args.tts_dir, "speaker.bin")
        if os.path.exists(spk_path):
            spk_np = np.fromfile(spk_path, dtype=np.float32).reshape(1, 512)
            ctx["speaker_embedding"] = ov.Tensor(spk_np)
            log.info(f"Speaker embedding loaded from {spk_path}")
        else:
            ctx["speaker_embedding"] = None
            log.info("No speaker embedding — using model default")

        ctx["tts_device"] = tts_dev
        ctx["speaker"] = args.speaker
        log.info(f"SpeechT5 ready in {time.time() - t0:.1f}s on {tts_dev}")

        # 3. Wake word detector — NPU
        # Three-stage pipeline: melspectrogram (CPU) → embedding CNN (NPU) → classifier (NPU).
        # Melspectrogram runs on CPU because the reflect-pad op is not supported on NPU.
        if os.path.isdir(args.wakeword_dir):
            write_status("loading wake word on NPU")
            log.info(f"Loading wake word '{args.wakeword}' from {args.wakeword_dir} on NPU...")
            t0 = time.time()
            from wakeword import WakeWordDetector
            try:
                ctx["wakeword"] = WakeWordDetector(
                    model_dir=args.wakeword_dir,
                    wake_word=args.wakeword,
                    threshold=args.wakeword_threshold,
                    npu_device="NPU",
                )
                log.info(f"Wake word ready in {time.time() - t0:.1f}s on NPU")
            except Exception as e:
                log.warning(f"Wake word init failed: {e} — wake word disabled")
                ctx["wakeword"] = None
        else:
            log.info("No wake word directory — wake word disabled")
            ctx["wakeword"] = None

        # 4. LLM — GPU (falls back to CPU if GPU unavailable)
        # On first NPU run, model blobs are compiled and cached — this can take
        # several minutes. Subsequent runs load from cache in seconds.
        llm_dev = args.device if args.device in ctx["devices"] else "CPU"
        write_status(f"loading LLM on {llm_dev}")
        log.info(f"Loading LLM from {args.model} on {llm_dev}...")
        if llm_dev == "NPU":
            log.info("(First run on NPU compiles blobs — may take several minutes)")
        t0 = time.time()
        kwargs = build_pipeline_kwargs(args.model, llm_dev, args)
        ctx["llm_pipe"] = openvino_genai.LLMPipeline(args.model, llm_dev, **kwargs)
        # start_chat / finish_chat are deprecated in recent openvino_genai but still functional.
        ctx["llm_pipe"].start_chat(effective_system_prompt(args))
        ctx["device"] = llm_dev
        log.info(f"LLM ready in {time.time() - t0:.1f}s")

        write_status("ready", device=llm_dev)
        ctx["ready_event"].set()
        log.info("All engines ready")

    except Exception as e:
        write_status(f"error: {e}")
        ctx["ready_event"].set()
        log.error(f"Failed to load engines: {e}", exc_info=True)


# ── Audio helpers ────────────────────────────────────────────────────────────

def decode_audio(audio_bytes: bytes) -> np.ndarray:
    """Decode arbitrary audio (WebM/Opus/WAV/etc.) to float32 at 16kHz mono."""
    cmd = [
        "ffmpeg", "-y",
        "-i", "pipe:0",
        "-f", "f32le",
        "-ar", "16000",
        "-ac", "1",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=audio_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg: {proc.stderr.decode()[:300]}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def run_asr(audio_f32: np.ndarray) -> str:
    """Transcribe float32 16kHz mono audio with Whisper. Returns raw transcript string."""
    result = ctx["whisper_pipe"].generate(audio_f32.tolist())
    if hasattr(result, "texts"):
        return result.texts[0].strip()
    return str(result).strip()


def run_tts(text: str) -> bytes:
    """
    Synthesize text to WAV bytes using SpeechT5 on CPU.

    Returns a 16kHz mono WAV. The speaker embedding (loaded from speaker.bin)
    controls voice characteristics. If no embedding is present the model uses
    its internal default voice.
    """
    spk = ctx.get("speaker_embedding")
    if spk is not None:
        result = ctx["tts_pipe"].generate(text, spk)
    else:
        result = ctx["tts_pipe"].generate(text)

    speech = result.speeches[0]
    # speech.data[0] raises RuntimeError if the tensor lives on a remote device.
    # Copy to host first in that case.
    try:
        raw = speech.data[0]
    except RuntimeError:
        host = ov.Tensor(speech.element_type, speech.shape)
        speech.copy_to(host)
        raw = host.data[0]

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(ctx["sample_rate"])
        audio_int16 = (raw * 32767).clip(-32768, 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ── Web server ───────────────────────────────────────────────────────────────

def run_web(args):
    app = FastAPI()

    class TextRequest(BaseModel):
        message: str

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return Path("/app/static/index.html").read_text()

    @app.get("/status")
    async def status():
        return {
            "ready": ctx["status"] == "ready",
            "status": ctx["status"],
            "device": ctx["device"],
            "whisper_device": ctx["whisper_device"],
            "devices": ctx["devices"],
        }

    @app.get("/model-info")
    async def model_info():
        ww = ctx.get("wakeword")
        return {
            "llm":              os.path.basename(args.model),
            "whisper":          os.path.basename(args.whisper),
            "voice":            ctx.get("speaker", args.speaker),
            "device":           ctx["device"],
            "whisper_device":   ctx["whisper_device"],
            "tts_device":       ctx.get("tts_device", "CPU"),
            "wakeword":         ww.active if ww else None,
            "available_wakewords": ww.available if ww else [],
        }

    def _settings_response():
        ww = ctx.get("wakeword")
        return {
            "thinking_enabled":    ctx["thinking_enabled"],
            "wakeword":            ww.active if ww else None,
            "available_wakewords": ww.available if ww else [],
        }

    @app.get("/settings")
    async def get_settings():
        return _settings_response()

    @app.post("/settings")
    async def set_settings(req: dict):
        if "thinking_enabled" in req:
            ctx["thinking_enabled"] = bool(req["thinking_enabled"])
            # Restart the chat session so the new system prompt (with/without /no_think) takes effect.
            with gen_lock:
                if ctx["llm_pipe"] is not None:
                    ctx["llm_pipe"].finish_chat()
                    ctx["llm_pipe"].start_chat(effective_system_prompt(args))
            log.info(f"Thinking mode: {'enabled' if ctx['thinking_enabled'] else 'disabled'}")
        if "wakeword" in req:
            ww = ctx.get("wakeword")
            if ww is None:
                raise HTTPException(503, "Wake word not loaded")
            try:
                ww.active = req["wakeword"]
            except ValueError as e:
                raise HTTPException(400, str(e))
        return _settings_response()

    @app.post("/transcribe")
    async def transcribe(audio: UploadFile = File(...)):
        """Audio file → transcript text. Useful for testing ASR in isolation."""
        if ctx["status"] != "ready":
            raise HTTPException(503, f"Not ready: {ctx['status']}")
        audio_bytes = await audio.read()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: run_asr(decode_audio(audio_bytes)))
        return {"text": result}

    @app.post("/speak")
    async def speak(req: TextRequest):
        """Text → WAV audio. Useful for testing TTS in isolation."""
        if ctx["status"] != "ready":
            raise HTTPException(503, f"Not ready: {ctx['status']}")
        text = req.message.strip()
        if not text:
            raise HTTPException(400, "Empty text")
        loop = asyncio.get_event_loop()
        wav = await loop.run_in_executor(None, lambda: run_tts(text))
        return Response(content=wav, media_type="audio/wav")

    @app.post("/chat")
    async def chat(req: TextRequest):
        """Text → SSE token stream. Full LLM conversation, no TTS."""
        if ctx["status"] != "ready":
            raise HTTPException(503, f"Not ready: {ctx['status']}")
        message = req.message.strip()
        if not message:
            raise HTTPException(400, "Empty message")

        loop = asyncio.get_event_loop()
        token_queue: asyncio.Queue = asyncio.Queue()

        def run_generate():
            with gen_lock:
                state = {"count": 0, "start": time.time(), "error": None, "in_think": False}

                def streamer(subword):
                    state["count"] += 1
                    if "<think>" in subword:
                        state["in_think"] = True
                    if state["in_think"]:
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put({"token": subword, "thinking": True}), loop)
                        if "</think>" in subword:
                            state["in_think"] = False
                    else:
                        asyncio.run_coroutine_threadsafe(
                            token_queue.put({"token": subword}), loop)
                    return openvino_genai.StreamingStatus.RUNNING

                config = build_gen_config(args)
                try:
                    ctx["llm_pipe"].generate(message, config, streamer=streamer)
                except Exception as e:
                    state["error"] = str(e)

                elapsed = time.time() - state["start"]
                tps = state["count"] / elapsed if elapsed > 0 else 0
                done = {
                    "done": True,
                    "tokens": state["count"],
                    "elapsed": round(elapsed, 1),
                    "tokens_per_sec": round(tps, 1),
                    "device": ctx["device"],
                }
                if state["error"]:
                    done["error"] = state["error"]
                asyncio.run_coroutine_threadsafe(token_queue.put(done), loop)

        threading.Thread(target=run_generate, daemon=True).start()

        async def event_stream():
            while True:
                data = await token_queue.get()
                yield f"data: {json.dumps(data)}\n\n"
                if "done" in data:
                    break

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/reset")
    async def reset_chat():
        """Clear LLM conversation history and restart with the system prompt."""
        with gen_lock:
            if ctx["llm_pipe"] is not None:
                ctx["llm_pipe"].finish_chat()
                ctx["llm_pipe"].start_chat(effective_system_prompt(args))
        return {"ok": True}

    @app.websocket("/voice-chat")
    async def voice_chat(websocket: WebSocket):
        """
        Full-duplex voice conversation WebSocket.

        Client → Server:
          binary frames:  raw audio (WebM/Opus from MediaRecorder, PTT release)
          text frames:    JSON control messages:
            {"type": "reset"}
            {"type": "ww_audio", "data": "<base64 int16 16kHz PCM>"}

        Server → Client:
          {"type": "status",      "stage": "transcribing|thinking|speaking|idle"}
          {"type": "transcript",  "text": "..."}
          {"type": "llm_token",   "text": "...", "thinking": true|false}
          {"type": "llm_done",    "text": "<full response>"}
          {"type": "audio_chunk", "data": "<base64 WAV>"}
          {"type": "audio_done"}
          {"type": "wakeword",    "score": 0.85}
          {"type": "reset_done"}
          {"type": "error",       "message": "..."}
        """
        await websocket.accept()
        loop = asyncio.get_event_loop()

        async def send(data: dict):
            try:
                await websocket.send_json(data)
            except Exception:
                pass

        # Thread-safe helper: schedule a WebSocket send from a non-async thread.
        def send_ts(data: dict):
            asyncio.run_coroutine_threadsafe(send(data), loop)

        # Set while a voice request is being processed. Prevents concurrent requests
        # and suppresses wake word processing during inference.
        busy = threading.Event()

        try:
            while True:
                msg = await websocket.receive()

                if msg["type"] == "websocket.disconnect":
                    break

                # ── Text control messages ────────────────────────────────────
                if "text" in msg and msg["text"]:
                    try:
                        data = json.loads(msg["text"])
                    except Exception:
                        continue

                    if data.get("type") == "reset":
                        with gen_lock:
                            if ctx["llm_pipe"] is not None:
                                ctx["llm_pipe"].finish_chat()
                                ctx["llm_pipe"].start_chat(effective_system_prompt(args))
                        await send({"type": "reset_done"})

                    elif data.get("type") == "ww_audio" and ctx.get("wakeword"):
                        # Continuous wake word audio stream from the browser mic.
                        # Each chunk is ~80ms of 16kHz int16 PCM, base64-encoded.
                        # Drop chunks while busy — no point detecting while processing.
                        if busy.is_set():
                            continue
                        try:
                            pcm = np.frombuffer(base64.b64decode(data["data"]), dtype=np.int16)
                            score = ctx["wakeword"].process(pcm)
                            if score is not None:
                                await send({"type": "wakeword", "score": round(score, 3)})
                                ctx["wakeword"].reset()
                        except Exception as e:
                            log.debug(f"Wake word error: {e}")
                    continue

                # ── Binary audio frame (PTT release) ─────────────────────────
                if "bytes" not in msg or not msg["bytes"]:
                    continue
                audio_bytes = msg["bytes"]

                if busy.is_set():
                    await send({"type": "error", "message": "busy — still processing previous request"})
                    continue

                if ctx["status"] != "ready":
                    await send({"type": "error", "message": f"Not ready: {ctx['status']}"})
                    continue

                def process_voice(audio_bytes=audio_bytes):
                    busy.set()
                    try:
                        # ── Stage 1: Decode + silence check + ASR ────────────
                        send_ts({"type": "status", "stage": "transcribing"})
                        try:
                            audio_f32 = decode_audio(audio_bytes)
                        except Exception as e:
                            send_ts({"type": "error", "message": f"Audio decode failed: {e}"})
                            send_ts({"type": "status", "stage": "idle"})
                            return

                        # Skip ASR if the audio is silent (e.g. PTT with no speech,
                        # or wake word triggered with nothing following).
                        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
                        if rms < SILENCE_RMS_THRESHOLD:
                            log.debug(f"Silent audio (RMS={rms:.4f}) — skipping ASR")
                            send_ts({"type": "status", "stage": "idle"})
                            return

                        try:
                            transcript = run_asr(audio_f32)
                        except Exception as e:
                            send_ts({"type": "error", "message": f"ASR failed: {e}"})
                            send_ts({"type": "status", "stage": "idle"})
                            return

                        # Drop known Whisper hallucinations on near-silence audio.
                        transcript_clean = re.sub(r"[^\w\s]", "", transcript).strip().lower()
                        if not transcript or transcript_clean in WHISPER_HALLUCINATIONS:
                            log.debug(f"Rejected transcript: {repr(transcript)}")
                            send_ts({"type": "status", "stage": "idle"})
                            return

                        send_ts({"type": "transcript", "text": transcript})

                        # ── Stage 2: LLM generation with streaming TTS ────────
                        # TTS is fired per clause inside the streamer callback so
                        # the user hears the first sentence while the LLM is still
                        # generating the rest of the response.
                        send_ts({"type": "status", "stage": "thinking"})
                        full_response = []
                        buffer = ""
                        token_count = 0
                        in_think = False

                        def streamer(subword):
                            nonlocal buffer, token_count, in_think
                            full_response.append(subword)

                            # <think>...</think> blocks are shown in the UI (italicised)
                            # but never sent to TTS — they're internal reasoning tokens.
                            if "<think>" in subword:
                                in_think = True
                                send_ts({"type": "llm_token", "text": subword, "thinking": True})
                                return openvino_genai.StreamingStatus.RUNNING
                            if in_think:
                                send_ts({"type": "llm_token", "text": subword, "thinking": True})
                                if "</think>" in subword:
                                    in_think = False
                                return openvino_genai.StreamingStatus.RUNNING

                            buffer += subword
                            token_count += 1
                            send_ts({"type": "llm_token", "text": subword})

                            # Runaway repetition detection — symptom of context overflow
                            # or temperature instability. Stop generation early.
                            if len(full_response) >= 8:
                                tok = full_response[-1]
                                if tok.strip() and all(t == tok for t in full_response[-8:]):
                                    log.warning("Token repetition detected — stopping generation")
                                    return openvino_genai.StreamingStatus.STOP
                            if len(full_response) >= 30:
                                tail = "".join(full_response[-30:])
                                half = len(tail) // 2
                                if tail[:half] == tail[half:]:
                                    log.warning("Phrase repetition detected — stopping generation")
                                    return openvino_genai.StreamingStatus.STOP

                            # Flush to TTS on clause boundary or token count limit.
                            m = CLAUSE_END_RE.search(buffer)
                            flush_text = None
                            if m:
                                flush_text = buffer[:m.start() + 1].strip()
                                buffer = buffer[m.end():]
                                token_count = 0
                            elif token_count >= TTS_MAX_TOKENS:
                                # Flush at a word boundary to avoid cutting mid-word.
                                last_space = buffer.rfind(' ')
                                if last_space > 0:
                                    flush_text = buffer[:last_space].strip()
                                    buffer = buffer[last_space + 1:]
                                else:
                                    flush_text = buffer.strip()
                                    buffer = ""
                                token_count = 0

                            if flush_text:
                                send_ts({"type": "status", "stage": "speaking"})
                                try:
                                    wav = run_tts(flush_text)
                                    b64 = base64.b64encode(wav).decode()
                                    send_ts({"type": "audio_chunk", "data": b64})
                                except Exception as e:
                                    log.warning(f"TTS error: {e}")

                            return openvino_genai.StreamingStatus.RUNNING

                        config = build_gen_config(args)
                        try:
                            with gen_lock:
                                ctx["llm_pipe"].generate(transcript, config, streamer=streamer)
                        except Exception as e:
                            send_ts({"type": "error", "message": f"LLM error: {e}"})
                            # Fall through to flush remaining buffer and send audio_done.

                        # Flush any remaining text that didn't end with punctuation.
                        tail = buffer.strip()
                        if tail:
                            try:
                                wav = run_tts(tail)
                                b64 = base64.b64encode(wav).decode()
                                send_ts({"type": "audio_chunk", "data": b64})
                            except Exception as e:
                                log.warning(f"TTS tail error: {e}")

                        full_text = "".join(full_response)
                        send_ts({"type": "llm_done", "text": full_text})
                        send_ts({"type": "audio_done"})
                        # Reset wake word buffers after each conversation turn.
                        # The feature buffer fills with zeros after reset, and
                        # the warmup period prevents false detections while it
                        # repopulates with real audio embeddings.
                        if ctx.get("wakeword"):
                            ctx["wakeword"].reset()
                        send_ts({"type": "status", "stage": "idle"})

                    except Exception as e:
                        log.error(f"voice process error: {e}", exc_info=True)
                        send_ts({"type": "error", "message": str(e)})
                        send_ts({"type": "status", "stage": "idle"})
                    finally:
                        busy.clear()

                threading.Thread(target=process_voice, daemon=True).start()

        except WebSocketDisconnect:
            pass

    uvicorn.run(app, host=args.host, port=args.port)


def main():
    args = parse_args()

    if not os.path.isdir(args.model):
        print(f"Error: LLM model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.whisper):
        print(f"Error: Whisper model directory not found: {args.whisper}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(args.tts_dir):
        print(f"Error: TTS model directory not found: {args.tts_dir}", file=sys.stderr)
        sys.exit(1)

    threading.Thread(target=load_initial, args=(args,), daemon=True).start()
    run_web(args)


if __name__ == "__main__":
    main()
