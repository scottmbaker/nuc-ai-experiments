"""
Wake word detection using openWakeWord ONNX models on Intel NPU.

Pipeline:
  1. Melspectrogram (CPU) — ONNX reflect-pad unsupported on NPU
  2. Embedding CNN (NPU) — batch=1, [1,76,32,1] → [1,96]
  3. Wake word classifier (NPU) — [1,16,96] → [1,1] score

All classifier .onnx files in model_dir are loaded at startup (shared
melspec + embedding, one compiled classifier per wake word). Switch the
active wake word at runtime via the `active` property.

Audio must be 16-bit 16kHz PCM, fed in chunks via process().
"""

import logging
import os
from collections import deque
from typing import List, Optional

import numpy as np
import openvino as ov

log = logging.getLogger("wakeword")

# ONNX files that are shared infrastructure, not wake word classifiers
_SHARED_MODELS = {"melspectrogram.onnx", "embedding_model.onnx"}


class WakeWordDetector:
    SAMPLE_RATE    = 16000
    CHUNK_SAMPLES  = 1280   # 80ms at 16kHz — openWakeWord's frame size
    MEL_BINS       = 32
    WINDOW_SIZE    = 76     # mel frames needed for one embedding
    EMBED_DIM      = 96
    FEATURE_FRAMES = 16     # classifier input: last 16 embeddings

    def __init__(self, model_dir: str, wake_word: str = None,
                 threshold: float = 0.5, npu_device: str = "NPU",
                 consecutive: int = 3):
        """
        Args:
            model_dir:   Directory containing melspectrogram.onnx,
                         embedding_model.onnx, and *.onnx classifier files.
            wake_word:   Name of the classifier to activate initially.
                         If None or not found, defaults to the first discovered.
            threshold:   Detection threshold (0–1).
            npu_device:  Device for embedding + classifier (default: NPU).
            consecutive: Number of consecutive frames that must exceed threshold
                         before a detection is declared (default: 3 = ~240ms).
                         Reduces false positives from brief ambient audio spikes.
        """
        self.threshold = threshold
        self._consecutive = consecutive
        core = ov.Core()

        # 1. Melspectrogram on CPU (reflect-pad op unsupported on NPU)
        melspec_path = os.path.join(model_dir, "melspectrogram.onnx")
        self._melspec = core.compile_model(melspec_path, "CPU")
        log.info("Melspectrogram compiled on CPU")

        # 2. Embedding model on NPU (reshaped to static batch=1)
        embed_path = os.path.join(model_dir, "embedding_model.onnx")
        embed_model = core.read_model(embed_path)
        embed_model.reshape({"input_1": [1, self.WINDOW_SIZE, self.MEL_BINS, 1]})
        self._embed = core.compile_model(embed_model, npu_device)
        log.info(f"Embedding model compiled on {npu_device}")

        # 3. Discover and load all classifier .onnx files in model_dir
        # Name is derived by stripping _v0.1.onnx or .onnx suffix.
        self._classifiers: dict = {}   # name -> (compiled_model, input_name)
        for fname in sorted(os.listdir(model_dir)):
            if fname in _SHARED_MODELS or not fname.endswith(".onnx"):
                continue
            name = fname.replace("_v0.1.onnx", "").replace(".onnx", "")
            try:
                ww_model = core.read_model(os.path.join(model_dir, fname))
                for inp in ww_model.inputs:
                    if inp.partial_shape.is_dynamic:
                        shape = [d.get_length() if not d.is_dynamic else 1
                                 for d in inp.partial_shape]
                        ww_model.reshape({inp.any_name: shape})
                compiled = core.compile_model(ww_model, npu_device)
                input_name = ww_model.inputs[0].any_name
                self._classifiers[name] = (compiled, input_name)
                log.info(f"Wake word '{name}' compiled on {npu_device}")
            except Exception as e:
                log.warning(f"Failed to load wake word '{name}': {e}")

        if not self._classifiers:
            raise RuntimeError(f"No wake word classifiers found in {model_dir}")

        # Set initial active wake word
        if wake_word and wake_word in self._classifiers:
            self._active = wake_word
        else:
            self._active = next(iter(self._classifiers))
            if wake_word:
                log.warning(
                    f"Wake word '{wake_word}' not found, defaulting to '{self._active}'"
                )

        self._reset_buffers()

    # ── Public properties ────────────────────────────────────────

    @property
    def active(self) -> str:
        """Name of the currently active wake word classifier."""
        return self._active

    @active.setter
    def active(self, name: str):
        """Switch the active classifier. Resets all buffers."""
        if name not in self._classifiers:
            raise ValueError(
                f"Unknown wake word '{name}'. Available: {self.available}"
            )
        self._active = name
        self._reset_buffers()
        log.info(f"Active wake word switched to '{name}'")

    @property
    def available(self) -> List[str]:
        """List of all loaded wake word names."""
        return list(self._classifiers.keys())

    # ── Internal helpers ─────────────────────────────────────────

    def _reset_buffers(self):
        self._raw_buffer: deque = deque(maxlen=self.SAMPLE_RATE * 10)
        # Ones (not zeros) for the mel buffer: the mel model output for silence
        # is a small positive value (~2.0 after normalization), so ones avoid
        # the zero-initialization artifacts that make the embedding CNN produce
        # garbage outputs for the first window after a reset.
        self._mel_buffer = np.ones(
            (self.WINDOW_SIZE, self.MEL_BINS), dtype=np.float32)
        self._feature_buffer = np.zeros(
            (self.FEATURE_FRAMES, self.EMBED_DIM), dtype=np.float32)
        self._accumulated = 0
        self._remainder = np.empty(0, dtype=np.int16)
        self._hit_count = 0
        self._warmup = self.FEATURE_FRAMES  # skip classifier until buffer has real embeddings

    def _compute_melspec(self, audio_int16: np.ndarray) -> np.ndarray:
        x = audio_int16.astype(np.float32)[np.newaxis, :]
        result = self._melspec({"input": x})
        spec = np.squeeze(list(result.values())[0])
        return spec / 10.0 + 2.0

    def _compute_embedding(self, mel_window: np.ndarray) -> np.ndarray:
        x = mel_window[np.newaxis, :, :, np.newaxis].astype(np.float32)
        result = self._embed({"input_1": x})
        return np.squeeze(list(result.values())[0])

    def _classify(self) -> float:
        compiled, input_name = self._classifiers[self._active]
        x = self._feature_buffer[np.newaxis, :, :].astype(np.float32)
        result = compiled({input_name: x})
        return float(np.squeeze(list(result.values())[0]))

    # ── Public API ───────────────────────────────────────────────

    def process(self, audio_int16: np.ndarray) -> Optional[float]:
        """
        Feed an audio chunk and check for the active wake word.

        Args:
            audio_int16: 16-bit PCM at 16kHz (any length).

        Returns:
            Detection score (> threshold) if wake word detected, else None.
        """
        if self._remainder.shape[0] > 0:
            audio_int16 = np.concatenate([self._remainder, audio_int16])
            self._remainder = np.empty(0, dtype=np.int16)

        self._raw_buffer.extend(audio_int16.tolist())
        self._accumulated += len(audio_int16)

        if self._accumulated < self.CHUNK_SAMPLES:
            return None

        n_full = (self._accumulated // self.CHUNK_SAMPLES) * self.CHUNK_SAMPLES
        if self._accumulated > n_full:
            leftover = self._accumulated - n_full
            self._remainder = audio_int16[-leftover:]
            self._accumulated = n_full
        else:
            self._remainder = np.empty(0, dtype=np.int16)

        # Include 3 mel hops (3 × 160 = 480 samples) of history before the
        # current accumulated frames. The STFT window overlaps adjacent frames,
        # so without this context the first few mel frames of each chunk would
        # be computed with zero-padding at the left edge rather than real audio.
        raw = np.array(
            list(self._raw_buffer)[-(self._accumulated + 160 * 3):],
            dtype=np.int16)
        new_mel = self._compute_melspec(raw)

        self._mel_buffer = np.vstack([self._mel_buffer, new_mel])
        # Cap mel history at ~10 seconds: mel hop = 160 samples = 10ms, so
        # the mel model produces ~97 frames per 80ms chunk (after STFT windowing
        # the effective rate is slightly under 100 fps). 10 × 97 ≈ 970 frames.
        max_mel = 10 * 97
        if self._mel_buffer.shape[0] > max_mel:
            self._mel_buffer = self._mel_buffer[-max_mel:]

        # Compute one embedding per new 80ms chunk, using a sliding window of
        # WINDOW_SIZE=76 mel frames. Each chunk adds 8 mel frames (1280/160),
        # so consecutive windows are spaced 8 frames apart. For multiple chunks
        # in one call (n_new_embeddings > 1), we work backwards from the latest
        # frame: the last embedding window ends at mel_buffer[-1], the second-
        # to-last ends 8 frames earlier, and so on.
        n_new_embeddings = self._accumulated // self.CHUNK_SAMPLES
        for i in range(n_new_embeddings):
            offset = (
                -self.WINDOW_SIZE
                + self._mel_buffer.shape[0]
                - 8 * (n_new_embeddings - 1 - i)
            )
            end = offset + self.WINDOW_SIZE
            if end > self._mel_buffer.shape[0]:
                end = self._mel_buffer.shape[0]
            window = self._mel_buffer[end - self.WINDOW_SIZE:end]
            if window.shape[0] == self.WINDOW_SIZE:
                emb = self._compute_embedding(window)
                self._feature_buffer = np.vstack(
                    [self._feature_buffer[1:], emb[np.newaxis, :]])

        self._accumulated = 0

        # Don't classify until the feature buffer is filled with real embeddings.
        # Zeros from _reset_buffers() produce artificially high classifier scores.
        if self._warmup > 0:
            self._warmup = max(0, self._warmup - n_new_embeddings)
            if self._warmup > 0:
                return None

        score = self._classify()
        if score > self.threshold:
            self._hit_count += 1
            if self._hit_count >= self._consecutive:
                self._hit_count = 0
                return score
        else:
            self._hit_count = 0
        return None

    def reset(self):
        """Reset all audio/feature buffers (e.g. after a detection)."""
        self._reset_buffers()
