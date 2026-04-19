# How It Works — A Guide to the AI Voice Chat

This guide explains the AI technologies behind the voice chatbot: what they are, how they work, and how the Python code ties them together. It assumes you're comfortable with Python but may be new to AI/ML concepts.

## Table of contents

1. [The big picture](#the-big-picture)
2. [OpenVINO — the inference engine](#openvino--the-inference-engine)
3. [Wake word detection](#wake-word-detection)
4. [Speech recognition (ASR)](#speech-recognition-asr)
5. [Large language model (LLM)](#large-language-model-llm)
6. [Text-to-speech (TTS)](#text-to-speech-tts)
7. [Streaming — tying it all together](#streaming--tying-it-all-together)

---

## The big picture

The chatbot is a pipeline of four AI models, each handling a different stage of a voice conversation:

1. **Wake word detection** — listens continuously for a trigger phrase (e.g. "Hey Athena")
2. **Speech recognition (ASR)** — converts the user's spoken words into text
3. **Large language model (LLM)** — reads the text, thinks about it, and generates a response
4. **Text-to-speech (TTS)** — converts the response text back into spoken audio

Each model is a neural network — a mathematical function with millions of learned parameters that transforms one kind of data into another. The models don't share any architecture or training data; they're independent components chained together.

All four models run through the same inference engine: OpenVINO.

---

## OpenVINO — the inference engine

### What is an inference engine?

When an AI model is being developed, it goes through two phases:

- **Training** — the model learns its parameters by processing enormous datasets. This requires powerful hardware (clusters of GPUs) and can take weeks.
- **Inference** — the trained model is used to make predictions on new inputs. This is what happens at runtime: you give it audio, it gives you text.

An **inference engine** is the software that executes a trained model efficiently on specific hardware. Think of it like a virtual machine for neural networks — it takes a model file and runs it, handling all the low-level details of memory management, parallelism, and hardware-specific optimizations.

### What is OpenVINO?

OpenVINO (Open Visual Inference and Neural Network Optimization) is Intel's inference engine. It can run models on Intel CPUs, integrated GPUs, and NPUs (Neural Processing Units). The key concepts:

**Model formats.** OpenVINO works with two kinds of model files:
- **ONNX** (.onnx) — an open standard format that most ML frameworks can export to. Our wake word models use this format.
- **OpenVINO IR** (.xml + .bin) — Intel's own optimized format. Our Whisper, LLM, and TTS models use this. Models are typically converted from PyTorch or ONNX to IR using the `optimum-cli` tool.

**Quantization.** Neural network weights are normally stored as 32-bit or 16-bit floating point numbers. **Quantization** compresses them to smaller types like INT8 (8-bit integers) or INT4 (4-bit integers). This reduces memory usage and speeds up inference at the cost of a small accuracy loss. Our LLM uses INT8 quantization — roughly half the memory of FP16 with minimal quality impact.

**Device targeting.** When you load a model in OpenVINO, you specify which hardware device should run it:

```python
import openvino as ov

core = ov.Core()
# Compile for a specific device
model = core.compile_model("model.onnx", "CPU")   # run on CPU
model = core.compile_model("model.onnx", "GPU")   # run on integrated GPU
model = core.compile_model("model.onnx", "NPU")   # run on neural processing unit
```

The same model file can often run on different devices, though not always — some operations aren't supported on every device.

**openvino_genai.** On top of the core OpenVINO runtime, Intel provides a higher-level library called `openvino_genai` that wraps common generative AI patterns:

```python
import openvino_genai

# High-level pipelines that handle tokenization, decoding, etc.
whisper = openvino_genai.WhisperPipeline("/models/whisper-base", "GPU")
llm = openvino_genai.LLMPipeline("/models/qwen3-8b", "GPU")
tts = openvino_genai.Text2SpeechPipeline("/models/speecht5", "CPU")
```

These pipelines handle the boilerplate — loading tokenizers, managing input/output shapes, handling streaming — so you can focus on the application logic.

---

## Wake word detection

### What is wake word detection?

Wake word detection is the problem of continuously monitoring audio and recognizing when a specific phrase is spoken — "Hey Siri", "Alexa", or in our case, "Hey Athena". It's different from general speech recognition because:

- It runs **continuously** (always listening, even when idle)
- It only needs to recognize **one specific phrase**, not arbitrary speech
- It must be **lightweight** — it shouldn't consume significant power or compute while waiting

### The model: openWakeWord

We use [openWakeWord](https://github.com/dscripka/openWakeWord), an open-source framework that breaks detection into three small ONNX models run in sequence:

#### Stage 1: Melspectrogram

Raw audio is just a sequence of amplitude values sampled thousands of times per second. This isn't a great format for recognizing speech — it contains too much raw detail and not enough structure.

A **mel spectrogram** is a visual representation of audio that captures the information humans actually hear. It works by:
1. Slicing the audio into short overlapping windows (a few milliseconds each)
2. Running a Fourier transform on each window to find which frequencies are present
3. Mapping those frequencies onto the **mel scale** — a scale that matches how human hearing perceives pitch (we're more sensitive to differences in low frequencies than high frequencies)

The result is a 2D grid: time on one axis, frequency on the other, with brightness representing energy. If you've ever seen a colorful "waterfall" visualization of audio, that's a spectrogram.

In code, this is a single ONNX model call:

```python
def _compute_melspec(self, audio_int16):
    x = audio_int16.astype(np.float32)[np.newaxis, :]
    result = self._melspec({"input": x})
    spec = np.squeeze(list(result.values())[0])
    return spec / 10.0 + 2.0   # normalize to a useful range
```

The output is a matrix of shape `[frames, 32]` — each row is one time step, each column is one of 32 mel frequency bins.

#### Stage 2: Embedding CNN

The mel spectrogram is informative but high-dimensional. The embedding model is a **CNN (Convolutional Neural Network)** — a type of neural network originally designed for image recognition — that compresses a window of 76 mel frames down to a single 96-dimensional vector called an **embedding**.

An embedding is a compact numerical fingerprint of the audio. Similar-sounding audio produces similar embeddings; different audio produces different ones. The CNN was pre-trained on large amounts of speech data to learn which audio features matter for distinguishing words.

```python
def _compute_embedding(self, mel_window):
    x = mel_window[np.newaxis, :, :, np.newaxis].astype(np.float32)
    result = self._embed({"input_1": x})
    return np.squeeze(list(result.values())[0])   # 96-dimensional vector
```

#### Stage 3: Classifier

The classifier is a small fully-connected neural network that looks at the last 16 embeddings (about 1.3 seconds of audio) and outputs a single score between 0 and 1: how confident it is that the wake word was just spoken.

Each wake word has its own classifier — a tiny ONNX file, roughly 200KB. The melspectrogram and embedding models are shared across all wake words; only the final classifier is specific to each phrase.

```python
def _classify(self):
    compiled, input_name = self._classifiers[self._active]
    x = self._feature_buffer[np.newaxis, :, :].astype(np.float32)
    result = compiled({input_name: x})
    return float(np.squeeze(list(result.values())[0]))   # 0.0 to 1.0
```

### Avoiding false detections

A single frame scoring above the threshold isn't enough — ambient noise can cause brief spikes. We require **three consecutive frames** above the threshold (about 240ms of sustained high confidence) before declaring a detection:

```python
score = self._classify()
if score > self.threshold:
    self._hit_count += 1
    if self._hit_count >= self._consecutive:
        self._hit_count = 0
        return score   # detected!
else:
    self._hit_count = 0
```

There's a subtler problem too. After each detection, the feature buffer is reset to zeros. But the classifier was trained on real audio embeddings, not zeros — and it turns out that an all-zero feature buffer produces artificially high confidence scores, causing a cascade of false detections immediately after each real detection.

The fix is a **warmup period**: after a reset, the classifier is suppressed for 16 frames (about 1.3 seconds) while the buffer fills with real embeddings:

```python
if self._warmup > 0:
    self._warmup = max(0, self._warmup - n_new_embeddings)
    if self._warmup > 0:
        return None   # don't classify yet
```

---

## Speech recognition (ASR)

### What is ASR?

**ASR (Automatic Speech Recognition)** is the task of converting spoken audio into text. It's what happens when you dictate a text message or ask a voice assistant a question.

### The model: Whisper

[Whisper](https://github.com/openai/whisper) is OpenAI's speech recognition model. It's an **encoder-decoder transformer** — a neural network architecture with two halves:

- The **encoder** processes the input audio and builds an internal representation of what was said. It takes in a mel spectrogram (the same kind of frequency representation used in wake word detection) and produces a sequence of hidden state vectors that capture the meaning of the audio.
- The **decoder** reads the encoder's output and generates text one word (actually one **token**) at a time. A token is a piece of a word — common words like "the" are single tokens, while longer words get split into multiple tokens. The decoder predicts the next token, appends it, then predicts the next, and so on until it produces an end-of-sequence marker.

Whisper was trained on 680,000 hours of audio scraped from the internet, which gives it broad language coverage and noise robustness. We use pre-exported OpenVINO FP16 models from HuggingFace, so we don't need to convert anything ourselves.

In the code, ASR is a single function call:

```python
def run_asr(audio_f32):
    result = ctx["whisper_pipe"].generate(audio_f32.tolist())
    return result.texts[0].strip()
```

The `WhisperPipeline` handles all the internals: computing the mel spectrogram, running the encoder, running the decoder's autoregressive loop, and converting token IDs back to text.

### Pre-ASR filtering

Before calling Whisper, we check if the audio is actually silence. Computing a full ASR pass on silence is wasteful, and Whisper has a tendency to hallucinate on quiet input.

**RMS (Root Mean Square)** is a measure of audio loudness. For audio normalized to the range [-1, 1], an RMS below 0.01 (about -40 dBFS) is effectively silence:

```python
rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
if rms < 0.01:
    return   # skip ASR — it's silence
```

### Post-ASR filtering

Even with the silence gate, Whisper sometimes hallucinates — producing phantom transcriptions when the input is borderline quiet. These hallucinations are artifacts of its YouTube training data and tend to be the same phrases over and over:

```python
WHISPER_HALLUCINATIONS = {
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    # ...
}

transcript_clean = re.sub(r"[^\w\s]", "", transcript).strip().lower()
if transcript_clean in WHISPER_HALLUCINATIONS:
    return   # drop it
```

We strip punctuation and lowercase before checking because Whisper's exact capitalization and punctuation varies from run to run.

---

## Large language model (LLM)

### What is an LLM?

A **Large Language Model** is a neural network trained to predict the next token in a sequence of text. Given "The capital of France is", it predicts "Paris". By repeatedly predicting the next token and appending it to the sequence, the model generates coherent multi-sentence responses.

LLMs are **transformers** — the same architecture family as Whisper's decoder, but much larger. The "large" in LLM refers to both the parameter count (billions) and the training data (trillions of tokens of text from books, websites, and other sources).

### The model: Qwen3-8B

We use Qwen3-8B, an 8-billion parameter model from Alibaba's Qwen team. It's quantized to INT8, meaning each of its 8 billion parameters is stored as an 8-bit integer rather than a 16-bit float. This halves the memory footprint (roughly 8GB instead of 16GB) with minimal impact on response quality.

Loading the model:

```python
ctx["llm_pipe"] = openvino_genai.LLMPipeline("/models/qwen3-8b", "GPU")
ctx["llm_pipe"].start_chat(system_prompt)
```

`start_chat()` initializes a conversation session. The **system prompt** tells the model how to behave — in our case, we instruct it to give brief, spoken-style answers without markdown formatting. The model retains conversation context across turns, so it remembers what was said earlier.

### How text generation works

The model generates text one token at a time through a process called **autoregressive decoding**:

1. The input text is converted to token IDs by a **tokenizer** (a lookup table that maps text fragments to numbers)
2. The model processes all tokens and predicts a probability distribution over the vocabulary for the next token
3. A token is sampled from that distribution (not always the most probable one — randomness makes the output more natural)
4. The sampled token is appended to the sequence, and step 2 repeats

Several parameters control the sampling:

- **Temperature** (0.7) — how random the sampling is. Lower = more predictable, higher = more creative.
- **Top-p** (0.9) — only sample from the smallest set of tokens whose cumulative probability exceeds 0.9. This eliminates unlikely tokens while preserving variety.
- **Top-k** (50) — only consider the 50 most probable tokens at each step.
- **Repetition penalty** (1.1) — penalize tokens that have already appeared, reducing repetitive output.

```python
config = openvino_genai.GenerationConfig()
config.max_new_tokens = 512
config.temperature = 0.7
config.top_p = 0.9
config.top_k = 50
config.repetition_penalty = 1.1
```

### Streaming with a callback

Rather than waiting for the full response, we pass a **streamer callback** that receives each token as it's generated:

```python
def streamer(subword):
    # subword is a string fragment, e.g. "The", " capital", " of"
    send_to_client({"type": "llm_token", "text": subword})
    return openvino_genai.StreamingStatus.RUNNING

ctx["llm_pipe"].generate(prompt, config, streamer=streamer)
```

The callback returns `RUNNING` to continue generation or `STOP` to halt it early. We use `STOP` when we detect runaway repetition — a symptom of the model getting stuck in a loop:

```python
# If the same token repeats 8 times in a row, stop
if all(t == tok for t in full_response[-8:]):
    return openvino_genai.StreamingStatus.STOP
```

### Thinking mode

Qwen3 supports a **thinking mode** where the model emits chain-of-thought reasoning inside `<think>...</think>` tags before giving its answer. This can improve response quality for complex questions but adds latency — the model generates many extra tokens that the user doesn't need to hear.

We control this with the `/no_think` directive in the system prompt:

```python
def effective_system_prompt(args):
    if ctx["thinking_enabled"]:
        return args.system                   # thinking on
    return "/no_think " + args.system        # thinking off (default)
```

When thinking is enabled, the streamer detects `<think>` tokens and sends them to the UI (displayed in italics) but never routes them to TTS.

---

## Text-to-speech (TTS)

### What is TTS?

**Text-to-speech** converts written text into spoken audio. It's the inverse of ASR: text goes in, a waveform comes out.

### The model: SpeechT5

[SpeechT5](https://huggingface.co/microsoft/speecht5_tts) is a sequence-to-sequence transformer from Microsoft. Like Whisper, it has an encoder-decoder structure, but running in the opposite direction:

1. The **encoder** processes the input text (after tokenization) and builds an internal representation
2. The **decoder** generates a mel spectrogram — the same frequency-over-time representation used in wake word detection and Whisper, but now we're creating it from scratch instead of analyzing it
3. A **vocoder** (HiFi-GAN) converts the mel spectrogram into a raw audio waveform

The mel spectrogram is the common language between speech and text in many audio AI systems. It captures the essential structure of speech (pitch, rhythm, phonemes) without the fine-grained detail of a raw waveform. The vocoder's job is to fill in that detail, producing natural-sounding audio from the spectrogram sketch.

### Speaker embeddings

SpeechT5 is a **multi-speaker** model — it can produce different voices depending on a **speaker embedding** you provide. A speaker embedding is a 512-dimensional vector that encodes the characteristics of a particular voice: pitch range, timbre, speaking style.

We extract our speaker embedding from the CMU Arctic x-vectors dataset, a collection of pre-computed voice fingerprints from recorded speakers. Speaker 7306 is an American English female voice:

```python
# At image build time:
embedding = np.array(dataset[7306]["xvector"], dtype=np.float32)
embedding.tofile("speaker.bin")

# At runtime:
spk_np = np.fromfile("speaker.bin", dtype=np.float32).reshape(1, 512)
speaker_embedding = ov.Tensor(spk_np)
```

The embedding is passed to the model alongside the text:

```python
def run_tts(text):
    result = ctx["tts_pipe"].generate(text, speaker_embedding)
    speech = result.speeches[0]
    # ... convert to WAV
```

### From float samples to WAV

The TTS model outputs audio as float32 samples in the range [-1, 1]. To produce a WAV file that browsers can play, we convert to 16-bit integers and wrap in the WAV container format:

```python
buf = io.BytesIO()
with wave.open(buf, "wb") as wf:
    wf.setnchannels(1)          # mono
    wf.setsampwidth(2)          # 16-bit (2 bytes per sample)
    wf.setframerate(16000)      # 16kHz sample rate
    audio_int16 = (raw * 32767).clip(-32768, 32767).astype(np.int16)
    wf.writeframes(audio_int16.tobytes())
```

The `* 32767` scales the float [-1, 1] range to the int16 [-32768, 32767] range. `.clip()` ensures we don't overflow if the model produces values slightly outside [-1, 1].

---

## Streaming — tying it all together

The individual components are straightforward. The interesting engineering is in how they're connected to minimize latency — the delay between the user finishing their question and hearing the first word of the response.

### The latency problem

Without streaming, the pipeline would be strictly sequential:

1. Record audio (1-5 seconds)
2. Transcribe with Whisper (~0.5 seconds)
3. Generate full LLM response (~3-10 seconds)
4. Synthesize all audio with TTS (~1-2 seconds)
5. Play audio

The user would wait 5-17 seconds of silence before hearing anything. That's not a conversation — it's a voicemail system.

### Clause-boundary streaming

The solution is to overlap LLM generation and TTS. Instead of waiting for the full response, we flush text to TTS every time we hit a clause boundary — a period, exclamation mark, question mark, comma, or semicolon followed by whitespace:

```python
CLAUSE_END_RE = re.compile(r'(?<=[.!?,;])\s+')
```

Inside the LLM's streamer callback, we accumulate tokens into a buffer. When the regex matches, we split off the completed clause and send it to TTS immediately:

```python
buffer += subword
m = CLAUSE_END_RE.search(buffer)
if m:
    clause = buffer[:m.start() + 1].strip()
    buffer = buffer[m.end():]          # keep the remainder
    wav = run_tts(clause)              # synthesize this clause now
    send_audio_to_browser(wav)
```

This means the user hears "The capital of France is Paris." while the LLM is still generating the second sentence. The perceived latency drops from the full generation time to just the time to produce the first clause — typically 1-2 seconds.

As a fallback, if the model produces 25 tokens without any punctuation (a long run-on sentence), we force a flush at the nearest word boundary:

```python
elif token_count >= 25:
    last_space = buffer.rfind(' ')
    if last_space > 0:
        clause = buffer[:last_space].strip()
        buffer = buffer[last_space + 1:]
```

### Gapless audio playback

Each clause produces a separate WAV audio chunk. The browser receives these chunks over a WebSocket and needs to play them back-to-back without gaps or pops between them.

The web UI uses the Web Audio API's `AudioContext` to schedule chunks precisely. Each chunk is decoded into an `AudioBuffer`, and a `BufferSourceNode` is created and scheduled to start at exactly the moment the previous chunk ends:

```javascript
const startAt = Math.max(ctx.currentTime, nextPlayAt);
source.start(startAt);
nextPlayAt = startAt + buf.duration;
```

By tracking `nextPlayAt`, each new chunk is queued to start at the exact sample where the last one ended, producing seamless continuous audio from discrete chunks.

### The full flow

Putting it all together, a single voice interaction looks like this:

1. The browser mic continuously sends 80ms audio chunks to the server for wake word detection
2. The NPU processes each chunk through the embedding CNN and classifier
3. Wake word detected → the browser starts recording
4. User speaks, then stops (detected by silence timeout)
5. Recording is sent to the server as WebM/Opus audio
6. Server decodes to PCM with ffmpeg, checks for silence, runs Whisper
7. Transcript is sent to the LLM
8. LLM streams tokens; each clause is immediately synthesized by TTS
9. WAV chunks are sent to the browser over WebSocket
10. Browser plays chunks gaplessly through AudioContext
11. After the last chunk, the system returns to idle and resumes wake word listening
