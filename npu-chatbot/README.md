# NPU Chatbot

Interactive streaming LLM chatbot running on Intel NPU via OpenVINO GenAI.

## How it works

The chatbot uses `openvino_genai.LLMPipeline` to run the full model on a single
device. All components (embedding, attention, decoding) run on whichever device
you choose.

- **NPU mode** (default): The entire model runs on the NPU. First run is slow
  (~minutes) because the NPU compiler must compile the model graph into NPU blobs.
  Compiled blobs are cached in `<model_dir>/npu_blobs/` for fast subsequent loads.
- **CPU mode** (`--device CPU`): Falls back to CPU inference.

### NPU constraints

The NPU has a fixed context window determined at compile time:

| Parameter            | Default | Purpose                                |
|----------------------|---------|----------------------------------------|
| `--max-prompt-len`   | 2048    | Max input tokens the NPU can process   |
| `--min-response-len` | 512     | Tokens reserved for model responses    |

Total context = max-prompt-len + min-response-len. If conversation exceeds
this, you'll get an error -- type `reset` to start fresh.

## Models

Available models are defined in `models.conf`. List them with:

```bash
make models
```

All models are pre-exported INT4 OpenVINO models downloaded from HuggingFace
at build time. No local export or quantization is needed.

### General models

| Tier         | Examples                                 | NPU | CPU |
|--------------|------------------------------------------|-----|-----|
| Small (<2B)  | TinyLlama, Qwen3-0.6B, DeepSeek-1.5B     | Yes | Yes |
| Medium (2-4B)| Phi-3 Mini, Phi-4 Mini, Qwen3-4B         | Yes | Yes |
| Large (7-8B) | Qwen3-8B, DeepSeek-7B, Mistral-7B        | Yes | Yes |
| XL (14B)     | Qwen2.5-14B                              | Yes | Yes |

### Coding models

Qwen2.5-Coder (0.5B to 14B) and StarCoder2 (7B, 15B) are available. The Coder
models currently don't compile on NPU due to a compiler bug -- use `DEVICE=CPU`.

StarCoder2 is a base model (not instruct-tuned) and has no chat template. It
will error if used with the chatbot. It is included for code completion use
cases that may be added later.

## Build

```bash
# Build a specific model
sudo make build MODEL=tinyllama
sudo make build MODEL=qwen3-4b
sudo make build MODEL=qwen25-14b

# Build all models
sudo make build-all
```

## Deploy

```bash
# Deploy on NPU (default)
make deploy MODEL=tinyllama

# Deploy on CPU
make deploy MODEL=qwen25-coder-7b DEVICE=CPU

# Attach to the chatbot
make attach

# Type 'quit' to exit, 'reset' for new conversation
```

The pod requests `npu.intel.com/accel: 1` for NPU access and uses a 4Gi
`/dev/shm` tmpfs mount for shared memory.

## Other commands

```bash
make help       # Show all targets
make models     # List available models
make attach     # Connect to running chatbot
make logs       # Tail pod logs
make status     # Show pod status, NPU resources, built images
make undeploy   # Remove the chatbot pod
make nuke       # Clean up orphaned containerd state
make fix-k3s    # Increase timeouts and lower eviction thresholds
make clean      # Remove all npu-chatbot images
```

## Files

| File                 | Purpose                                       |
|----------------------|-----------------------------------------------|
| npu-chatbot.py       | Interactive chatbot with streaming output      |
| Dockerfile           | Container build (downloads pre-exported models)|
| models.conf          | Model registry (name, HF ID, model dir)       |
| npu-chatbot-pod.yaml | K8s pod spec with NPU device request           |
| Makefile             | Build, deploy, and management targets          |
