# Image Generator

Web-based image generation service using Stable Diffusion models, with support
for CPU, GPU, and NPU inference via OpenVINO.

## Models

Available models are defined in `models.conf`. List them with `make models`.

| Name | Base | Steps | Guidance | Quality |
|------|------|-------|----------|---------|
| lcm-dreamshaper | SD 1.5 + LCM | 4 | 1.0 | Decent, fast |
| sd15 | Stable Diffusion 1.5 | 20 | 7.5 | Good, follows prompts well |
| sdxl-turbo | SDXL Turbo | 4 | 0.0 | Better quality, fast |
| sdxl-base | SDXL 1.0 | 20 | 7.5 | Unpredictable (INT8 artifacts) |
| playground-v25 | Playground v2.5 | 50 | 3.0 | Unpredictable (INT8 artifacts) |

## Architecture: Hybrid Device Inference

Stable Diffusion consists of three models that run in sequence:

1. **Text Encoder** — Converts the text prompt into embeddings. Runs once per
   generation. Lightweight.
2. **UNet** — The denoising backbone. Runs once per inference step (4-50 times
   depending on model). This is 95%+ of the total compute.
3. **VAE Decoder** — Converts the denoised latents into a visible image. Runs
   once at the end. Lightweight.

Because the UNet dominates the workload, only the UNet is offloaded to the
accelerator (NPU or GPU). The text encoder and VAE decoder stay on CPU — moving
them to an accelerator would add device-transfer overhead for negligible compute
savings. This hybrid approach is the standard for diffusion models on OpenVINO,
and is what Intel recommends in their own examples.

### Device compilation at startup

At startup, the server compiles the UNet for each available device:

1. **CPU** — Compiled with dynamic shapes. Supports any image resolution.
2. **GPU** — Compiled with dynamic shapes and FP32 inference precision.
   Supports any image resolution. FP32 is required because INT8 models produce
   black images at the GPU's default FP16 precision (dequantization overflow).
   Requires Intel GPU compute runtime from the kobuk-team PPA for Panther Lake.
3. **NPU** — Compiled with static shapes (512x512). The NPU compiler requires
   all tensor dimensions to be fixed. Images are locked to 512x512 on NPU.

The text encoder and VAE decoder are always compiled on CPU with dynamic shapes
before the UNet reshape, so they work regardless of which device runs the UNet.

The web UI provides a device dropdown to switch between devices per request.

### Batch size and classifier-free guidance

Models that use classifier-free guidance (guidance_scale > 1) send batch=2 to
the UNet (prompt + negative prompt). Models without guidance (LCM, SDXL Turbo)
send batch=1. The NPU static reshape accounts for this based on the model's
default guidance scale. The `pipe.reshape()` call auto-doubles the batch for
non-LCM models; for models without guidance, the UNet inputs are manually
reshaped back to batch=1 after.

## Build

The container is built from bare Ubuntu 24.04 with OpenVINO installed via pip
and the Intel GPU compute runtime from the kobuk-team PPA. This avoids version
mismatches between the OpenVINO GPU plugin and the compute runtime that occur
when using the official OpenVINO Docker images.

```bash
# List models
make models

# Build a specific model
sudo make build MODEL=sd15
sudo make build MODEL=sdxl-turbo

# Build all models
sudo make build-all
```

## Deploy

Deployment uses a Helm chart. The `DEVICES` parameter controls which hardware
is requested via K8s device plugins.

```bash
# Deploy with NPU only (default)
make deploy MODEL=sd15 DEVICES=npu

# Deploy with NPU and GPU
make deploy MODEL=sdxl-turbo DEVICES=npu,gpu

# Deploy CPU only (no device plugins)
make deploy MODEL=sd15 DEVICES=cpu

# Undeploy
make undeploy
```

The web UI is accessible at `http://<node-ip>:30080`.

## Test

```bash
# Test all models on all devices
make test

# Test specific models
make test TEST_MODELS="sd15 sdxl-turbo"

# Custom prompt
make test TEST_PROMPT="a mountain landscape at sunset"
```

This builds, deploys, and generates on CPU/GPU/NPU for each model, printing
a results table with execution times.

## Other commands

```bash
make help       # Show all targets
make models     # List available models
make logs       # Tail pod logs
make status     # Show pod status and built images
make nuke       # Clean up orphaned containerd state
make clean      # Remove all images
```

## Files

| File / Directory    | Purpose                                        |
|---------------------|------------------------------------------------|
| imagegen.py         | FastAPI server, multi-device UNet compilation  |
| static/index.html   | Web UI with device selector and history        |
| Dockerfile          | Ubuntu 24.04 + pip OpenVINO + PPA GPU runtime  |
| models.conf         | Model registry (name, HF ID, pipeline, defaults) |
| chart/              | Helm chart for K8s deployment                  |
| Makefile            | Build, deploy, test, and management targets    |
