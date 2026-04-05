#!/usr/bin/env python3

import argparse
import base64
import io
import logging
import threading
import time
from pathlib import Path

import openvino as ov
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from optimum.intel.openvino import (
    OVLatentConsistencyModelPipeline,
    OVStableDiffusionPipeline,
    OVStableDiffusionXLPipeline,
)
import uvicorn

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("imagegen")

PIPELINE_CLASSES = {
    "OVLatentConsistencyModelPipeline": OVLatentConsistencyModelPipeline,
    "OVStableDiffusionPipeline": OVStableDiffusionPipeline,
    "OVStableDiffusionXLPipeline": OVStableDiffusionXLPipeline,
}

app = FastAPI()
pipe = None
unet_compiled = {}
available_devices = []
pipeline_class_name = ""
model_defaults = {"steps": 4, "guidance_scale": 1.0}
loading_status = "starting"  # "starting", "loading", "compiling:<device>", "ready", "error:<msg>"


class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    device: str = "CPU"
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    steps: int = Field(default=4, ge=1, le=50)
    guidance_scale: float = Field(default=1.0, ge=0.0, le=20.0)
    seed: int = Field(default=-1, ge=-1)


class GenerateResponse(BaseModel):
    image_base64: str
    elapsed_seconds: float
    device: str


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("/app/static/index.html").read_text()


@app.get("/status")
async def status():
    ready = loading_status == "ready"
    return {"ready": ready, "status": loading_status}


@app.get("/devices")
async def devices():
    default = "NPU" if "NPU" in available_devices else "CPU"
    return {"devices": available_devices, "default": default}


@app.get("/model-info")
async def model_info():
    return {
        "pipeline_class": pipeline_class_name,
        "defaults": model_defaults,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if loading_status != "ready":
        raise HTTPException(503, f"Server is still loading: {loading_status}")

    import torch

    device = req.device.upper()
    if device not in unet_compiled:
        raise HTTPException(400, f"Device {device} not available. Available: {available_devices}")
    if device == "NPU" and (req.width != 512 or req.height != 512):
        raise HTTPException(400, "NPU requires 512x512 (static shapes)")


    log.info(f"Generating on {device}: prompt={req.prompt!r}, {req.width}x{req.height}, "
             f"steps={req.steps}, guidance={req.guidance_scale}, seed={req.seed}")

    pipe.unet.request = unet_compiled[device]

    generator = None
    if req.seed >= 0:
        generator = torch.Generator().manual_seed(req.seed)

    t0 = time.time()
    result = pipe(
        prompt=req.prompt,
        negative_prompt=req.negative_prompt or None,
        width=req.width,
        height=req.height,
        num_inference_steps=req.steps,
        guidance_scale=req.guidance_scale,
        generator=generator,
    )
    elapsed = time.time() - t0

    image = result.images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    log.info(f"Generated in {elapsed:.1f}s on {device}")
    return GenerateResponse(image_base64=b64, elapsed_seconds=round(elapsed, 2), device=device)


def load_and_compile(model_path):
    global pipe, unet_compiled, available_devices, pipeline_class_name, loading_status

    try:
        # Determine pipeline class
        class_file = Path(model_path) / "pipeline_class.txt"
        if class_file.exists():
            pipeline_class_name = class_file.read_text().strip()
        else:
            pipeline_class_name = "OVLatentConsistencyModelPipeline"
            log.warning(f"No pipeline_class.txt found, defaulting to {pipeline_class_name}")

        # Read model defaults
        steps_file = Path(model_path) / "default_steps.txt"
        guidance_file = Path(model_path) / "default_guidance.txt"
        if steps_file.exists():
            model_defaults["steps"] = int(steps_file.read_text().strip())
        if guidance_file.exists():
            model_defaults["guidance_scale"] = float(guidance_file.read_text().strip())
        log.info(f"Model defaults: steps={model_defaults['steps']}, guidance={model_defaults['guidance_scale']}")

        PipelineClass = PIPELINE_CLASSES.get(pipeline_class_name)
        if not PipelineClass:
            loading_status = f"error: unknown pipeline class {pipeline_class_name}"
            return

        loading_status = "loading model"
        log.info(f"Loading model from {model_path} ({pipeline_class_name})...")
        pipe = PipelineClass.from_pretrained(model_path, device="CPU", compile=False)

        core = ov.Core()

        # CPU and GPU: compile with dynamic shapes (before reshape)
        loading_status = "compiling for CPU"
        log.info("Compiling all components for CPU (dynamic shapes)...")
        pipe.text_encoder.compile()
        pipe.vae_decoder.compile()
        if hasattr(pipe, 'vae_encoder') and pipe.vae_encoder is not None:
            pipe.vae_encoder.compile()
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            pipe.text_encoder_2.compile()
        unet_compiled["CPU"] = core.compile_model(pipe.unet.model, "CPU")

        # GPU: compile with dynamic shapes (before static reshape for NPU)
        if "GPU" in set(core.available_devices):
            loading_status = "compiling UNet for GPU"
            log.info("Compiling UNet for GPU (dynamic shapes)...")
            try:
                gpu_config = {
                    "INFERENCE_PRECISION_HINT": "f32",
                    "GPU_ENABLE_LARGE_ALLOCATIONS": "YES",
                }
                unet_compiled["GPU"] = core.compile_model(pipe.unet.model, "GPU", gpu_config)
                log.info("  GPU: OK")
            except Exception as e:
                log.warning(f"  GPU: failed ({e})")

        # Save compiled text encoder/VAE requests before reshape destroys them
        saved_requests = {
            "text_encoder": pipe.text_encoder.request,
            "vae_decoder": pipe.vae_decoder.request,
        }
        if hasattr(pipe, 'vae_encoder') and pipe.vae_encoder is not None:
            saved_requests["vae_encoder"] = pipe.vae_encoder.request
        if hasattr(pipe, 'text_encoder_2') and pipe.text_encoder_2 is not None:
            saved_requests["text_encoder_2"] = pipe.text_encoder_2.request

        # NPU: reshape to static shapes and compile
        uses_cfg = model_defaults["guidance_scale"] > 1.0
        if "NPU" in set(core.available_devices):
            log.info(f"Reshaping for NPU (uses_cfg={uses_cfg})...")
            loading_status = "reshaping UNet for NPU"
            pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)

            if not uses_cfg:
                unet_model = pipe.unet.model
                shapes = {}
                for inp in unet_model.inputs:
                    shape = inp.get_partial_shape()
                    shape[0] = 1
                    shapes[inp] = shape
                unet_model.reshape(shapes)
                log.info("Fixed UNet batch size to 1 (no CFG)")

            loading_status = "compiling UNet for NPU"
            log.info("Compiling UNet for NPU...")
            try:
                unet_compiled["NPU"] = core.compile_model(pipe.unet.model, "NPU")
                log.info("  NPU: OK")
            except Exception as e:
                log.warning(f"  NPU: failed ({e})")

        # Restore text encoder/VAE to dynamic-shape compiled versions
        pipe.text_encoder.request = saved_requests["text_encoder"]
        pipe.vae_decoder.request = saved_requests["vae_decoder"]
        if "vae_encoder" in saved_requests:
            pipe.vae_encoder.request = saved_requests["vae_encoder"]
        if "text_encoder_2" in saved_requests:
            pipe.text_encoder_2.request = saved_requests["text_encoder_2"]

        available_devices = list(unet_compiled.keys())
        if not available_devices:
            loading_status = "error: no devices available"
            return

        # Set initial UNet to preferred device
        default_device = "NPU" if "NPU" in available_devices else "CPU"
        pipe.unet.request = unet_compiled[default_device]

        loading_status = "ready"
        log.info(f"Ready. Available devices: {available_devices}")

    except Exception as e:
        loading_status = f"error: {e}"
        log.error(f"Failed to load model: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to OpenVINO model directory")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Start model loading in background thread
    t = threading.Thread(target=load_and_compile, args=(args.model,), daemon=True)
    t.start()

    # Start web server immediately
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
