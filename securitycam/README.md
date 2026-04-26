# securitycam

Heterogeneous-compute security-camera AI pipeline for Intel NUC: YOLOv8 object detection and
license-plate recognition across NPU + CPU, served as a web UI.

## Overview

`securitycam` ingests up to four RTSP streams (or four bundled test videos), runs YOLOv8n object
detection and YOLOv9-t plate detection on the Intel NPU, runs plate OCR on the CPU via
[fast-alpr](https://github.com/ankandrew/fast-alpr), and exposes an MJPEG preview and JSON APIs
through FastAPI. Detections and plate reads are persisted to SQLite, and clips can be saved around
events. It is packaged as a container and deployed to k3s via a Helm chart.

Scope is an experiment / home-lab build on a single Intel NUC with an NPU + Xe GPU. It is not a
general-purpose NVR.

## Requirements

- Linux host with an Intel NPU (tested on Panther Lake / NPU5) and Intel Xe GPU
- k3s with the Intel device plugins exposing `npu.intel.com/accel` and `gpu.intel.com/xe`
- containerd reachable at `/run/k3s/containerd/containerd.sock`
- `buildkitd`, `nerdctl`, `helm`, `kubectl` on the build host
- Python 3 is only needed inside the container image — the host does not need it

The build downloads ~150 MB of Pexels test videos and the Intel NPU/GPU userspace packages, so the
first build needs network access.

## Build

```sh
make build
```

This starts `buildkitd` if needed, runs the two-stage Dockerfile (model export + runtime), and
tags `localhost/securitycam:latest` in the k3s containerd namespace. The model-builder stage
exports YOLOv8n to OpenVINO INT8 IR and converts the fast-alpr plate detector to OpenVINO IR with
a static `[1, 3, 384, 384]` shape for NPU compilation.

## Deploy

Test mode (bundled Pexels clips, no cameras.yaml needed):

```sh
make deploy MODE=test
```

Live mode with your own RTSP cameras:

```sh
cp cameras.yaml.example cameras.yaml
# edit cameras.yaml with real URLs
make cameras            # creates the securitycam-cameras k8s secret
make deploy MODE=live
```

Make variables:

- `MODE=test|live` — `test` uses bundled videos, `live` reads `cameras.yaml` from a secret
- `DEVICES=npu,gpu` — which Intel accelerators to request (defaults to both)

Once the pod is ready, `make deploy` prints the NodePort URL (default `http://<node>:30083`).

Other targets: `make logs`, `make status`, `make undeploy`, `make nuke` (clear orphaned containerd
state for the pod), `make clean` (remove built images). `make help` lists them all.

## Configuration

- `config.yaml` — detection confidence, whitelisted COCO classes, clip pre/post-roll, alerts,
  ALPR mode (`always` / `gated` / `never`), SQLite path. Baked into the image.
- `cameras.yaml` — list of `{name, url}` entries. Gitignored; loaded at runtime from a k8s secret
  in live mode. See `cameras.yaml.example`.

ALPR mode and source mode (`live` vs `test`) can also be toggled at runtime through the web UI or
`POST /api/source_mode/{mode}` and `POST /api/alpr/mode/{mode}`.

## Web UI and API

The UI at `/` shows a 2×2 grid of MJPEG streams with per-slot detection overlays and a live event
feed. Key endpoints:

- `GET /stream/{slot_idx}` — MJPEG stream for slot 0–3
- `GET /status` — liveness/readiness JSON (used by the k8s probe)
- `GET /api/telemetry`, `/api/events`, `/api/detections`, `/api/plates`, `/api/clips`
- `GET /api/cameras`, `/api/source_mode`, `/api/alpr/mode`
- `POST /api/source_mode/{mode}`, `POST /api/alpr/mode/{mode}`

## Layout

```
securitycam.py       # main FastAPI app, capture/inference/clip pipelines
export_models.py     # build-time model export to OpenVINO IR
config.yaml          # runtime config baked into the image
cameras.yaml.example # template for live-mode RTSP sources
Dockerfile           # two-stage: model-builder + runtime
Makefile             # build / deploy / ops targets
chart/               # Helm chart (single Pod + NodePort Service)
static/              # web UI (index.html)
```

## Development notes

- Runs outside the container too: `python3 securitycam.py --test --port 8080` from the
  `securitycam/` directory, given OpenVINO and the exported models are available locally. Inside
  the container this is the default entrypoint.
- Single `--source <path-or-rtsp>` fills slot 0 only, for debugging one stream at a time.
- NPU5 cannot run multiple OpenVINO contexts concurrently, so the pipeline serializes the YOLO
  detector and plate detector through a shared worker.

