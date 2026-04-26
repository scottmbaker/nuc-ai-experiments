# How securitycam works

A walkthrough of how this project turns four RTSP camera streams into annotated
video plus an event log, using every piece of compute silicon on an Intel NUC
at the same time.

This guide is written for two overlapping audiences:

- Engineers curious about **heterogeneous compute** (splitting AI workloads
  across CPU, GPU, and NPU) on small Intel platforms.
- Engineers curious about what's actually involved in **object detection**,
  **automatic license plate recognition (ALPR)**, and building AI features
  into a security-camera pipeline.

It assumes general programming literacy and some familiarity with Python and
Linux containers. It does **not** assume you've used OpenVINO, OpenCV, YOLO,
or neural-network inference before.

---

## 1. What this is

`securitycam` is a single Python process that:

1. Opens up to four video sources — either RTSP streams from IP cameras, or
   local test video files bundled into the container.
2. Runs **YOLOv8** object detection on every frame, looking for people,
   vehicles, pets, and a few other COCO classes.
3. When a vehicle is detected, runs a second neural network to locate the
   **license plate**, crops it, and feeds the crop into an OCR model to
   extract the text.
4. Serves the annotated frames back out over HTTP as **MJPEG** streams, plus
   a small REST/JSON API, plus a web UI.
5. Writes detection events and plate reads to SQLite, and saves short MP4
   clips around each event.

The interesting part is where the work runs:

| Work | Hardware |
| --- | --- |
| Object detection (YOLOv8n, INT8) | **NPU** |
| Plate detector (YOLOv9-t, FP16) | **NPU** |
| Plate OCR (CCT transformer) | **CPU** |
| RTSP decode, JPEG encode, MJPEG serving, orchestration | **CPU** |
| Pre-roll ring buffer, clip writing, SQLite, FastAPI | **CPU** |
| _(reserved for future work)_ | **GPU** |

The goal of the project was to demonstrate that a low-power Intel platform can
run a real 4-camera pipeline by spreading work across the accelerators
available on the SoC, instead of pushing everything to a single one.

---

## 2. Key concepts

If you're new to this domain, it's worth defining a few things before reading
the code.

- **NPU (Neural Processing Unit).** A fixed-function accelerator for neural
  network inference, integrated on recent Intel Core Ultra chips. Compared to
  a CPU it is much more power-efficient for the matrix math that neural
  networks are made of, but it is also much pickier about what it will accept
  (input shapes must usually be fixed at compile time, data types are often
  restricted to INT8 / FP16, and the compiler rejects models with ops it
  doesn't implement).
- **GPU (Xe iGPU).** The integrated Intel graphics engine. Capable of running
  neural networks too, but in this project it is deliberately left idle and
  reserved for future work like person re-identification.
- **OpenVINO.** Intel's open-source runtime for running neural networks on
  Intel CPUs, GPUs, and NPUs. You give it a model and the string name of a
  device (`"CPU"`, `"GPU"`, `"NPU"`), and it compiles the model for that
  device and hands you back a callable object. See
  [docs.openvino.ai](https://docs.openvino.ai/).
- **OpenVINO IR (Intermediate Representation).** OpenVINO's native model
  format — a pair of files, `model.xml` (graph) and `model.bin` (weights).
  Most models in the ecosystem ship as ONNX or PyTorch; they need to be
  _converted_ to IR before OpenVINO can run them.
- **ONNX (Open Neural Network Exchange).** A portable model file format.
  Many model authors publish ONNX because it can be consumed by nearly every
  runtime. OpenVINO can read ONNX directly, but for NPU you typically convert
  it to IR first so you can pin the input shape.
- **YOLO (You Only Look Once).** A family of fast single-shot object
  detectors. They take a fixed-size square image in and emit a list of
  bounding boxes plus class scores. We use two flavours:
  - `YOLOv8n` — the "nano" variant, trained on the COCO dataset (80 everyday
    object classes). Source: [ultralytics](https://github.com/ultralytics/ultralytics).
  - `YOLOv9-t` — a "tiny" YOLOv9 trained specifically on license plates,
    shipped by the `fast-alpr` project.
- **COCO.** A public image dataset of ~80 everyday object classes (person,
  car, dog, chair, …). Stock YOLO models output a score for each class;
  we filter to a whitelist.
- **INT8 quantization.** Replacing the 32-bit floating-point weights in a
  network with 8-bit integers. Done properly (with a _calibration_ dataset
  to tune the quantization ranges) you keep almost all the accuracy but the
  model gets smaller and is often the only way to get it to run on an NPU.
  We use **NNCF** (Neural Network Compression Framework) from Intel to do
  this, calibrated on `coco128` (a 128-image subset of COCO).
- **FP16.** 16-bit floating point. The plate detector runs in FP16 on the
  NPU (it wasn't quantized to INT8).
- **NMS (Non-Maximum Suppression).** A post-processing step that collapses
  overlapping duplicate boxes the detector emits for the same object. For
  YOLOv8 we do NMS ourselves with `cv2.dnn.NMSBoxes`; the fast-alpr plate
  detector has NMS baked into the exported model ("end-to-end").
- **Letterbox preprocessing.** Resizing an image to the model's expected
  square input (e.g. 640×640) while preserving aspect ratio, padding the
  leftover area with a neutral grey. Every YOLO implementation expects this.
- **RTSP (Real Time Streaming Protocol).** The de-facto streaming protocol
  for IP security cameras. OpenCV's `VideoCapture` can open an
  `rtsp://user:pass@host/...` URL and pull frames from it.
- **MJPEG (Motion-JPEG).** A "video" format that is just a sequence of JPEG
  frames concatenated with HTTP `multipart` boundaries. It's dead simple,
  the `<img>` tag in any browser will render it, and it doesn't require
  WebRTC or HLS plumbing. It's also inefficient (no interframe compression)
  which is fine for a 4-pane LAN UI.
- **ALPR (Automatic License Plate Recognition).** A two-stage pipeline:
  _detect_ where the plate is in the image (a bounding box), then _recognize_
  the characters on it (OCR). We use [`fast-alpr`](https://github.com/ankandrew/fast-alpr)
  for both model weights and the OCR engine.
- **OCR (Optical Character Recognition).** Turning a cropped image of text
  into a string. The plate OCR here is a small transformer model called
  `cct-xs-v2-global-model`, shipped by fast-alpr.
- **OpenCV.** A general-purpose computer-vision C++ library with Python
  bindings. We use it for everything non-neural: RTSP decode (`VideoCapture`),
  resize and pad (`cv2.resize`, `cv2.copyMakeBorder`), JPEG encode
  (`cv2.imencode`), MP4 muxing (`cv2.VideoWriter`), and overlay drawing
  (`cv2.rectangle`, `cv2.putText`).

---

## 3. The technologies involved

Concretely, the parts that show up in the code:

| Layer | What it is | Role here |
| --- | --- | --- |
| **OpenVINO runtime** (`openvino` Python package) | Intel's neural-net inference engine | Compiles YOLOv8 and the plate detector for the NPU, compiles OCR fallbacks for CPU |
| **Ultralytics** (`ultralytics`) | Official YOLO training/inference library | Used _only_ at build time to export YOLOv8n to OpenVINO INT8 IR |
| **NNCF** (`nncf`) | Intel's quantization toolkit | Does the INT8 post-training quantization with COCO128 as calibration data |
| **fast-alpr** (`fast-alpr`) | Python ALPR library by ankandrew | Supplies the plate-detector ONNX, the CCT OCR model, and a CPU-only fallback path |
| **OpenCV** (`opencv-python-headless`) | Computer vision library | RTSP decode, JPEG encode, MP4 write, drawing overlays, NMS |
| **FastAPI** + **uvicorn** | ASGI web framework and server | Serves the web UI, MJPEG streams, and JSON API |
| **SQLite** (stdlib `sqlite3`) | Embedded database | Persists detection and plate events |
| **Intel GPU user-mode driver** (`intel-opencl-icd`, `libze-intel-gpu1`, `libze1`) | Level Zero + Compute Runtime | Required in the image for OpenVINO's GPU plugin (even though we don't currently schedule work on GPU) |
| **Intel NPU driver** (`linux-npu-driver`) | User-mode NPU driver | Required in the image for OpenVINO's NPU plugin |
| **k3s + Helm** | Lightweight Kubernetes + templating | Packages the image as a single `Pod` with `npu.intel.com/accel` + `gpu.intel.com/xe` device requests |
| **buildkit / nerdctl / containerd** | Image build + local runtime | Builds the image and loads it into the k3s containerd namespace without a registry push |

---

## 4. How the pieces fit together

The application runs inside one container. At the top level it is a bundle of
Python threads plus a FastAPI event loop, all sharing a handful of queues and
a `slots[]` state list.

```
  RTSP / MP4 sources                             Web browser
        │                                             │
        │ (4×)                                        │ GET /stream/N
        ▼                                             ▼
 ┌──────────────┐  frame_queue     ┌──────────────────────────┐
 │ CaptureThread│----------------->│    DetectorThread        │ ─ NPU: YOLOv8n INT8
 │   × 4 slots  │                  │  (shared across 4 slots) │
 └──────────────┘                  └─────────────┬────────────┘
        ▲                                        │ det_result_queue
        │ per-slot                               ▼
        │ state                   ┌───────────────────────────┐
        │ (latest_frame,          │     PipelineThread        │
        │  frame_buffer,          │  - publish raw frame+dets │
        │  fps, plates,           │  - trigger clip writer    │
        │  connected)             │  - insert into SQLite     │
        │                         │  - push to alpr_queue?    │
        │                         └─────────────┬─────────────┘
        │                                       │ alpr_queue
        │                                       ▼
        │                         ┌───────────────────────────┐
        │                         │      ALPRThread           │
        │                         │  - plate detect (NPU FP16)│
        │                         │  - crop, OCR (CPU)        │
        │                         │  - dedup, write plates    │
        │                         └─────────────┬─────────────┘
        │                                       │
        └───────────────────────────────────────┘
                            │
        MJPEG generator reads each slot's latest
        (frame, dets, plates) and encodes JPEG on demand.
```

There is **one** detector thread and **one** ALPR thread, not four of each.
This is a deliberate design choice driven by a hardware constraint of the NPU
(see §5.4).

Four independent `CaptureThread` objects (one per slot) push frames onto a
single shared `frame_queue`. Each queued item carries its originating
`slot_idx`, so downstream workers can route results back to the correct
slot's state dict.

---

## 5. How the code works

There are two code paths worth understanding: what happens at **build time**
(model preparation, in the Docker image) and what happens at **runtime**
(the Python application).

### 5.1 Build-time: getting models into a shape the NPU can run

The NPU plugin in OpenVINO is strict: it wants **static input shapes**
(no dynamic dimensions), and YOLOv8 runs much better on it in **INT8** than
in FP32. Neither of those is true of the off-the-shelf model, so the Docker
build's first stage prepares them.

The relevant file is
[`export_models.py`](export_models.py), called from the `model-builder`
stage of the [`Dockerfile`](Dockerfile).

**YOLOv8n.** Exported by Ultralytics in one line:

```python
# export_models.py:22-23
model = YOLO("yolov8n.pt")
model.export(format="openvino", int8=True, imgsz=640, data="coco128.yaml")
```

That call does three things behind the scenes:

1. Downloads `yolov8n.pt` (PyTorch weights) from the Ultralytics CDN.
2. Traces the graph to ONNX, then converts it to OpenVINO IR.
3. Runs NNCF post-training quantization with the first 128 COCO images as
   calibration data, and saves an INT8 IR pair.

The result is a pair of files under `/models/yolov8n/` with a fixed
`[1, 3, 640, 640]` input.

**Plate detector.** `fast-alpr` ships its plate detector as an ONNX file with
a _dynamic_ input dimension, which the NPU will reject. We convert it
ourselves in [`export_plate_detector`](export_models.py#L70-L108):

```python
# export_models.py:94-106
model = ov.convert_model(str(onnx_path))
...
model.reshape([1, 3, PLATE_DETECTOR_SIZE, PLATE_DETECTOR_SIZE])
ov.save_model(model, str(out / "plate_detector.xml"))
```

`ov.convert_model` parses the ONNX and builds an OpenVINO model object
in-process; `model.reshape(...)` pins the input to a fixed
`[1, 3, 384, 384]`; `ov.save_model` writes the IR pair. The reason we do
this with `ov.convert_model` rather than `ov.Core().read_model` is a
practical one: initialising the full `Core` inside a buildkit worker hits
a SIGBUS. `convert_model` is a lighter code path that does the conversion
without spinning up device plugins.

**Plate OCR.** Not exported — it stays as an ONNX file inside fast-alpr's
own cache. At runtime we just call fast-alpr's `ocr.predict(crop)` on each
plate crop and let it use its own ONNX Runtime session on CPU.

### 5.2 Runtime: loading models onto devices

When the container starts, [`securitycam.py`](securitycam.py) spawns the HTTP
server synchronously but loads models on a background thread so the readiness
probe can go green quickly.

[`load_models`](securitycam.py#L1240-L1312) is where device assignment
happens. The key lines:

```python
# securitycam.py:1241-1246
core = ov.Core()
available = core.available_devices
...
npu_dev = "NPU" if "NPU" in available else "CPU"
```

`core.available_devices` is OpenVINO's list of device strings discovered on
the host. In the container, if the NPU device plugin has successfully
mapped `/dev/accel/accel0` into the Pod, `"NPU"` will be in the list.
If something is wrong (driver missing, device plugin not wired up, NPU
already locked by another process), it won't be, and we fall back to CPU.

Then for each model:

```python
# securitycam.py:1254-1255
model = core.read_model(str(yolo_xml))
ctx["detector"] = core.compile_model(model, npu_dev)
```

`compile_model` is where the heavy lifting happens. For the NPU target, it
JIT-compiles the IR into the NPU's internal instruction format — this takes
several seconds the first time and is cached. The returned object is a
callable; you pass it a dict of `{input_name_or_index: numpy_array}` and
it returns a dict of output tensors.

The plate detector is loaded the same way, with a `try`/`except` that falls
back to CPU if NPU compile fails. The active device is stored in
`ctx["plate_detector_device"]` so the UI can render a "NPU" vs
"CPU fallback" badge.

### 5.3 Capture: four sources into one queue

Each RTSP camera (or local test video) gets its own [`CaptureThread`](securitycam.py#L661-L766).
Its job is to open the source, read frames, and shove them onto the shared
`frame_queue`.

Each thread:

1. Opens `cv2.VideoCapture(url)`. For RTSP, OpenCV uses FFmpeg under the
   hood to connect, parse the SDP, and decode H.264 / H.265.
2. Reads frames in a tight loop, using the stream's reported FPS to decide
   how often to pull:
   ```python
   # securitycam.py:704-705
   native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
   interval = 1.0 / native_fps
   ```
3. Appends every frame to a **per-slot ring buffer**
   (`collections.deque(maxlen=150)`). At ~30 fps this is 5 seconds of
   history — the pre-roll source for clip saving.
4. Pushes `(slot_idx, frame_id, frame, timestamp)` onto `frame_queue`
   using `put_nowait` so capture never blocks on downstream congestion:
   ```python
   # securitycam.py:737-741
   try:
       frame_queue.put_nowait((slot["idx"], self.frame_id, frame, now))
   except queue.Full:
       pass  # backpressure — drop rather than block
   ```
5. On any read failure or EOF, marks the slot disconnected and retries
   forever with exponential backoff capped at 30 s. This is what lets the
   pod survive a camera rebooting, a network hiccup, or a wrong URL.

### 5.4 Detection: the NPU bottleneck and why there's only one detector thread

[`DetectorThread`](securitycam.py#L772-L813) is the hot path on the NPU. It
blocks on `frame_queue`, letterboxes the frame to 640×640, runs YOLOv8,
post-processes to boxes, and emits a `(slot_idx, frame, detections, …)`
tuple onto `det_result_queue`.

Preprocessing is in [`preprocess_yolo`](securitycam.py#L411-L422):

```python
# securitycam.py:411-422
def preprocess_yolo(frame, size=640):
    h, w = frame.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top = (size - nh) // 2
    left = (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, top, left, h, w
```

This is the standard YOLO "letterbox": compute the scale that fits the frame
into the 640×640 square without distortion, place it centred on a grey
(RGB 114/114/114) canvas, convert to CHW float32 in `[0, 1]`, add a batch
dimension. The `scale`, `top`, `left` values come back so post-processing
can undo the transformation and give boxes in the original frame's
coordinates.

Post-processing in [`postprocess_yolo`](securitycam.py#L425-L468) decodes the
`[8400, 84]` raw output into `(box, class, score)` triples, filters by
confidence, undoes the letterbox, runs NMS with `cv2.dnn.NMSBoxes`, and
drops anything not in the COCO whitelist.

The interesting piece is the NPU lock:

```python
# securitycam.py:115-124
npu_lock = threading.Lock()

def npu_guard(device):
    return npu_lock if device == "NPU" else nullcontext()
```

...used like this:

```python
# securitycam.py:793-794
with npu_guard(ctx.get("npu_device", "")):
    out = ctx["detector"]({0: blob})
```

The Panther Lake NPU firmware serializes work across clients via
time-division multiplexing, but racing two OpenVINO inference calls from
different Python threads can still trip `ZE_RESULT_ERROR_DEVICE_LOST` and
hang the device until the container restarts. This is a documented hardware
limitation, tracked at
[intel/linux-npu-driver#128](https://github.com/intel/linux-npu-driver/issues/128).

The fix is structural: route **all** NPU-bound inference through a single
application-level mutex. `npu_guard` returns the real lock when the device
is `"NPU"`, and a no-op `nullcontext()` when it isn't (so CPU fallback
inference isn't artificially serialised).

This is also the reason there's one `DetectorThread` and one `ALPRThread`,
not four of each. If we fanned out per-camera, all those threads would
contend on the same lock and the scheduling would be whatever Python's
GIL release pattern happens to produce. Funnelling through a single worker
makes the ordering explicit and latency accounting simple.

### 5.5 Pipeline: route results, trigger clips, maybe queue ALPR

[`PipelineThread`](securitycam.py#L895-L969) consumes `det_result_queue`.
For each frame, it:

1. Atomically updates the slot's `latest_frame` + `latest_dets` so the MJPEG
   generator has fresh data to serve.
2. Decides whether to run ALPR. The decision depends on `ctx["alpr_mode"]`:
   - `always` → every frame
   - `gated` → only if YOLO detected a class in `{"car", "truck", "bus", "motorcycle"}`
   - `never` → skip
   If the gate passes, it pushes the frame onto `alpr_queue`.
3. For every detection, inserts a row into SQLite's `detections` table,
   appends to the in-memory `event_log` deque, calls the stubbed alerter,
   and triggers the per-slot `ClipWriter`.

One subtlety: the raw frame is stored on the slot, but the frame is _not_
annotated here. Annotation happens lazily in the MJPEG generator (§5.8).
This keeps the pipeline O(1) per frame and lets annotation rate follow
viewer rate: if nobody's watching, we don't burn cycles drawing boxes.

### 5.6 ALPR: two stages, two devices

[`ALPRThread`](securitycam.py#L975-L1040) does license-plate recognition
asynchronously — the detection pipeline never blocks on it.

The work splits across devices:

- **Plate detection (NPU).** [`run_alpr_npu`](securitycam.py#L569-L607)
  letterboxes to 384×384, runs the YOLOv9-t plate IR, and decodes the
  end-to-end output:
  ```python
  # securitycam.py:579-580
  with npu_guard(ctx.get("plate_detector_device", "")):
      out = detector({0: blob})
  ```
  The plate-detector ONNX was exported with NMS baked in, so its output is
  already a clean `[N, 7]` table of `(batch, x1, y1, x2, y2, class, score)`
  rows. [`_postprocess_plate_detector`](securitycam.py#L496-L531) undoes
  the letterbox and returns boxes in original-frame coordinates.
- **OCR (CPU).** Each decoded box is cropped out of the original frame and
  passed to fast-alpr's OCR engine:
  ```python
  # securitycam.py:543-547
  ocr_engine = getattr(alpr, "ocr", None)
  ...
  result = ocr_engine.predict(crop)
  ```
  `alpr.ocr` is fast-alpr's CCT OCR (a transformer trained on plate
  crops), running on CPU via ONNX Runtime. Calling `.predict` directly on
  our own crop bypasses fast-alpr's internal detector — we've already done
  that stage on the NPU.

Both stages are behind the same `npu_lock` (via `npu_guard`) so the plate
detector can't race the YOLOv8 detector. The OCR stage, being on CPU,
takes the no-op path.

If the plate-detector IR failed to compile, the thread falls back to
[`run_alpr_fallback`](securitycam.py#L610-L655), which calls
`alpr.predict(frame)` — fast-alpr's internal end-to-end pipeline, entirely
on CPU. The UI shows whichever path is active.

**Dedup.** Plates tend to stay visible for many frames, so
`ALPRThread.recent_plates` suppresses writes of the same plate text within
a 30-second window. This prevents one visiting car from producing 900
database rows.

### 5.7 Clip writer: pre-roll + post-roll MP4 around each event

Each slot has its own [`ClipWriter`](securitycam.py#L265-L382). The design
is "the ring buffer is always full; when something interesting happens, dump
the last N seconds and keep recording for M more."

The mechanics:

- The `CaptureThread` appends every raw frame to `slot["frame_buffer"]`
  (a `deque(maxlen=150)`), independent of clip state.
- When `PipelineThread` sees a detection, it calls
  [`ClipWriter.trigger(class_name)`](securitycam.py#L294-L334). If no clip
  is currently writing, trigger snapshots the pre-roll by copying the
  deque, spawns a background worker thread with that snapshot, and returns
  immediately.
- The worker opens a `cv2.VideoWriter` with `mp4v` fourcc, writes the
  pre-roll frames, then consumes live frames off a per-clip `live_queue`
  until a post-roll deadline expires.
- `PipelineThread` also calls `ClipWriter.add_frame(frame)` for every
  post-detection frame; it `put_nowait`s onto the live queue and drops
  if the queue is full.
- If a new detection of a different class arrives mid-clip, `trigger`
  just extends `write_until` — it doesn't start a second clip.

The point of all this indirection: `VideoWriter.write(frame)` can block on
disk, and we never want the hot pipeline thread to pay that cost.

### 5.8 Web UI: MJPEG streams, lazy annotation

FastAPI hosts the UI. The interesting endpoint is
[`/stream/{slot_idx}`](securitycam.py#L1085-L1112), a streaming generator:

```python
# securitycam.py:1091-1112
def generate():
    while True:
        with slot["latest_frame_lock"]:
            raw = slot["latest_frame"]
            dets = slot["latest_dets"]
        if raw is None:
            frame = _placeholder_frame(slot)
        else:
            with telemetry_lock:
                gt = dict(telemetry)
            frame = annotate_frame(raw, dets or [], slot, gt)
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + buf.tobytes() + b"\r\n")
        time.sleep(0.033)
return StreamingResponse(
    generate(), media_type="multipart/x-mixed-replace; boundary=frame")
```

Three things happen here:

1. The response `Content-Type` is `multipart/x-mixed-replace; boundary=frame`.
   This is the original MJPEG-over-HTTP convention from 1990s webcams, and
   every browser still implements it — point an `<img src="/stream/0">` at
   it and the image updates forever.
2. [`annotate_frame`](securitycam.py#L832-L889) is called per-yield. It
   draws boxes, plate overlays, the camera name, and a telemetry bar
   _on the raw frame, at view time_. If no browser is looking, this work
   doesn't happen.
3. `cv2.imencode` does JPEG compression on CPU. This is the single
   largest CPU cost of the UI path at 4 cameras.

The web UI itself (`static/index.html`) is static HTML+JavaScript. It
polls `/api/telemetry`, `/api/events`, and `/api/plates` on a timer and
embeds the four `<img>` elements that point at the MJPEG endpoints.
Layout state (2×2 grid vs. 1×1 focus) lives in `localStorage` and never
hits the server.

### 5.9 Runtime toggles

Two controls can be flipped live from the UI (or by `curl`):

- `POST /api/source_mode/{live|test}` —
  [`api_set_source_mode`](securitycam.py#L1178-L1190) stops every
  `CaptureThread`, reassigns slot identities from `cameras.yaml` or the
  bundled test videos, drains the shared queues (so stale frames from the
  old mode don't reach the UI), and starts fresh captures. This is handy
  during development: flip to `test` to see a known-good scene, flip back
  to `live` when done.
- `POST /api/alpr/mode/{always|gated|never}` —
  [`api_set_alpr_mode`](securitycam.py#L1203-L1220) just rewrites
  `ctx["alpr_mode"]` and clears any stale plate overlays.

---

## 6. Deployment shape

The runtime image is built by the two-stage
[`Dockerfile`](Dockerfile): stage one exports models and downloads test
videos, stage two installs the Intel GPU and NPU user-mode drivers plus the
Python runtime, and copies the models over. `buildkit` runs the build and
`nerdctl` loads the resulting image into the local k3s containerd
namespace — no registry is involved.

The Helm chart in [`chart/`](chart/) deploys a single `Pod` with two
device requests:

```yaml
# chart/templates/pod.yaml:36-41
resources:
  requests:
    npu.intel.com/accel: "1"
    gpu.intel.com/xe: "1"
```

Those resource names come from the [**Intel Device Plugins for Kubernetes**](https://github.com/intel/intel-device-plugins-for-kubernetes),
which are expected to be running on the node. The NPU plugin advertises
`npu.intel.com/accel` and mounts `/dev/accel/accel0` into the container;
the GPU plugin advertises `gpu.intel.com/xe` and mounts the DRI render
node. Without those plugins, the Pod would schedule but OpenVINO would see
`available_devices == ["CPU"]` and silently fall back to CPU-only inference.

---

## 7. Going further

The authoritative references for the technologies in this project:

- **OpenVINO.** The [OpenVINO documentation](https://docs.openvino.ai/) —
  especially the sections on
  [NPU device](https://docs.openvino.ai/2024/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
  and [model conversion](https://docs.openvino.ai/2024/openvino-workflow/model-preparation/convert-model-to-ir.html).
- **Ultralytics YOLO.** The [Ultralytics docs](https://docs.ultralytics.com/)
  and the [OpenVINO export guide](https://docs.ultralytics.com/integrations/openvino/).
- **NNCF.** [NNCF on GitHub](https://github.com/openvinotoolkit/nncf) —
  post-training quantization is in `nncf.quantize`.
- **fast-alpr.** The [fast-alpr project](https://github.com/ankandrew/fast-alpr)
  explains the two-stage detect-then-read pipeline and lists the model zoo.
- **Intel NPU driver.** [linux-npu-driver on GitHub](https://github.com/intel/linux-npu-driver)
  — the issue tracker is the canonical source for what the NPU does and
  doesn't support today.
- **Intel Device Plugins for Kubernetes.**
  [intel/intel-device-plugins-for-kubernetes](https://github.com/intel/intel-device-plugins-for-kubernetes)
  — how `gpu.intel.com/xe` and `npu.intel.com/accel` get onto your node.
- **COCO dataset.** [cocodataset.org](https://cocodataset.org/) — the
  class list and annotation format that stock YOLO is trained against.
- **RTSP.** [RFC 2326](https://www.rfc-editor.org/rfc/rfc2326) is the
  original protocol spec. In practice IP cameras implement a subset plus
  vendor-specific paths; OpenCV/FFmpeg handle the common shapes.


