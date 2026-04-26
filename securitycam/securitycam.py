#!/usr/bin/env python3
"""Security camera AI pipeline — heterogeneous compute demo.

NPU: YOLOv8n object detection + YOLOv9-t plate detector (OpenVINO, serialized)
CPU: ALPR CCT OCR, orchestration, capture, encoding, clip saving
GPU: free for future use (person re-ID, etc.)

Up to 4 concurrent RTSP streams (or 4 bundled test videos), one shared
detector + ALPR worker pool, per-slot frame state and MJPEG streams.
"""

import argparse
import collections
import logging
import os
import queue
import sqlite3
import threading
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import openvino as ov
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("securitycam")

# ---------------------------------------------------------------------------
# COCO class names (YOLOv8 output indices)
# ---------------------------------------------------------------------------
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

DEFAULT_WHITELIST = {
    "person", "car", "bicycle", "dog", "cat", "truck", "motorcycle", "bus",
}

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle"}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
ALPR_MODES = ("always", "gated", "never")
SOURCE_MODES = ("live", "test")
NUM_SLOTS = 4

# In test mode, slots 0..3 fill from this fixed list of bundled videos.
TEST_VIDEOS = [
    ("parking-lot", "/app/videos/parking-lot.mp4"),
    ("intersection", "/app/videos/intersection.mp4"),
    ("london-traffic", "/app/videos/london-traffic.mp4"),
    ("nyc-street", "/app/videos/nyc-street.mp4"),
]

ctx = {
    "status": "starting",
    "detector": None,
    "config": {},
    "cameras": [],            # from cameras.yaml — list of {name, url}
    "source_mode": "live",    # "live" | "test" — runtime-toggled
    "npu_device": "NPU",
    "alpr": None,                 # fast-alpr fallback (end-to-end on CPU)
    "plate_detector": None,       # OpenVINO compiled plate detector (NPU/CPU)
    "plate_detector_device": "",  # "NPU" or "CPU" (fallback) — reported to UI
    "plate_detector_input": None,  # (H, W) of compiled input
    "alpr_mode": "gated",         # always | gated | never (runtime-toggled)
    "db": None,
    "ready_event": threading.Event(),
    # The per-slot state lives in `slots` below, built by init_slots().
}

# Per-slot state (one dict per slot, 0..NUM_SLOTS-1). Built lazily in
# init_slots() because some fields depend on config values read at startup.
slots = []

# Telemetry shared across slots. Per-slot telemetry lives on the slot dict.
telemetry = {
    "npu_latency_ms": 0.0,
    "npu_active": False,
    "alpr_latency_ms": 0.0,   # latency of the most recent ALPR run (any slot)
    "alpr_mode": "gated",
    "alpr_device": "",        # reported NPU/CPU so UI can show fallback badge
    "last_alpr_slot": -1,
}
telemetry_lock = threading.Lock()

# Panther Lake NPU (NPU5) firmware serializes multi-context work via
# time-division multiplexing; racing two inferences from Python can still
# trip ZE_RESULT_ERROR_DEVICE_LOST (github.com/intel/linux-npu-driver#128).
# Serialize all NPU-bound inference at the app level to avoid the race.
npu_lock = threading.Lock()


def npu_guard(device):
    """Return the NPU lock if `device` is 'NPU', else a no-op context.

    Used to wrap inference calls so CPU-resident models don't pay lock cost
    and don't serialize themselves against unrelated NPU work.
    """
    return npu_lock if device == "NPU" else nullcontext()

event_log = collections.deque(maxlen=200)
event_log_lock = threading.Lock()

# Shared inter-thread queues — every item carries slot_idx so results route
# back to the correct slot. Sizes scaled for 4 cameras.
frame_queue = queue.Queue(maxsize=40)       # capture(any slot) → NPU detector
det_result_queue = queue.Queue(maxsize=40)  # detector → pipeline
alpr_queue = queue.Queue(maxsize=20)        # pipeline → ALPR thread


def new_slot(idx):
    """Create the persistent per-slot state container. Capture-specific
    fields (name, url, type, capture thread) are reset on source-mode
    swaps via assign_slot(); everything else persists across swaps."""
    return {
        "idx": idx,
        # Source identity (rewritten by assign_slot on mode change)
        "name": "",
        "url": "",
        "source_type": "none",   # "rtsp" | "video" | "none"
        "loop": False,
        "capture": None,         # CaptureThread or None
        # Frame state. `latest_frame` stores the RAW frame plus the
        # detections that go with it; annotation is done lazily by the
        # MJPEG stream generator. `latest_frame_lock` covers both fields
        # so readers see a matched (frame, dets) pair.
        "latest_frame": None,
        "latest_dets": [],
        "latest_frame_lock": threading.Lock(),
        "frame_buffer": collections.deque(maxlen=150),  # ~5 s at 30 fps
        "frame_buffer_lock": threading.Lock(),
        # Telemetry
        "fps": 0.0,
        "last_frame_time": 0.0,
        "connected": False,
        "last_error": "",
        "detections": [],
        "plates": [],
        # Per-slot clip writer; assigned in main() once config is loaded.
        "clip_writer": None,
    }


def init_slots():
    """Populate `slots` with NUM_SLOTS empty-slot state containers."""
    global slots
    slots = [new_slot(i) for i in range(NUM_SLOTS)]


# ---------------------------------------------------------------------------
# SQLite detection log
# ---------------------------------------------------------------------------
class DetectionDB:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        # WAL + synchronous=NORMAL trades one fsync-per-commit for one
        # fsync-per-checkpoint. For a detection log this is a safe tradeoff
        # (tiny crash window) and gets us out of the fsync hot path when
        # 4 cameras drive hundreds of inserts/sec.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                class      TEXT NOT NULL,
                confidence REAL NOT NULL,
                x REAL, y REAL, w REAL, h REAL,
                scene_label TEXT,
                clip_file   TEXT,
                alert_sent  INTEGER DEFAULT 0,
                camera      TEXT
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT NOT NULL,
                plate_text TEXT NOT NULL,
                confidence REAL NOT NULL,
                camera     TEXT
            )
        """)
        self.conn.commit()

    def insert(self, timestamp, cls, confidence, box,
               scene_label, clip_file=None, alert_sent=False, camera=None):
        with self.lock:
            x, y, w, h = box if box else (0, 0, 0, 0)
            self.conn.execute(
                "INSERT INTO detections "
                "(timestamp,class,confidence,x,y,w,h,"
                "scene_label,clip_file,alert_sent,camera) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (timestamp, cls, confidence, x, y, w, h,
                 scene_label, clip_file, int(alert_sent), camera),
            )
            self.conn.commit()

    def recent(self, limit=50):
        with self.lock:
            rows = self.conn.execute(
                "SELECT timestamp,class,confidence,scene_label,"
                "clip_file,camera FROM detections "
                "ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [
            {"timestamp": r[0], "class": r[1], "confidence": r[2],
             "scene_label": r[3], "clip_file": r[4], "camera": r[5]}
            for r in rows
        ]

    def insert_plate(self, timestamp, plate_text, confidence, camera=None):
        with self.lock:
            self.conn.execute(
                "INSERT INTO plates (timestamp,plate_text,confidence,camera)"
                " VALUES (?,?,?,?)",
                (timestamp, plate_text, confidence, camera),
            )
            self.conn.commit()

    def recent_plates(self, limit=50):
        with self.lock:
            rows = self.conn.execute(
                "SELECT timestamp,plate_text,confidence,camera "
                "FROM plates ORDER BY id DESC LIMIT ?", (limit,),
            ).fetchall()
        return [
            {"timestamp": r[0], "plate_text": r[1],
             "confidence": r[2], "camera": r[3]}
            for r in rows
        ]


# ---------------------------------------------------------------------------
# Clip writer
# ---------------------------------------------------------------------------
class ClipWriter:
    """Per-slot clip recorder. All disk I/O runs on a background worker
    thread so PipelineThread never blocks on VideoWriter.

    Flow:
      - trigger() takes a snapshot of the pre-roll buffer, allocates a
        live-frame queue, spawns a worker, returns immediately.
      - add_frame() enqueues a frame non-blocking; drops if queue full.
      - The worker writes pre-roll first, then drains live frames until
        write_until expires (post-roll has passed), then releases the
        VideoWriter. A subsequent trigger() starts a fresh clip.
    """

    LIVE_QUEUE_MAX = 400  # ~13s at 30fps — generous cushion

    def __init__(self, slot, output_dir, pre_roll=5, post_roll=10, fps=15):
        self.slot = slot
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pre_roll = pre_roll
        self.post_roll = post_roll
        self.fps = fps
        self.lock = threading.Lock()
        self.writing = False
        self.write_until = 0.0
        self.filename = None
        self.live_queue = None
        self.worker = None

    def trigger(self, detection_class):
        """Fast path — takes a buffer snapshot and spawns a writer thread.
        Returns the output filename, or None if no pre-roll was available.
        """
        with self.lock:
            now = time.time()
            if self.writing:
                # Extend post-roll window; worker will read the new
                # write_until on its next tick.
                self.write_until = now + self.post_roll
                return self.filename

            ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            cam = (self.slot.get("name") or
                   f"slot{self.slot.get('idx', '?')}")
            self.filename = f"{cam}_{ts}_{detection_class}.mp4"
            filepath = self.output_dir / self.filename

            with self.slot["frame_buffer_lock"]:
                pre_frames = list(self.slot["frame_buffer"])
            n = int(self.pre_roll * self.fps)
            pre_frames = pre_frames[-n:] if len(pre_frames) > n else pre_frames

            if not pre_frames:
                # Nothing to seed with — abort cleanly
                self.filename = None
                return None

            self.writing = True
            self.write_until = now + self.post_roll
            self.live_queue = queue.Queue(maxsize=self.LIVE_QUEUE_MAX)
            fname = self.filename
            self.worker = threading.Thread(
                target=self._writer_loop,
                args=(str(filepath), pre_frames),
                daemon=True,
                name=f"clip-{cam}-{ts}",
            )
            self.worker.start()
            log.info("Started clip: %s", fname)
            return fname

    def add_frame(self, frame):
        """Non-blocking — drop the frame if the worker can't keep up."""
        if not self.writing:
            return
        if time.time() > self.write_until:
            # Don't feed past the post-roll window; worker will wind down
            return
        lq = self.live_queue
        if lq is None:
            return
        try:
            lq.put_nowait(frame)
        except queue.Full:
            pass  # drop — clip gets fewer frames, but pipeline stays fluid

    def _writer_loop(self, filepath, pre_frames):
        """Runs off the pipeline thread. Writes pre-roll, then live frames
        until write_until passes, then releases the VideoWriter."""
        name = self.filename
        try:
            h, w = pre_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))
            try:
                for f in pre_frames:
                    writer.write(f)
                while True:
                    try:
                        frame = self.live_queue.get(timeout=0.5)
                    except queue.Empty:
                        frame = None
                    if time.time() > self.write_until and frame is None:
                        break
                    if frame is not None:
                        writer.write(frame)
            finally:
                writer.release()
                log.info("Finished clip: %s", name)
        except Exception as e:
            log.exception("Clip writer failed for %s: %s", name, e)
        finally:
            with self.lock:
                self.writing = False
                self.write_until = 0.0
                self.filename = None
                self.live_queue = None
                self.worker = None


# ---------------------------------------------------------------------------
# Alert stub
# ---------------------------------------------------------------------------
class Alerter:
    def __init__(self, config):
        self.enabled = config.get("enabled", False)
        self.method = config.get("method", "ntfy")
        self.cooldown = config.get("cooldown", 60)
        self.last_alert = {}

    def send(self, cls, confidence, scene_label, thumbnail=None):
        if not self.enabled:
            return False
        now = time.time()
        if now - self.last_alert.get(cls, 0) < self.cooldown:
            return False
        self.last_alert[cls] = now
        log.info("ALERT [%s]: %s conf=%.2f scene=%s",
                 self.method, cls, confidence, scene_label)
        # TODO: implement ntfy POST or SMTP
        return True


# ---------------------------------------------------------------------------
# YOLOv8 pre/post-processing
# ---------------------------------------------------------------------------
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


def postprocess_yolo(output, scale, pad_top, pad_left,
                     orig_h, orig_w, conf=0.5, iou=0.45,
                     whitelist=None):
    preds = output[0].T  # [8400, 84]
    boxes = preds[:, :4]
    scores = preds[:, 4:]
    max_sc = np.max(scores, axis=1)
    cls_ids = np.argmax(scores, axis=1)

    mask = max_sc > conf
    boxes, max_sc, cls_ids = boxes[mask], max_sc[mask], cls_ids[mask]
    if len(boxes) == 0:
        return []

    x1 = (boxes[:, 0] - boxes[:, 2] / 2 - pad_left) / scale
    y1 = (boxes[:, 1] - boxes[:, 3] / 2 - pad_top) / scale
    x2 = (boxes[:, 0] + boxes[:, 2] / 2 - pad_left) / scale
    y2 = (boxes[:, 1] + boxes[:, 3] / 2 - pad_top) / scale
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    nms_boxes = [[float(x1[i]), float(y1[i]),
                  float(x2[i] - x1[i]), float(y2[i] - y1[i])]
                 for i in range(len(x1))]
    indices = cv2.dnn.NMSBoxes(nms_boxes, max_sc.tolist(), conf, iou)

    results = []
    for i in (indices.flatten() if len(indices) else []):
        name = COCO_CLASSES[cls_ids[i]] if cls_ids[i] < len(COCO_CLASSES) \
            else f"class_{cls_ids[i]}"
        if whitelist and name not in whitelist:
            continue
        results.append({
            "box": [float(x1[i]), float(y1[i]),
                    float(x2[i]), float(y2[i])],
            "box_norm": [float(x1[i] / orig_w), float(y1[i] / orig_h),
                         float(x2[i] / orig_w), float(y2[i] / orig_h)],
            "score": float(max_sc[i]),
            "class_id": int(cls_ids[i]),
            "class_name": name,
        })
    return results


# ---------------------------------------------------------------------------
# ALPR: plate detector on NPU + fast-alpr OCR on CPU (fallback: fast-alpr e2e)
# ---------------------------------------------------------------------------
PLATE_CONF_THRESHOLD = 0.35


def _letterbox(img, new_size):
    """Resize with unchanged aspect ratio, pad to new_size (H, W). Returns
    (blob CHW float32 [0,1], scale, pad_x, pad_y, orig_h, orig_w).
    """
    new_h, new_w = new_size
    h, w = img.shape[:2]
    scale = min(new_w / w, new_h / h)
    rw, rh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_h, new_w, 3), 114, dtype=np.uint8)
    pad_x = (new_w - rw) // 2
    pad_y = (new_h - rh) // 2
    canvas[pad_y:pad_y + rh, pad_x:pad_x + rw] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]
    return blob, scale, pad_x, pad_y, h, w


def _postprocess_plate_detector(output, scale, pad_x, pad_y, orig_h, orig_w):
    """Decode plate detector output to list of boxes in original image coords.

    The ONNX is fast-alpr/open-image-models' yolo-v9-t-384 end2end, which
    emits `[N, 7]` rows (N is dynamic, up to 100):
        (batch_idx, x1, y1, x2, y2, class_id, score)
    Boxes are in the 384x384 letterboxed space; we undo pad+scale to get
    original frame coordinates. NMS is baked into the ONNX.
    """
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2 or arr.size == 0 or arr.shape[1] < 7:
        return []

    boxes = []
    for row in arr:
        score = float(row[6])
        if score < PLATE_CONF_THRESHOLD:
            continue
        # Undo letterbox (columns 1..4 are x1, y1, x2, y2)
        ox1 = (float(row[1]) - pad_x) / scale
        oy1 = (float(row[2]) - pad_y) / scale
        ox2 = (float(row[3]) - pad_x) / scale
        oy2 = (float(row[4]) - pad_y) / scale
        ox1 = max(0, min(orig_w - 1, int(ox1)))
        oy1 = max(0, min(orig_h - 1, int(oy1)))
        ox2 = max(0, min(orig_w - 1, int(ox2)))
        oy2 = max(0, min(orig_h - 1, int(oy2)))
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        boxes.append({
            "box": [ox1, oy1, ox2, oy2],
            "score": score,
        })
    return boxes


def _ocr_crop(crop):
    """Run fast-alpr's OCR on a single plate crop. Returns (text, conf) or
    (None, 0.0) on failure / unreadable plate.
    """
    alpr = ctx.get("alpr")
    if alpr is None or crop is None or crop.size == 0:
        return None, 0.0
    # fast-alpr exposes its OCR via alpr.ocr.predict(image) — same API used
    # internally by alpr.predict(). This stays on CPU.
    ocr_engine = getattr(alpr, "ocr", None)
    if ocr_engine is None or not hasattr(ocr_engine, "predict"):
        return None, 0.0
    try:
        result = ocr_engine.predict(crop)
    except Exception as e:
        log.debug("OCR predict failed: %s", e)
        return None, 0.0
    if result is None:
        return None, 0.0
    text = (getattr(result, "text", "") or "").strip()
    conf = getattr(result, "confidence", 0.0)
    if isinstance(conf, (list, tuple)):
        conf = sum(conf) / len(conf) if conf else 0.0
    return (text or None), float(conf or 0.0)


def _record_alpr_result(slot_idx, plates, ms):
    """Write ALPR result back to the originating slot and update global
    telemetry (latency, which slot produced it)."""
    slots[slot_idx]["plates"] = plates
    with telemetry_lock:
        telemetry["alpr_latency_ms"] = ms
        telemetry["last_alpr_slot"] = slot_idx


def run_alpr_npu(slot_idx, frame):
    """Run plate detection on NPU, then OCR each crop via fast-alpr on CPU."""
    detector = ctx.get("plate_detector")
    size = ctx.get("plate_detector_input") or (384, 384)
    if detector is None:
        return []

    t0 = time.time()
    blob, scale, pad_x, pad_y, oh, ow = _letterbox(frame, size)
    try:
        with npu_guard(ctx.get("plate_detector_device", "")):
            out = detector({0: blob})
    except Exception as e:
        log.warning("Plate detector inference failed: %s", e)
        return []
    raw = next(iter(out.values()))
    detections = _postprocess_plate_detector(raw, scale, pad_x, pad_y, oh, ow)

    plates = []
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        crop = frame[y1:y2, x1:x2]
        text, ocr_conf = _ocr_crop(crop)
        if not text or len(text) < 2:
            continue
        plates.append({
            "text": text,
            "confidence": ocr_conf,
            "box": det["box"],
            "vehicle_class": "plate",
        })

    ms = (time.time() - t0) * 1000
    _record_alpr_result(slot_idx, plates, ms)

    if plates:
        log.info("ALPR(NPU) slot %d found %d plates: %s",
                 slot_idx, len(plates), [p["text"] for p in plates])
    return plates


def run_alpr_fallback(slot_idx, frame):
    """End-to-end fast-alpr on CPU — used when NPU plate detector is absent."""
    alpr = ctx.get("alpr")
    if alpr is None:
        return []

    t0 = time.time()
    plates = []
    try:
        results = alpr.predict(frame)
    except Exception as e:
        log.warning("ALPR fallback predict failed: %s", e)
        return []

    for r in results:
        det = r.detection
        ocr = r.ocr
        if not ocr:
            continue
        conf = ocr.confidence
        if isinstance(conf, (list, tuple)):
            avg_conf = sum(conf) / len(conf) if conf else 0.0
        else:
            avg_conf = float(conf) if conf else 0.0
        text = ocr.text.strip() if ocr.text else ""
        if not text or len(text) < 2:
            continue
        if det and det.bounding_box:
            bb = det.bounding_box
            box = [int(bb.x1), int(bb.y1), int(bb.x2), int(bb.y2)]
        else:
            box = [0, 0, 0, 0]
        plates.append({
            "text": text,
            "confidence": avg_conf,
            "box": box,
            "vehicle_class": "plate",
        })

    ms = (time.time() - t0) * 1000
    _record_alpr_result(slot_idx, plates, ms)

    if plates:
        log.info("ALPR(CPU fallback) slot %d found %d plates: %s",
                 slot_idx, len(plates), [p["text"] for p in plates])
    return plates


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------
class CaptureThread(threading.Thread):
    """Captures frames from a single source into one slot.

    Responsibilities:
      - Open and read the source (RTSP URL or local video file)
      - Push frames onto the shared `frame_queue` tagged with slot_idx
      - Maintain per-slot latest_frame + frame_buffer + FPS EMA
      - On error/EOF: mark the slot disconnected and retry forever with
        exponential backoff (1s → 2s → … capped at 30s)
    """

    RECONNECT_BASE = 1.0
    RECONNECT_CAP = 30.0

    def __init__(self, slot):
        super().__init__(daemon=True)
        self.slot = slot
        self.running = True
        self.frame_id = 0
        self.cap = None

    def _set_status(self, connected, err=""):
        self.slot["connected"] = connected
        self.slot["last_error"] = err

    def run(self):
        slot = self.slot
        backoff = self.RECONNECT_BASE
        while self.running:
            self.cap = cv2.VideoCapture(slot["url"])
            if not self.cap.isOpened():
                self._set_status(False, "cannot open source")
                log.warning("Slot %d (%s): cannot open %s — retry in %.0fs",
                            slot["idx"], slot["name"], slot["url"], backoff)
                # Sleep in small chunks so stop() is responsive
                self._sleep_interruptible(backoff)
                backoff = min(backoff * 2, self.RECONNECT_CAP)
                continue

            log.info("Slot %d (%s) capture started: %s",
                     slot["idx"], slot["name"], slot["url"])
            self._set_status(True, "")
            backoff = self.RECONNECT_BASE
            native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 15
            interval = 1.0 / native_fps

            while self.running:
                t0 = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    if slot["loop"]:
                        # Looping local video — restart from the top
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._set_status(False, "stream ended")
                    log.warning(
                        "Slot %d (%s): read returned False, reconnecting",
                        slot["idx"], slot["name"])
                    break

                self.frame_id += 1
                now = time.time()

                # Per-slot FPS EMA
                dt = now - slot.get("last_frame_time", 0.0)
                if slot.get("last_frame_time", 0.0) > 0 and dt > 0:
                    inst_fps = 1.0 / dt
                    prev = slot.get("fps", 0.0)
                    slot["fps"] = 0.8 * prev + 0.2 * inst_fps if prev else inst_fps
                slot["last_frame_time"] = now

                # Per-slot clip pre-roll buffer
                with slot["frame_buffer_lock"]:
                    slot["frame_buffer"].append(frame.copy())

                # Push onto shared detector queue tagged with slot_idx
                try:
                    frame_queue.put_nowait(
                        (slot["idx"], self.frame_id, frame, now))
                except queue.Full:
                    pass  # backpressure — drop rather than block

                elapsed = time.time() - t0
                if elapsed < interval:
                    time.sleep(interval - elapsed)

            if self.cap:
                self.cap.release()
                self.cap = None
            # Short grace before reopen attempt
            if self.running:
                self._sleep_interruptible(2.0)

    def _sleep_interruptible(self, seconds):
        """Sleep in small chunks so stop() causes a timely exit."""
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(min(0.2, end - time.time()))

    def stop(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# NPU detector thread (YOLOv8)
# ---------------------------------------------------------------------------
class DetectorThread(threading.Thread):
    def __init__(self, config):
        super().__init__(daemon=True)
        self.running = True
        det = config.get("detection", {})
        self.conf = det.get("confidence", 0.5)
        self.whitelist = set(det.get("classes", DEFAULT_WHITELIST))

    def run(self):
        ctx["ready_event"].wait()
        log.info("NPU detector thread started (device=%s)",
                 ctx["npu_device"])

        while self.running:
            try:
                slot_idx, fid, frame, ts = frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            t0 = time.time()
            blob, scale, pt, pl, oh, ow = preprocess_yolo(frame)
            with npu_guard(ctx.get("npu_device", "")):
                out = ctx["detector"]({0: blob})
            output = next(iter(out.values()))
            dets = postprocess_yolo(
                output, scale, pt, pl, oh, ow,
                conf=self.conf, whitelist=self.whitelist,
            )
            ms = (time.time() - t0) * 1000

            with telemetry_lock:
                telemetry["npu_latency_ms"] = ms
                telemetry["npu_active"] = True

            # Stash the latest detections on the slot so the UI's per-slot
            # card can read it without needing the annotated frame.
            slots[slot_idx]["detections"] = dets

            det_result_queue.put((slot_idx, fid, frame, dets, ts))

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Frame annotation
# ---------------------------------------------------------------------------
BOX_COLORS = {
    "person": (0, 200, 0),
    "car": (255, 165, 0),
    "truck": (255, 165, 0),
    "bus": (255, 165, 0),
    "motorcycle": (255, 165, 0),
    "bicycle": (200, 200, 0),
    "dog": (200, 0, 200),
    "cat": (200, 0, 200),
}
DEFAULT_COLOR = (0, 180, 255)


def annotate_frame(frame, detections, slot, global_telem):
    """Draw detections, plates, and a per-slot telemetry overlay.

    `slot` supplies per-camera fields (fps, plates, name). `global_telem`
    supplies fields shared across cameras (npu_latency_ms).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det["box"])
        color = BOX_COLORS.get(det["class_name"], DEFAULT_COLOR)
        label = f"{det['class_name']} {det['score']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1),
                      color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Plate annotations — per-slot plates (may be slightly stale vs frame)
    for plate in slot.get("plates", []):
        px1, py1, px2, py2 = (int(v) for v in plate["box"])
        color = (0, 255, 255)
        label = plate["text"]
        cv2.rectangle(out, (px1, py1), (px2, py2), color, 2)
        (tw, th), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (px1, py2), (px1 + tw + 4, py2 + th + 8),
                      color, -1)
        cv2.putText(out, label, (px1 + 2, py2 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Camera label — top-left
    name = slot.get("name") or f"slot {slot.get('idx', '?')}"
    (tw, th), _ = cv2.getTextSize(
        name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(out, (0, 0), (tw + 10, th + 10), (0, 0, 0), -1)
    cv2.putText(out, name, (5, th + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Telemetry bar — bottom
    bar_h = 36
    cv2.rectangle(out, (0, h - bar_h), (w, h), (0, 0, 0), -1)

    parts = [
        f"FPS: {slot.get('fps', 0):.1f}",
        f"NPU: {global_telem.get('npu_latency_ms', 0):.0f} ms",
    ]
    if global_telem.get("alpr_latency_ms", 0) > 0 and \
            global_telem.get("last_alpr_slot", -1) == slot.get("idx"):
        parts.append(
            f"ALPR: {global_telem.get('alpr_latency_ms', 0):.0f} ms")
    text = "   ".join(parts)
    cv2.putText(out, text, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# Pipeline / merger thread
# ---------------------------------------------------------------------------
class PipelineThread(threading.Thread):
    def __init__(self, db, alerter):
        super().__init__(daemon=True)
        self.db = db
        self.alerter = alerter
        self.running = True

    def run(self):
        ctx["ready_event"].wait()
        log.info("Pipeline thread started")

        while self.running:
            try:
                slot_idx, fid, frame, dets, ts = (
                    det_result_queue.get(timeout=1))
            except queue.Empty:
                continue

            slot = slots[slot_idx]
            camera = slot.get("name") or f"slot{slot_idx}"

            # Publish the raw frame + dets atomically. The MJPEG stream
            # generator will annotate on demand — this keeps pipeline work
            # O(1) per frame and matches annotation rate to UI rate.
            with slot["latest_frame_lock"]:
                slot["latest_frame"] = frame
                slot["latest_dets"] = dets

            # ALPR gating — honor runtime-selected mode (global, applies
            # to all slots)
            mode = ctx.get("alpr_mode", "gated")
            should_run_alpr = False
            if mode == "always":
                should_run_alpr = True
            elif mode == "gated":
                should_run_alpr = any(
                    d["class_name"] in VEHICLE_CLASSES for d in dets
                )

            if should_run_alpr:
                try:
                    alpr_queue.put_nowait((slot_idx, frame, dets, camera))
                except queue.Full:
                    pass

            # Per-detection bookkeeping
            clip_writer = slot.get("clip_writer")
            for det in dets:
                ts_iso = datetime.now().isoformat()
                self.db.insert(
                    timestamp=ts_iso,
                    cls=det["class_name"],
                    confidence=det["score"],
                    box=det.get("box_norm", [0, 0, 0, 0]),
                    scene_label="",
                    camera=camera,
                )
                event = {
                    "timestamp": ts_iso,
                    "class": det["class_name"],
                    "confidence": round(det["score"], 3),
                    "camera": camera,
                }
                with event_log_lock:
                    event_log.appendleft(event)

                if clip_writer:
                    clip_writer.trigger(det["class_name"])
                self.alerter.send(det["class_name"], det["score"], "")

            if clip_writer:
                clip_writer.add_frame(frame)

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# ALPR thread (runs asynchronously, doesn't block pipeline)
# ---------------------------------------------------------------------------
class ALPRThread(threading.Thread):
    DEDUP_WINDOW = 30.0   # suppress duplicate DB/event writes for this long

    def __init__(self, db):
        super().__init__(daemon=True)
        self.db = db
        self.running = True
        self.recent_plates = {}  # text → timestamp of last log

    def run(self):
        ctx["ready_event"].wait()
        # Prefer NPU plate detector; fall back to fast-alpr end-to-end
        use_npu = ctx.get("plate_detector") is not None
        if not use_npu and not ctx.get("alpr"):
            log.info("ALPR not available (no NPU detector, no fast-alpr), "
                     "thread exiting")
            return
        log.info("ALPR thread started (path=%s)",
                 "NPU+OCR" if use_npu else "CPU fallback (fast-alpr e2e)")

        while self.running:
            try:
                slot_idx, frame, dets, camera = alpr_queue.get(timeout=1)
            except queue.Empty:
                continue

            now = time.time()
            if use_npu:
                plate_reads = run_alpr_npu(slot_idx, frame)
            else:
                plate_reads = run_alpr_fallback(slot_idx, frame)

            for plate in plate_reads:
                text = plate["text"]

                # Dedup — don't spam DB with same plate every frame
                if text in self.recent_plates and \
                        now - self.recent_plates[text] < self.DEDUP_WINDOW:
                    continue
                self.recent_plates[text] = now

                ts_iso = datetime.now().isoformat()
                self.db.insert_plate(
                    timestamp=ts_iso,
                    plate_text=text,
                    confidence=plate["confidence"],
                    camera=camera,
                )
                event = {
                    "timestamp": ts_iso,
                    "class": f"PLATE: {text}",
                    "confidence": round(plate["confidence"], 3),
                    "camera": camera,
                }
                with event_log_lock:
                    event_log.appendleft(event)
                log.info("ALPR: %s (conf=%.2f)", text, plate["confidence"])

            # Clean old dedup entries
            cutoff = now - self.DEDUP_WINDOW
            self.recent_plates = {
                k: v for k, v in self.recent_plates.items() if v > cutoff
            }

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(title="SecurityCam")


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("/app/static/index.html").read_text()


@app.get("/status")
async def status():
    return {"ready": ctx["status"] == "ready", "status": ctx["status"]}


def _placeholder_frame(slot):
    """Generate a 640x360 BGR placeholder for an empty or disconnected slot."""
    h, w = 360, 640
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (24, 28, 38)  # matches UI bg

    if slot["source_type"] == "none":
        msg = "No camera configured"
        sub = f"slot {slot['idx']}"
    elif not slot["connected"]:
        msg = "Reconnecting…"
        sub = slot.get("last_error", "") or slot["name"]
    else:
        msg = "Waiting for video…"
        sub = slot["name"]

    (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.putText(img, msg, ((w - tw) // 2, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2)
    if sub:
        (sw, sh), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(img, sub, ((w - sw) // 2, h // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 140), 1)
    return img


@app.get("/stream/{slot_idx}")
async def stream(slot_idx: int):
    if slot_idx < 0 or slot_idx >= NUM_SLOTS:
        raise HTTPException(404, f"slot {slot_idx} out of range")
    slot = slots[slot_idx]

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
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n"
                       + buf.tobytes() + b"\r\n")
            time.sleep(0.033)
    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def _slot_public(slot):
    """Serializable subset of a slot for JSON endpoints."""
    return {
        "idx": slot["idx"],
        "name": slot["name"],
        "source_type": slot["source_type"],
        "connected": slot["connected"],
        "fps": round(slot.get("fps", 0.0), 1),
        "last_error": slot.get("last_error", ""),
        "detections_count": len(slot.get("detections", [])),
        "plates": [
            {"text": p["text"], "confidence": p["confidence"]}
            for p in slot.get("plates", [])
        ],
    }


@app.get("/api/telemetry")
async def api_telemetry():
    with telemetry_lock:
        glob = dict(telemetry)
    glob["source_mode"] = ctx.get("source_mode", "live")
    return {
        "slots": [_slot_public(s) for s in slots],
        "global": glob,
    }


@app.get("/api/events")
async def api_events():
    with event_log_lock:
        return list(event_log)


@app.get("/api/detections")
async def api_detections(limit: int = 50):
    return ctx["db"].recent(limit)


@app.get("/api/plates")
async def api_plates(limit: int = 50):
    return ctx["db"].recent_plates(limit)


@app.get("/api/cameras")
async def api_cameras():
    """Current slot assignments — used by UI to populate focus dropdown."""
    return {
        "slots": [_slot_public(s) for s in slots],
        "source_mode": ctx.get("source_mode", "live"),
        "source_modes": list(SOURCE_MODES),
    }


@app.get("/api/source_mode")
async def api_get_source_mode():
    return {
        "mode": ctx.get("source_mode", "live"),
        "modes": list(SOURCE_MODES),
    }


@app.post("/api/source_mode/{mode}")
async def api_set_source_mode(mode: str):
    if mode not in SOURCE_MODES:
        raise HTTPException(
            400, f"invalid mode {mode!r}; must be one of {SOURCE_MODES}")
    if mode == ctx.get("source_mode"):
        return {"status": "ok", "mode": mode, "note": "no change"}
    log.info("Switching source_mode %s → %s",
             ctx.get("source_mode", "?"), mode)
    stop_slot_captures()
    ctx["source_mode"] = mode
    assign_slots(mode)
    start_slot_captures()
    return {"status": "switched", "mode": mode}


@app.get("/api/alpr/mode")
async def api_get_alpr_mode():
    return {
        "mode": ctx.get("alpr_mode", "gated"),
        "modes": list(ALPR_MODES),
        "device": ctx.get("plate_detector_device", ""),
    }


@app.post("/api/alpr/mode/{mode}")
async def api_set_alpr_mode(mode: str):
    if mode not in ALPR_MODES:
        raise HTTPException(
            400, f"invalid mode {mode!r}; must be one of {ALPR_MODES}")
    ctx["alpr_mode"] = mode
    # Drain the ALPR queue and clear stale overlay so UI reacts immediately
    while True:
        try:
            alpr_queue.get_nowait()
        except queue.Empty:
            break
    for s in slots:
        s["plates"] = []
    with telemetry_lock:
        telemetry["alpr_mode"] = mode
        telemetry["alpr_latency_ms"] = 0.0
    log.info("ALPR mode set to %s", mode)
    return {"status": "ok", "mode": mode}


@app.get("/api/clips")
async def api_clips():
    clip_dir = Path(
        ctx["config"].get("clips", {}).get("output_dir", "/app/clips"))
    if not clip_dir.exists():
        return []
    clips = sorted(
        clip_dir.glob("*.mp4"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    return [{"filename": c.name, "size": c.stat().st_size}
            for c in clips[:50]]


# ---------------------------------------------------------------------------
# Model loading (background)
# ---------------------------------------------------------------------------
def load_models(model_dir):
    core = ov.Core()
    available = core.available_devices
    log.info("OpenVINO devices: %s", available)

    gpu_dev = "GPU" if "GPU" in available else "CPU"
    npu_dev = "NPU" if "NPU" in available else "CPU"
    ctx["gpu_device"] = gpu_dev
    ctx["npu_device"] = npu_dev

    # YOLOv8n on NPU
    ctx["status"] = f"loading:yolov8 ({npu_dev})"
    yolo_xml = Path(model_dir) / "yolov8n" / "yolov8n.xml"
    log.info("Loading YOLOv8 from %s for %s", yolo_xml, npu_dev)
    model = core.read_model(str(yolo_xml))
    ctx["detector"] = core.compile_model(model, npu_dev)
    log.info("YOLOv8 compiled on %s", npu_dev)

    # fast-alpr — always loaded; provides CCT OCR and end-to-end fallback
    ctx["status"] = "loading:alpr"
    try:
        from fast_alpr import ALPR
        ctx["alpr"] = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v2-global-model",
        )
        log.info("fast-alpr loaded (used for OCR + end-to-end fallback)")
    except Exception as e:
        log.warning("fast-alpr failed to load: %s", e)
        ctx["alpr"] = None

    # Plate detector on NPU (our own OpenVINO IR). Falls back to CPU if NPU
    # compile fails; if even that fails, ALPRThread uses fast-alpr end-to-end.
    ctx["status"] = f"loading:plate_detector ({npu_dev})"
    plate_xml = Path(model_dir) / "plate_detector" / "plate_detector.xml"
    if plate_xml.exists():
        try:
            pd_model = core.read_model(str(plate_xml))
            # Capture compiled input size for preprocess
            inp = pd_model.input(0)
            shape = list(inp.shape)  # [N, C, H, W]
            ctx["plate_detector_input"] = (int(shape[2]), int(shape[3]))
            try:
                ctx["plate_detector"] = core.compile_model(pd_model, npu_dev)
                ctx["plate_detector_device"] = npu_dev
                log.info("Plate detector compiled on %s (input %dx%d)",
                         npu_dev, shape[2], shape[3])
            except Exception as e:
                log.warning(
                    "Plate detector NPU compile failed (%s); "
                    "falling back to CPU", e)
                ctx["plate_detector"] = core.compile_model(pd_model, "CPU")
                ctx["plate_detector_device"] = "CPU"
                log.info("Plate detector compiled on CPU (fallback)")
        except Exception as e:
            log.warning(
                "Plate detector load failed (%s); "
                "ALPRThread will use fast-alpr end-to-end on CPU", e)
            ctx["plate_detector"] = None
            ctx["plate_detector_device"] = "CPU"
    else:
        log.warning(
            "Plate detector IR not found at %s; "
            "ALPRThread will use fast-alpr end-to-end on CPU", plate_xml)
        ctx["plate_detector"] = None
        ctx["plate_detector_device"] = "CPU"

    with telemetry_lock:
        telemetry["alpr_device"] = ctx["plate_detector_device"]

    ctx["status"] = "ready"
    ctx["ready_event"].set()
    log.info("All models loaded — pipeline ready")


# ---------------------------------------------------------------------------
# Slot assignment + capture lifecycle
# ---------------------------------------------------------------------------
def assign_slots(mode):
    """Rewrite the identity (name, url, source_type) of each slot for the
    given `mode`. Does NOT start/stop captures — caller handles that.

    - "live": slots[i] = cameras.yaml[i] for first NUM_SLOTS entries; rest empty.
    - "test": slots[i] = TEST_VIDEOS[i] (hardcoded list of 4 bundled videos).
    - "manual": slots[0] = override from ctx["manual_source"]; rest empty.
      (Used only by the --source debug flag.)
    """
    if mode == "manual":
        ms = ctx.get("manual_source") or {}
        for i, s in enumerate(slots):
            if i == 0 and ms:
                s["name"] = ms["name"]
                s["url"] = ms["url"]
                s["loop"] = ms.get("loop", False)
                s["source_type"] = "rtsp" if s["url"].startswith("rtsp") \
                    else "video"
            else:
                _clear_slot_identity(s)
        return

    if mode == "test":
        for i, s in enumerate(slots):
            if i < len(TEST_VIDEOS):
                name, path = TEST_VIDEOS[i]
                s["name"] = name
                s["url"] = path
                s["loop"] = True
                s["source_type"] = "video"
            else:
                _clear_slot_identity(s)
        return

    # "live" — fill from cameras.yaml
    cameras = ctx.get("cameras") or []
    if len(cameras) > NUM_SLOTS:
        log.warning("cameras.yaml has %d entries; only first %d used",
                    len(cameras), NUM_SLOTS)
    for i, s in enumerate(slots):
        if i < min(len(cameras), NUM_SLOTS):
            cam = cameras[i]
            s["name"] = cam["name"]
            s["url"] = cam["url"]
            s["loop"] = False
            s["source_type"] = "rtsp" if str(s["url"]).startswith("rtsp") \
                else "video"
        else:
            _clear_slot_identity(s)


def _clear_slot_identity(s):
    s["name"] = ""
    s["url"] = ""
    s["loop"] = False
    s["source_type"] = "none"
    s["connected"] = False
    s["last_error"] = ""
    s["fps"] = 0.0
    s["last_frame_time"] = 0.0
    s["detections"] = []
    s["plates"] = []
    # Leave latest_frame as-is so UI doesn't flicker; placeholder will be
    # served once the CaptureThread is gone and latest_frame is cleared.


def stop_slot_captures():
    """Stop every running CaptureThread and wait briefly for them to exit."""
    threads = []
    for s in slots:
        cap = s.get("capture")
        if cap:
            cap.stop()
            threads.append(cap)
            s["capture"] = None
    for t in threads:
        t.join(timeout=2.0)
    # Clear stale frames so the next mode shows fresh content
    for s in slots:
        with s["latest_frame_lock"]:
            s["latest_frame"] = None
            s["latest_dets"] = []
        with s["frame_buffer_lock"]:
            s["frame_buffer"].clear()
    # Drain pipeline queues of any in-flight frames from the old mode
    for q in (frame_queue, det_result_queue, alpr_queue):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break


def start_slot_captures():
    """Spawn a CaptureThread for every slot that has a source_type != 'none'."""
    for s in slots:
        if s["source_type"] == "none" or not s["url"]:
            continue
        cap = CaptureThread(s)
        cap.start()
        s["capture"] = cap


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Security Camera AI Pipeline")
    p.add_argument("--config", default="/app/config.yaml")
    p.add_argument("--cameras", default="/app/cameras.yaml")
    p.add_argument("--models", default="/models")
    p.add_argument("--source", default=None,
                   help="Single source override — fills slot 0, others empty "
                        "(debug only)")
    p.add_argument("--test", action="store_true",
                   help="Start in test source mode (4 bundled videos)")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    # Config
    config = {}
    if Path(args.config).exists():
        config = yaml.safe_load(Path(args.config).read_text()) or {}
    ctx["config"] = config

    # ALPR mode default from config; runtime-toggled via /api/alpr/mode
    alpr_default = config.get("alpr", {}).get("mode", "gated")
    if alpr_default not in ALPR_MODES:
        log.warning("Invalid alpr.mode %r in config; using 'gated'",
                    alpr_default)
        alpr_default = "gated"
    ctx["alpr_mode"] = alpr_default
    with telemetry_lock:
        telemetry["alpr_mode"] = alpr_default

    # Cameras
    cameras = []
    if Path(args.cameras).exists():
        cam_data = yaml.safe_load(Path(args.cameras).read_text()) or {}
        cameras = cam_data.get("cameras", [])
    ctx["cameras"] = cameras

    # Determine initial source mode.
    if args.source:
        # Single-slot debug override — uses the "manual" pseudo-mode.
        loop = not args.source.startswith("rtsp")
        ctx["manual_source"] = {
            "name": "manual", "url": args.source, "loop": loop}
        initial_mode = "manual"
    elif args.test:
        initial_mode = "test"
    else:
        default = config.get("source_mode", "live")
        if default not in SOURCE_MODES:
            log.warning("Invalid source_mode %r in config; using 'live'",
                        default)
            default = "live"
        initial_mode = default
    ctx["source_mode"] = initial_mode if initial_mode in SOURCE_MODES \
        else "live"

    # Slot state + per-slot ClipWriter
    init_slots()
    db = DetectionDB(config.get("database", "/app/data/detections.db"))
    ctx["db"] = db
    clip_cfg = config.get("clips", {})
    clip_out = clip_cfg.get("output_dir", "/app/clips")
    pre_roll = clip_cfg.get("pre_roll", 5)
    post_roll = clip_cfg.get("post_roll", 10)
    for s in slots:
        s["clip_writer"] = ClipWriter(
            slot=s, output_dir=clip_out,
            pre_roll=pre_roll, post_roll=post_roll)

    alerter = Alerter(config.get("alerts", {}))

    # Background model loading — starts the NPU/CPU compile work, then sets
    # ctx["ready_event"]. Inference threads block on that event.
    threading.Thread(
        target=load_models, args=(args.models,), daemon=True,
    ).start()

    # Shared worker threads (one each; they multiplex across slots)
    DetectorThread(config).start()
    PipelineThread(db, alerter).start()
    ALPRThread(db).start()

    assign_slots(initial_mode)
    start_slot_captures()

    # Sanity: warn if we ended up with no sources at all
    live_slots = [s for s in slots if s["source_type"] != "none"]
    if not live_slots:
        log.warning("No sources configured in mode %r. Use --test, --source, "
                    "or add entries to cameras.yaml.", initial_mode)

    log.info("Web UI on port %d; source_mode=%s; slots=%s",
             args.port, initial_mode,
             [(s["idx"], s["name"] or "-") for s in slots])
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
