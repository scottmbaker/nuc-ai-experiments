#!/usr/bin/env python3
"""Export models to OpenVINO IR format.

Run during Docker build (model-builder stage).
"""

import os
from pathlib import Path

from ultralytics import YOLO

MODELS_DIR = Path("/models")

# fast-alpr plate detector input size (model name: yolo-v9-t-384-...)
PLATE_DETECTOR_SIZE = 384


def export_yolov8():
    out = MODELS_DIR / "yolov8n"
    out.mkdir(parents=True, exist_ok=True)
    print("Exporting YOLOv8n to OpenVINO IR (INT8, static shapes for NPU)...")
    model = YOLO("yolov8n.pt")
    model.export(format="openvino", int8=True, imgsz=640, data="coco128.yaml")
    src = Path("yolov8n_int8_openvino_model")
    if not src.exists():
        src = Path("yolov8n_openvino_model")
    for f in src.iterdir():
        f.rename(out / f.name)
    print(f"YOLOv8n exported to {out}")


def _find_detector_onnx(alpr):
    """Locate the plate detector ONNX file after fast-alpr has downloaded it.

    fast-alpr's internals are not a public API — we check several likely
    attribute names, and fall back to scanning its cache dirs.
    """
    # 1) Ask the detector for the ONNX path by common attribute names
    det = getattr(alpr, "detector", None)
    if det is not None:
        for attr in ("model_path", "_model_path", "onnx_path",
                     "_onnx_path", "path"):
            val = getattr(det, attr, None)
            if isinstance(val, (str, Path)) and str(val).endswith(".onnx"):
                p = Path(val)
                if p.exists():
                    return p

    # 2) Scan common cache locations for the expected ONNX name.
    # fast-alpr's HF repo name encodes 'yolo-v9-t-384'; match loosely.
    search_roots = [
        Path(os.path.expanduser("~/.cache")),
        Path("/root/.cache"),
        Path("/tmp"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*.onnx"):
            name = p.name.lower()
            if "yolo-v9-t-384" in name or (
                    "plate" in name and "ocr" not in name):
                return p

    raise RuntimeError(
        "Could not locate fast-alpr plate-detector ONNX after ALPR init. "
        "Tried detector attributes and ~/.cache scan.")


def export_plate_detector():
    """Convert fast-alpr's plate detector ONNX to OpenVINO IR.

    Uses ov.convert_model (not Core/read_model) to avoid initializing the
    OpenVINO runtime plugins inside buildkit — we previously hit SIGBUS
    doing that here. Output is pinned to static [1, 3, 384, 384] so the
    NPU can compile it.
    """
    import openvino as ov
    from fast_alpr import ALPR

    out = MODELS_DIR / "plate_detector"
    out.mkdir(parents=True, exist_ok=True)

    print("Loading fast-alpr to trigger plate detector ONNX download...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    onnx_path = _find_detector_onnx(alpr)
    print(f"Found plate detector ONNX: {onnx_path}")

    print(f"Converting to OpenVINO IR with static "
          f"[1, 3, {PLATE_DETECTOR_SIZE}, {PLATE_DETECTOR_SIZE}]...")
    model = ov.convert_model(str(onnx_path))

    # Log what the ONNX looked like before reshape, for diagnosis
    print("  ONNX inputs:")
    for inp in model.inputs:
        print(f"    {inp.get_any_name()!r} "
              f"shape={inp.partial_shape} dtype={inp.element_type}")
    print("  ONNX outputs:")
    for o in model.outputs:
        print(f"    {o.get_any_name()!r} "
              f"shape={o.partial_shape} dtype={o.element_type}")

    model.reshape([1, 3, PLATE_DETECTOR_SIZE, PLATE_DETECTOR_SIZE])
    ov.save_model(model, str(out / "plate_detector.xml"))
    print(f"Plate detector exported to {out}")


def download_alpr_models():
    """Pre-download fast-alpr ONNX models so they're cached in the image.

    export_plate_detector() also calls ALPR(...) which triggers downloads,
    but calling this explicitly up front makes the build log clearer and
    ensures the OCR model is cached even if plate-detector export fails.
    """
    from fast_alpr import ALPR
    print("Pre-downloading ALPR models (YOLOv9 plate detector + CCT OCR)...")
    ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )
    print("ALPR models downloaded and cached")


if __name__ == "__main__":
    export_yolov8()
    download_alpr_models()
    export_plate_detector()
