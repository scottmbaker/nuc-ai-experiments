#!/usr/bin/env python3
"""Extract openWakeWord ONNX models from pip package to a target directory."""
import os, shutil, sys
import openwakeword

src = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
dst = sys.argv[1] if len(sys.argv) > 1 else "/models/wakeword"
os.makedirs(dst, exist_ok=True)

for f in ["melspectrogram.onnx", "embedding_model.onnx"]:
    shutil.copy(os.path.join(src, f), os.path.join(dst, f))
    sz = os.path.getsize(os.path.join(dst, f)) // 1024
    print(f"Copied {f} ({sz} KB)")
