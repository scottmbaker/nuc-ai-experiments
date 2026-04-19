#!/usr/bin/env python3
"""Extract a speaker x-vector embedding from the CMU Arctic dataset and save as .bin."""
import sys
import numpy as np

speaker_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 7306
out_path = sys.argv[2] if len(sys.argv) > 2 else "speaker.bin"

from datasets import load_dataset
ds = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
embedding = np.array(ds[speaker_idx]["xvector"], dtype=np.float32)
print(f"Speaker {speaker_idx}: embedding shape={embedding.shape}, "
      f"range=[{embedding.min():.4f}, {embedding.max():.4f}]")
embedding.tofile(out_path)
print(f"Saved {out_path} ({embedding.nbytes} bytes)")
