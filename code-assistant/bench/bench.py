#!/usr/bin/env python3
"""Feasibility benchmark for the code-assistant LLM on Intel iGPU.

Measures:
  - load + compile time
  - peak resident memory
  - time-to-first-token (TTFT) for a representative prompt
  - decode throughput (tok/s) over a fixed output budget

Run before committing to OVMS / Helm scaffolding. If decode tok/s on the iGPU
is below ~10, the 30B-A3B model is not viable here and we should fall back to
Qwen2.5-Coder-14B Instruct INT4 dense.
"""

from __future__ import annotations

import argparse
import os
import resource
import statistics
import sys
import time
from pathlib import Path

try:
    import openvino_genai as ov_genai
except ImportError:
    print("openvino-genai not installed. Install with:", file=sys.stderr)
    print("  pip install openvino-genai", file=sys.stderr)
    sys.exit(2)


PROMPTS = {
    "short": "Write a Python function that returns the nth Fibonacci number.",
    "medium": (
        "Refactor the following code to use list comprehensions and explain "
        "your changes:\n\n"
        "def squares(n):\n"
        "    out = []\n"
        "    for i in range(n):\n"
        "        out.append(i * i)\n"
        "    return out\n"
    ),
    "long": (
        "You are a senior engineer reviewing a pull request. The diff below "
        "modifies a request handler to add caching. Identify any bugs, "
        "concurrency issues, or violations of the project's existing "
        "conventions, and propose a corrected version.\n\n"
        + ("# placeholder code line\n" * 80)
    ),
}


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns KB, macOS returns bytes — assume KB on Linux NUC.
    return usage / 1024.0


def run(model_dir: Path, device: str, max_new_tokens: int, prompt_key: str,
        warmup: int, iters: int) -> dict:
    print(f"Loading {model_dir} on {device}...", flush=True)
    t0 = time.perf_counter()
    pipe = ov_genai.LLMPipeline(str(model_dir), device)
    load_s = time.perf_counter() - t0
    print(f"  load+compile: {load_s:.2f}s  rss: {peak_rss_mb():.0f} MB",
          flush=True)

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    cfg.temperature = 0.0

    prompt = PROMPTS[prompt_key]

    # Warmup — first run pays a kernel-cache cost we don't want to measure.
    for i in range(warmup):
        print(f"warmup {i+1}/{warmup}...", flush=True)
        pipe.generate(prompt, cfg)

    ttfts: list[float] = []
    decode_tps: list[float] = []
    total_tokens = 0

    for i in range(iters):
        t_start = time.perf_counter()
        first_token_at = {"t": None}
        token_count = {"n": 0}

        def streamer(piece: str) -> bool:
            if first_token_at["t"] is None:
                first_token_at["t"] = time.perf_counter() - t_start
            token_count["n"] += 1
            return False  # don't stop early

        pipe.generate(prompt, cfg, streamer)
        elapsed = time.perf_counter() - t_start

        ttft = first_token_at["t"] or elapsed
        decoded = max(token_count["n"] - 1, 1)
        decode_time = max(elapsed - ttft, 1e-6)
        tps = decoded / decode_time

        ttfts.append(ttft)
        decode_tps.append(tps)
        total_tokens += token_count["n"]
        print(f"  iter {i+1}/{iters}: ttft={ttft*1000:.0f}ms "
              f"decode={tps:.1f} tok/s tokens={token_count['n']}",
              flush=True)

    return {
        "model_dir": str(model_dir),
        "device": device,
        "prompt": prompt_key,
        "max_new_tokens": max_new_tokens,
        "load_s": load_s,
        "peak_rss_mb": peak_rss_mb(),
        "ttft_ms_mean": statistics.mean(ttfts) * 1000,
        "ttft_ms_p50": statistics.median(ttfts) * 1000,
        "decode_tps_mean": statistics.mean(decode_tps),
        "decode_tps_p50": statistics.median(decode_tps),
        "iters": iters,
        "total_tokens": total_tokens,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path,
                   help="Path to exported OpenVINO model directory")
    p.add_argument("--device", default="GPU", choices=("CPU", "GPU", "NPU"))
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--prompt", default="medium",
                   choices=tuple(PROMPTS.keys()))
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    args = p.parse_args()

    if not args.model.exists():
        print(f"model dir does not exist: {args.model}", file=sys.stderr)
        return 1

    result = run(args.model, args.device, args.max_new_tokens, args.prompt,
                 args.warmup, args.iters)

    print()
    print("===== summary =====")
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Gate from the plan: < 10 tok/s decode means we should fall back.
    if result["decode_tps_mean"] < 10.0:
        print()
        print("WARN: decode throughput below the 10 tok/s gate "
              "documented in PLAN.md")
        return 3

    return 0


if __name__ == "__main__":
    sys.exit(main())
