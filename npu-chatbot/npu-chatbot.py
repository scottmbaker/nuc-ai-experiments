#!/usr/bin/env python3
"""
Interactive streaming chatbot using OpenVINO GenAI on Intel NPU.

Usage:
    python3 npu-chatbot.py --model /models/phi-3-mini-npu
    python3 npu-chatbot.py --model /models/phi-3-mini-npu --device CPU  # fallback to CPU

First run on NPU will be slow (model compilation). Subsequent runs use cached blobs.
"""

import argparse
import sys
import os
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Interactive LLM chatbot on Intel NPU")
    parser.add_argument("--model", required=True, help="Path to OpenVINO model directory")
    parser.add_argument("--device", default="NPU", choices=["NPU", "CPU", "GPU"],
                        help="Inference device (default: NPU)")
    parser.add_argument("--max-prompt-len", type=int, default=2048,
                        help="Max prompt length in tokens (NPU only, default: 2048)")
    parser.add_argument("--min-response-len", type=int, default=512,
                        help="Reserved response tokens (NPU only, default: 512)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Max tokens to generate per response (default: 1024)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Nucleus sampling threshold (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling (default: 50)")
    parser.add_argument("--repetition-penalty", type=float, default=1.1,
                        help="Repetition penalty (default: 1.1)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy decoding instead of sampling")
    parser.add_argument("--system", type=str,
                        default="You are a helpful, concise assistant.",
                        help="System prompt")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable NPU blob caching")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check model path exists
    if not os.path.isdir(args.model):
        print(f"Error: Model directory not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    try:
        import openvino_genai
    except ImportError:
        print("Error: openvino-genai is not installed.", file=sys.stderr)
        print("Install it with: pip3 install openvino-genai", file=sys.stderr)
        sys.exit(1)

    # Token counter for stats
    token_count = 0
    gen_start = 0

    def streamer(subword):
        nonlocal token_count
        token_count += 1
        print(subword, end="", flush=True)
        return openvino_genai.StreamingStatus.RUNNING

    # Build pipeline kwargs
    pipeline_kwargs = {}
    if args.device == "NPU":
        pipeline_kwargs["max_prompt_len"] = args.max_prompt_len
        pipeline_kwargs["min_response_len"] = args.min_response_len
        if not args.no_cache:
            blob_dir = os.path.join(args.model, "npu_blobs")
            os.makedirs(blob_dir, exist_ok=True)
            pipeline_kwargs["blob_path"] = blob_dir

    # Load model
    print(f"Loading model from {args.model} on {args.device}...")
    print("(First run on NPU may take several minutes for compilation)")
    load_start = time.time()

    try:
        pipe = openvino_genai.LLMPipeline(args.model, args.device, **pipeline_kwargs)
    except Exception as e:
        print(f"\nError loading model: {e}", file=sys.stderr)
        if args.device == "NPU":
            print("Tip: Try --device CPU to verify the model works before using NPU", file=sys.stderr)
        sys.exit(1)

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Configure generation
    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = args.max_new_tokens
    config.repetition_penalty = args.repetition_penalty

    if args.greedy:
        config.do_sample = False
    else:
        config.do_sample = True
        config.temperature = args.temperature
        config.top_p = args.top_p
        config.top_k = args.top_k

    # Start chat session
    pipe.start_chat(args.system)

    print()
    print("=" * 60)
    print(f"  NPU Chatbot  |  Device: {args.device}")
    print(f"  Model: {os.path.basename(args.model)}")
    print(f"  Type 'quit' to exit, 'reset' for new conversation")
    print("=" * 60)
    print()

    while True:
        try:
            prompt = input("\033[1;32mYou:\033[0m ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        prompt = prompt.strip()
        if not prompt:
            continue

        if prompt.lower() in ("quit", "exit"):
            break

        if prompt.lower() == "reset":
            pipe.finish_chat()
            pipe.start_chat(args.system)
            print("[Conversation reset]\n")
            continue

        # Generate response with streaming
        print(f"\033[1;34mAssistant:\033[0m ", end="", flush=True)
        token_count = 0
        gen_start = time.time()

        try:
            pipe.generate(prompt, config, streamer=streamer)
        except Exception as e:
            print(f"\n\nGeneration error: {e}", file=sys.stderr)
            if "prompt" in str(e).lower() and "exceed" in str(e).lower():
                print("Conversation too long for NPU context. Type 'reset' to start fresh.",
                      file=sys.stderr)
            continue

        gen_time = time.time() - gen_start
        tokens_per_sec = token_count / gen_time if gen_time > 0 else 0

        print(f"\n\033[0;37m[{token_count} tokens, {tokens_per_sec:.1f} t/s, {gen_time:.1f}s]\033[0m\n")

    # Cleanup
    pipe.finish_chat()
    print("Goodbye!")


if __name__ == "__main__":
    main()
