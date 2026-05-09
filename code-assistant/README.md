# Code Assistant

Local agentic coding assistant on an Intel NUC (Panther Lake or similar).
A Claude-Code-equivalent CLI experience served entirely from the iGPU —
no outbound network calls for inference.

## How it works

| Component | Device | Model |
|-----------|--------|-------|
| LLM (chat + tool calls) | iGPU | Qwen3-Coder-30B-A3B-Instruct INT4 (default) |

The model is a Mixture-of-Experts: 30.5 B total parameters with ~3.3 B
active per token. INT4 weights occupy ~16 GB; the iGPU shares system RAM
so peak resident is ~17 GB.

Inference is served by **OpenVINO Model Server (OVMS) 2026.1**, which
exposes an OpenAI-compatible REST API at `/v3/chat/completions`. Tool
calling is handled by OVMS's `qwen3coder` parser — `tool_calls[]` round
trips cleanly to clients like OpenCode, Cline, and Continue.

Pipeline:

```
[Workstation: opencode]
   → HTTP /v3/chat/completions (OpenAI-compatible)
   → [k3s pod: code-assistant on NUC]
       → OVMS — continuous batching, INT8 KV cache,
         embedded OpenVINO runtime → iGPU
       → streamed tokens / tool_calls back to client
```

## Requirements

- Intel NUC (or similar) with integrated GPU (Panther Lake tested,
  device id `0xb0a0`)
- k3s with the Intel GPU device plugin (`gpu.intel.com/xe`)
- Helm, `buildkitd`, `nerdctl`
- ≥ 32 GB available to the workload (full system RAM or LXC cgroup limit)
- ≥ 25 GB free disk on the build node
- A privileged LXC container if running k3s under Proxmox — required for
  the overlayfs snapshotter
- **No other workload holding the iGPU** while serving. The helm chart
  requests `gpu.intel.com/xe: 1`, and k3s gates pods on it.
  Stop any other AI pod that requests it before deploying this one.

## Quick start

```bash
# On the node:
cd ~/code-assistant
make build MODEL=qwen3-coder-30b   # ~3-5 min: model download + image bake
make deploy MODEL=qwen3-coder-30b  # ~30s once image is local; first GPU compile is ~13s
make smoke                          # verifies chat + tool calls
```

Once `make deploy` finishes, the OpenAI-compatible endpoint is live at
`http://<NUC_IP>:30083/v3`. Plug a coding-agent client at it (next
section).

## Build options

```
make build [MODEL=qwen3-coder-30b]
```

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `qwen3-coder-30b` | Model name (see `make models`) |

List available models:

```bash
make models
```

The model is downloaded from HuggingFace's OpenVINO org at build time
and baked into the image. First build takes ~3–5 min on a fast connection;
subsequent builds reuse Buildkit's cache.

## Deploy options

```
make deploy [MODEL=qwen3-coder-30b]
```

The pod is exposed on:
- **NodePort 30083** — REST (OpenAI-compatible) — `http://<NUC_IP>:30083/v3`
- **NodePort 30093** — gRPC

## Connecting OpenCode

OpenCode is a terminal-based agentic coder (Claude-Code-style). It speaks
the OpenAI-compatible API natively, so it works against this OVMS endpoint
without modification.

### 1. Install

```powershell
# Windows / npm (cleanest)
npm install -g opencode-ai

# macOS / Linux / Git Bash / WSL
curl -fsSL https://opencode.ai/install | bash
```

### 2. Configure

Drop a config file at one of these locations:

- **Project-scoped** (recommended): `opencode.json` at the root of the
  project you're working in.
- **User-scoped**: `%USERPROFILE%\.config\opencode\config.json` on
  Windows, `~/.config/opencode/config.json` on macOS/Linux.

Contents (replace `<NUC_IP>` with the k3s-node address):

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ovms-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Local OVMS (NUC)",
      "options": {
        "baseURL": "http://<NUC_IP>:30083/v3",
        "apiKey": "unused"
      },
      "models": {
        "qwen3-coder": {
          "name": "Qwen3-Coder-30B-A3B (NUC iGPU)",
          "limit": { "context": 32768, "output": 8192 }
        }
      }
    }
  },
  "model": "ovms-local/qwen3-coder"
}
```

`apiKey` must be present (OpenAI-compatible adapter requires it) but its
value is ignored by OVMS — any non-empty string works.

### 3. Run

```bash
cd <some project>
opencode
```

The bottom-right model indicator should read `ovms-local/qwen3-coder`.
First request from a cold pod: TTFT ~150 ms, decode 19–36 tok/s. Tool
calls work — plan/agent modes can read files, write diffs, run commands.

### Troubleshooting

- **Endpoint reachable?** From the workstation:
  `curl http://<NUC_IP>:30083/v3/models` should list `qwen3-coder`.
  If it times out, port 30083 isn't reachable — verify
  `kubectl get svc code-assistant` shows the NodePort.
- **Path is `/v3` not `/v1`.** OVMS uses `/v3` for the OpenAI-compatible
  root. Keep `/v3` in `baseURL` — clients that hard-code `/v1` will need
  a small reverse-proxy.
- **Context ceiling.** Capped at 32 k in the config above. Qwen3-Coder
  natively supports more, but iGPU shared memory is the bottleneck.
  Raise cautiously and watch `kubectl top pod`.

See [opencode/README.md](opencode/README.md) for additional notes.

## Other clients

Any OpenAI-compatible client works once you point its `baseURL` at
`http://<NUC_IP>:30083/v3` with any non-empty API key:

- **Cline** (VSCode extension) — agentic, edits files via tool calls
- **Continue.dev** (VSCode extension) — chat plus tab autocomplete (the
  autocomplete side wants a smaller second model on a separate endpoint,
  not configured here)
- **Aider** (CLI) — `aider --openai-api-base http://<NUC_IP>:30083/v3
  --model qwen3-coder`

## Endpoints

| Path | Use |
|------|-----|
| `/v3/chat/completions` | OpenAI-compatible chat / tool calls |
| `/v3/models` | List served models |
| `/v2/health/ready` | Liveness check (used by the pod's readinessProbe) |

The smoke scripts under `smoke/` exercise these directly:

```bash
bash smoke/chat.sh    http://<NUC_IP>:30083  # plain chat
bash smoke/tools.sh   http://<NUC_IP>:30083  # tool-calling round trip
bash smoke/models.sh  http://<NUC_IP>:30083  # /models + health
```

## Operations

```bash
make logs        # stream pod logs
make status      # pod state, GPU resource availability, built images
make undeploy    # remove the Helm release and pod
make nuke        # clean up orphaned containerd state for this pod
make clean       # remove built images
```

## Available models

The 30B is the verified working configuration. The 14B and 7B entries
are listed as smaller alternatives in case you want to trade quality
for speed or footprint; they should work but haven't been exercised
end-to-end here.

| Name | HF Repo | Size (INT4) | Notes |
|------|---------|-------------|-------|
| `qwen3-coder-30b` (default) | `OpenVINO/Qwen3-Coder-30B-A3B-Instruct-int4-ov` | ~16 GB | MoE — 3.3 B active params; tuned by Qwen for agentic tool use |
| `qwen25-coder-14b` | `OpenVINO/Qwen2.5-Coder-14B-Instruct-int4-ov` | ~8 GB | Dense alternative |
| `qwen25-coder-7b` | `OpenVINO/Qwen2.5-Coder-7B-Instruct-int4-ov` | ~4 GB | Smaller alternative |

Add new entries in [models.conf](models.conf) using the format
`NAME|HF_MODEL_ID|MODEL_DIR`.

See [opencode/README.md](opencode/README.md) for OpenCode-specific setup notes.
