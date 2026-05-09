# OpenCode client setup

OpenCode is a Claude-Code-style agentic coder; we point it at the local
OVMS endpoint running on the NUC.

## Install

```powershell
# Windows / npm (cleanest)
npm install -g opencode-ai
```

```bash
# macOS / Linux / Git Bash / WSL
curl -fsSL https://opencode.ai/install | bash
```

## Configure

Two config-file locations work; pick one:

- **Project-scoped** (recommended): `opencode.json` at the root of the
  project you'll work in. Doesn't pollute global state.
- **User-scoped**:
  `%USERPROFILE%\.config\opencode\config.json` on Windows,
  `~/.config/opencode/config.json` on macOS/Linux.

[`config.example.json`](config.example.json) is a ready-to-edit template.
Replace `NUC_IP` with the k3s-node address:

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

`apiKey` is required by the OpenAI-compatible provider but OVMS does
not validate it — any non-empty string works.

## Run

```bash
cd <some project>
opencode
```

OpenCode's bottom-right model indicator should read
`ovms-local/qwen3-coder`. First request from a cold pod: TTFT ~150 ms,
decode 19–36 tok/s. Plan/agent modes drive tool calls (read files, write
diffs, run commands) which round-trip cleanly through OVMS's
`qwen3coder` parser.

## Verify before launching

```bash
curl http://<NUC_IP>:30083/v3/models
```

Should return a JSON list with `qwen3-coder`. If it times out, port
30083 isn't reachable from the workstation — `kubectl get svc
code-assistant` to confirm the NodePort, and check firewalls.

## Known quirks

- **OVMS path is `/v3` not `/v1`.** The `baseURL` in the config includes
  `/v3` to match OVMS's OpenAI-compatible endpoint root. OpenAI clients
  that hard-code `/v1` will need a small reverse-proxy.
- **Tool-call streaming.** OVMS emits Qwen-Coder-style tool calls; the
  `tool_parser: "qwen3coder"` line in [`graph.pbtxt`](../graph.pbtxt) is
  what makes this work. If OpenCode reports malformed function calls
  after a model swap, start there.
- **Context limit.** Capped at 32 k tokens in the config above. The
  model natively supports more, but iGPU shared memory + KV cache make
  32 k the realistic working ceiling. Raise cautiously and watch
  `kubectl top pod`.
- **Single user.** OVMS is configured for `max_num_seqs: 256` but the
  iGPU has finite memory; running multiple concurrent OpenCode sessions
  against this endpoint will likely OOM. One workstation at a time.
