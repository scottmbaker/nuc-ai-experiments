#!/usr/bin/env bash
# Smoke test: tool-calling round-trip against OVMS.
#
# Sends a chat completion with a single function tool. A correctly-wired
# server should respond with a `tool_calls` field selecting `read_file`
# and emitting JSON arguments. This is the make-or-break check before
# wiring up OpenCode.
#
# Usage:  bash smoke/tools.sh http://localhost:30083
set -euo pipefail

ENDPOINT="${1:-http://localhost:30083}"
MODEL="${MODEL:-qwen3-coder}"

echo "POST ${ENDPOINT}/v3/chat/completions  (tools, model: ${MODEL})"

curl -fsS "${ENDPOINT}/v3/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<'JSON'
{
  "model": "qwen3-coder",
  "messages": [
    {"role": "system", "content": "You are a coding assistant. Use the provided tools when they help answer the user's request."},
    {"role": "user", "content": "Read the file /etc/hostname and tell me what it contains."}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk and return them as a string.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Absolute path to the file"
            }
          },
          "required": ["path"]
        }
      }
    }
  ],
  "tool_choice": "auto",
  "temperature": 0,
  "max_tokens": 256
}
JSON
)" | tee /tmp/code-assistant-tools.json

echo
echo "---"
python3 -c "
import json
data = json.load(open('/tmp/code-assistant-tools.json'))
msg = data['choices'][0]['message']
calls = msg.get('tool_calls') or []
if calls:
    for c in calls:
        fn = c.get('function', {})
        print('tool_call:', fn.get('name'), 'args:', fn.get('arguments'))
else:
    print('NO tool_calls field — model returned plain text:')
    print(msg.get('content', ''))
    print()
    print('This is the failure mode we worry about. If OVMS does not parse')
    print('tool calls correctly, OpenCode will not work. Check graph.pbtxt')
    print('tool_parser and the OVMS log output.')
"
