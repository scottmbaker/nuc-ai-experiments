#!/usr/bin/env bash
# Smoke test: simple chat completion against OVMS.
# Usage:  bash smoke/chat.sh http://localhost:30083
set -euo pipefail

ENDPOINT="${1:-http://localhost:30083}"
MODEL="${MODEL:-qwen3-coder}"

echo "POST ${ENDPOINT}/v3/chat/completions  (model: ${MODEL})"

curl -fsS "${ENDPOINT}/v3/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$(cat <<JSON
{
  "model": "${MODEL}",
  "messages": [
    {"role": "system", "content": "You are a terse coding assistant."},
    {"role": "user", "content": "In one sentence, what does this Python function do?\n\ndef f(xs):\n    return sum(x*x for x in xs)"}
  ],
  "temperature": 0,
  "max_tokens": 128
}
JSON
)" | tee /tmp/code-assistant-chat.json

echo
echo "---"
python3 -c "
import json, sys
data = json.load(open('/tmp/code-assistant-chat.json'))
content = data['choices'][0]['message']['content']
print('assistant:', content.strip())
print('tokens:', data.get('usage', {}))
"
