#!/usr/bin/env bash
# Smoke test: list of models served by OVMS.
# Usage:  bash smoke/models.sh http://localhost:30083
set -euo pipefail

ENDPOINT="${1:-http://localhost:30083}"

echo "GET ${ENDPOINT}/v3/models"
curl -fsS "${ENDPOINT}/v3/models" | python3 -m json.tool

echo
echo "GET ${ENDPOINT}/v2/health/ready"
curl -fsS "${ENDPOINT}/v2/health/ready" -w "  http %{http_code}\n" || true
