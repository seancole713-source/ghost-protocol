#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-http://127.0.0.1:5000}
AUTH="Authorization: Bearer ${GHOST_API_TOKEN:-}"
fail=$(curl -s -H "$AUTH" "$HOST/source/status" | jq '.sources|to_entries|map(select(.value.ok==false))|length')
echo "$fail"