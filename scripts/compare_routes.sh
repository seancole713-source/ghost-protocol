#!/usr/bin/env bash
# Compare local vs production OpenAPI route counts and diff path presence
set -euo pipefail
BASE_LOCAL="http://127.0.0.1:${PORT:-8444}"
BASE_PROD="https://web-production-8e9a0.up.railway.app"

fetch_paths() {
  curl -s "$1/openapi.json" | python3 - <<'PY'
import json,sys
try:
    data=json.load(sys.stdin)
    for p in sorted(data.get('paths',{}).keys()):
        print(p)
except Exception:
    pass
PY
}

echo "🔎 Fetching route lists..."
local_paths=$(fetch_paths "$BASE_LOCAL" || true)
prod_paths=$(fetch_paths "$BASE_PROD" || true)

lc=$(printf "%s\n" "$local_paths" | sed '/^$/d' | wc -l | tr -d ' ')
pc=$(printf "%s\n" "$prod_paths" | sed '/^$/d' | wc -l | tr -d ' ')

echo "Local (${BASE_LOCAL}) routes:    ${lc:-0}"
echo "Prod  (${BASE_PROD}) routes:    ${pc:-0}"

if [ -n "$local_paths" ] && [ -n "$prod_paths" ]; then
  echo "\n➕ In local but NOT in prod:"
  comm -23 <(printf "%s\n" "$local_paths" | sort) <(printf "%s\n" "$prod_paths" | sort) || true
  echo "\n➖ In prod but NOT in local:"
  comm -13 <(printf "%s\n" "$local_paths" | sort) <(printf "%s\n" "$prod_paths" | sort) || true
else
  echo "⚠️  Could not fetch one or both OpenAPI documents."
fi
