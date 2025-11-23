#!/usr/bin/env bash
set -euo pipefail

BASE_RAILWAY_URL="${GHOST_RAILWAY_BASE_URL:-https://ghost-protocol-production.up.railway.app}"
HEALTH_PATH="${GHOST_RAILWAY_HEALTH_PATH:-/health}"
RAILWAY_TMP_BODY="${TMPDIR:-/tmp}/ghost_railway_body.txt"

BASE_RAILWAY_URL="${BASE_RAILWAY_URL%/}"

http_check() {
  local label="$1"
  local url="$2"
  echo "▶ ${label}: ${url}"
  : >"${RAILWAY_TMP_BODY}"
  local status_output
  if ! status_output="$(curl -sS --max-time 10 -o "${RAILWAY_TMP_BODY}" -w "STATUS:%{http_code}" "${url}")"; then
    echo "❌ ${label} failed (curl error)"
    return 1
  fi
  local status="${status_output#STATUS:}"
  if [[ "${status}" != "200" && "${status}" != "307" ]]; then
    echo "❌ ${label} returned HTTP ${status}"
    if [[ -s "${RAILWAY_TMP_BODY}" ]]; then
      echo "--- response body ---"
      head -n 100 "${RAILWAY_TMP_BODY}"
      echo "---------------------"
    fi
    return 1
  fi
  echo "✅ ${label} OK (HTTP ${status})"
}

http_check "railway health" "${BASE_RAILWAY_URL}${HEALTH_PATH}"
http_check "railway cockpit" "${BASE_RAILWAY_URL}/cockpit"
echo "✅ Railway service is healthy."
