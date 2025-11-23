#!/usr/bin/env bash
set -euo pipefail

BASE_LOCAL_URL="http://localhost:8080"
BASE_RAILWAY_URL="${GHOST_RAILWAY_BASE_URL:-https://ghost-protocol-production.up.railway.app}"
HEALTH_PATH="${GHOST_RAILWAY_HEALTH_PATH:-/health}"
SMOKE_TMP_BODY="${TMPDIR:-/tmp}/ghost_smoke_body.txt"
SMOKE_TMP_HTML="${TMPDIR:-/tmp}/ghost_cockpit.html"
FORBIDDEN_TMP="${TMPDIR:-/tmp}/ghost_forbidden.txt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SMOKE_MODE=""

BASE_LOCAL_URL="${BASE_LOCAL_URL%/}"
BASE_RAILWAY_URL="${BASE_RAILWAY_URL%/}"

usage() {
  cat <<'USAGE'
Usage: scripts/ghost_smoke.sh <local|railway|full>

Modes:
  local    Run health, cockpit, and API checks against localhost (port 8080)
  railway  Run the same checks against the production Railway deployment
  full     Run local followed by railway
USAGE
}

finalize() {
  local status="$1"
  local mode_label="$2"
  [[ -z "${mode_label}" ]] && return
  if [[ "${status}" -eq 0 ]]; then
    echo "✅ Ghost smoke test (${mode_label}) OK"
  else
    echo "❌ Ghost smoke test (${mode_label}) FAILED"
  fi
}

run_forbidden_scan() {
  echo "🔎 Checking for disallowed patterns..."
  local -a include_globs=(
    "--include=*.py"
    "--include=*.pyi"
    "--include=*.sh"
    "--include=*.env"
    "--include=*.json"
    "--include=*.yaml"
    "--include=*.yml"
    "--include=*.toml"
    "--include=*.ini"
    "--include=*.cfg"
    "--include=*.js"
    "--include=*.ts"
    "--include=*.css"
    "--include=*.html"
    "--include=Dockerfile"
    "--include=Procfile"
    "--include=Makefile"
  )
  local -a exclude_files=(
    "--exclude=ghost_smoke.sh"
    "--exclude=check_no_placeholders.sh"
  )
  if grep -R --line-number --exclude-dir=.git "${include_globs[@]}" "${exclude_files[@]}" -E "SIM_MODE=1|SIMULATION_MODE|your_key_here|example_api_key|PLACEHOLDER" "${REPO_ROOT}" >"${FORBIDDEN_TMP}" 2>/dev/null; then
    echo "❌ Forbidden patterns detected:"
    cat "${FORBIDDEN_TMP}"
    exit 1
  fi
  echo "✅ No forbidden simulation/placeholder patterns found."
}

http_check() {
  local label="$1"
  local url="$2"
  echo "▶ ${label}: ${url}"
  : >"${SMOKE_TMP_BODY}"
  local status_output
  if ! status_output="$(curl -sS --max-time 10 -o "${SMOKE_TMP_BODY}" -w "STATUS:%{http_code}" "${url}")"; then
    echo "❌ ${label} failed (curl error)"
    return 1
  fi
  local status="${status_output#STATUS:}"
  if [[ "${status}" != "200" && "${status}" != "307" ]]; then
    echo "❌ ${label} returned HTTP ${status}"
    if [[ -s "${SMOKE_TMP_BODY}" ]]; then
      echo "--- response body ---"
      head -n 200 "${SMOKE_TMP_BODY}"
      echo "---------------------"
    fi
    return 1
  fi
  echo "✅ ${label} OK (HTTP ${status})"
}

assert_cockpit_assets() {
  local base_label="$1"
  cp "${SMOKE_TMP_BODY}" "${SMOKE_TMP_HTML}"
  if ! grep -q "cockpit_v3.css" "${SMOKE_TMP_HTML}"; then
    echo "❌ ${base_label} missing cockpit_v3.css reference"
    return 1
  fi
  if ! grep -q "cockpit_v3.js" "${SMOKE_TMP_HTML}"; then
    echo "❌ ${base_label} missing cockpit_v3.js reference"
    return 1
  fi
  if grep -qi "cockpit_v1" "${SMOKE_TMP_HTML}" || grep -qi "cockpit_v2" "${SMOKE_TMP_HTML}"; then
    echo "❌ Detected legacy cockpit (v1/v2) markers in /cockpit output"
    return 1
  fi
  echo "✅ ${base_label} rendering Cockpit V3 assets only"
}

run_core_v3_checks() {
  local base_url="$1"
  local prefix="$2"
  http_check "${prefix} cockpit status" "${base_url}/api/v3/cockpit/status"
  http_check "${prefix} hunter feed" "${base_url}/api/v3/hunter/feed"
  http_check "${prefix} predictions latest" "${base_url}/api/v3/predictions/latest"
  http_check "${prefix} providers health" "${base_url}/api/v3/providers/health"
}

run_local_checks() {
  local base="${BASE_LOCAL_URL}"
  http_check "local health" "${base}${HEALTH_PATH}"
  http_check "local cockpit" "${base}/cockpit"
  assert_cockpit_assets "local cockpit"
  run_core_v3_checks "${base}" "local"
}

run_railway_checks() {
  local base="${BASE_RAILWAY_URL}"
  http_check "railway health" "${base}${HEALTH_PATH}"
  http_check "railway cockpit" "${base}/cockpit"
  assert_cockpit_assets "railway cockpit"
  run_core_v3_checks "${base}" "railway"
}

main() {
  if [[ $# -ne 1 ]]; then
    usage
    exit 1
  fi

  local mode="$1"
  SMOKE_MODE="${mode}"
  trap 'finalize "$?" "${SMOKE_MODE}"' EXIT

  cd "${REPO_ROOT}"
  run_forbidden_scan

  case "${mode}" in
    local)
      run_local_checks
      ;;
    railway)
      run_railway_checks
      ;;
    full)
      run_local_checks
      run_railway_checks
      ;;
    *)
      usage
      exit 1
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
