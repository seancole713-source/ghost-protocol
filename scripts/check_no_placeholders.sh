#!/usr/bin/env bash
# Ghost Protocol – zero-placeholder and smoke verification gate
# Usage: scripts/check_no_placeholders.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PLACEHOLDER_PATTERNS=(
  "your-key-here"
  "example-api-key"
  "example_api_key"
  "dummy-"
  "dummy_"
  "changeme"
  "lorem ipsum"
  "fake-api-key"
  "fake_token"
  "test-value"
)

GENERIC_PLACEHOLDER_PREFIX="your"
GENERIC_PLACEHOLDER_SUFFIX="_"

IGNORE_DIRS=(
  ".git" ".venv" "venv" "__pycache__" ".mypy_cache" ".ruff_cache" ".pytest_cache"
  "node_modules" "dist" "build" ".archive"
)

DOC_WHITELIST=(
  "README.md"
  "GHOST_DEV_WORKFLOW.md"
  "RAILWAY_SERVICE_POLICY.md"
  "GHOST_NO_PLACEHOLDER_ENFORCEMENT.md"
)

require_tool() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "❌ Required tool '$1' is not available" >&2
    exit 1
  fi
}

require_tool python3
require_tool curl

join_lines() {
  local joined=""
  for entry in "$@"; do
    joined+="$entry"$'\n'
  done
  printf '%s' "$joined"
}

scan_for_placeholders() {
  local matches
  matches=$(PLACEHOLDER_BLOB="$(join_lines "${PLACEHOLDER_PATTERNS[@]}")" \
IGNORE_BLOB="$(join_lines "${IGNORE_DIRS[@]}")" \
DOC_BLOB="$(join_lines "${DOC_WHITELIST[@]}")" \
  GENERIC_PREFIX="$GENERIC_PLACEHOLDER_PREFIX" \
  GENERIC_SUFFIX="$GENERIC_PLACEHOLDER_SUFFIX" \
python3 - "$REPO_ROOT" <<'PY'
import sys
from pathlib import Path
import os

repo = Path(sys.argv[1])
patterns = [line.strip().lower() for line in os.environ["PLACEHOLDER_BLOB"].splitlines() if line.strip()]
ignore_dirs = set(line.strip() for line in os.environ["IGNORE_BLOB"].splitlines() if line.strip())
doc_whitelist = set(line.strip() for line in os.environ["DOC_BLOB"].splitlines() if line.strip())
  generic_prefix = os.environ.get("GENERIC_PREFIX", "").strip().lower()
  generic_suffix = os.environ.get("GENERIC_SUFFIX", "").strip().lower()
  generic_marker = generic_prefix + generic_suffix if generic_prefix and generic_suffix else ""

def should_skip(path: Path) -> bool:
  return any(part in ignore_dirs for part in path.parts)

def code_fence_lines(text: str):
  inside = False
  allowed = set()
  for idx, line in enumerate(text.splitlines(), 1):
    stripped = line.strip()
    if stripped.startswith("```"):
      inside = not inside
      continue
    if inside:
      allowed.add(idx)
  return allowed

violations = []

for path in repo.rglob("*"):
  if not path.is_file():
    continue
  rel = path.relative_to(repo)
  if should_skip(rel):
    continue
  try:
    text = path.read_text(encoding="utf-8")
  except Exception:
    try:
      text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
      continue

  rel_str = str(rel)
  is_doc = rel.suffix.lower() in {".md", ".markdown"}
  whitelist = rel.name in doc_whitelist
  allowed_lines = code_fence_lines(text) if (is_doc and whitelist) else set()

  for idx, line in enumerate(text.splitlines(), 1):
    lower_line = line.lower()
    matched = False
    for pat in patterns:
      if pat in lower_line:
        matched = True
        break

    if not matched and generic_marker and generic_marker in lower_line:
      matched = True

    if matched:
      if is_doc and whitelist and idx in allowed_lines:
        continue
      violations.append(f"{rel_str}:{idx}:{line.strip()}".strip())

if violations:
  print("\n".join(violations))
PY
)

  if [[ -n "$matches" ]]; then
    echo "❌ PLACEHOLDERS FOUND – fix these before commit/push"
    echo "$matches"
    exit 1
  fi

  echo "✅ NO PLACEHOLDERS FOUND"
}

run_smoke_tests() {
  local explicit_target="${1:-}" explicit_label="${2:-}" label target

  if [[ -n "$explicit_target" ]]; then
    target="$explicit_target"
    label="${explicit_label:-custom}"
  elif [[ "${GHOST_ENV:-}" == "local" || -z "${RAILWAY_URL:-}" ]]; then
    target="${LOCAL_GHOST_URL:-http://localhost:8080}"
    label="local"
  else
    target="${RAILWAY_URL%/}"
    label="railway"
  fi

  target="${target%/}"
  if [[ $target != http://* && $target != https://* ]]; then
    target="https://$target"
  fi

  echo "🌐 Running smoke tests against [$label] $target"

  local timeout="${SMOKE_TIMEOUT:-10}"
  local curl_flags=(--fail --show-error --silent --max-time "$timeout")

  echo "→ GET $target/health"
  local health
  health=$(curl "${curl_flags[@]}" "$target/health")

  echo "→ GET $target/cockpit"
  curl "${curl_flags[@]}" "$target/cockpit" >/dev/null

  echo "→ GET $target/api/v3/cockpit/version"
  local version_json
  version_json=$(curl "${curl_flags[@]}" "$target/api/v3/cockpit/version")

  python3 - "$version_json" <<'PY'
import json, sys

try:
  payload = json.loads(sys.argv[1])
except Exception as exc:
  raise SystemExit(f"Version endpoint returned invalid JSON: {exc}")

if payload.get("ui") != "cockpit_v3" or payload.get("status") != "live":
  raise SystemExit("Version endpoint does not confirm cockpit_v3 live status")
PY

  echo "✅ Smoke tests passed"
}

main() {
  scan_for_placeholders
  run_smoke_tests "${1:-}"
  echo "✅ GHOST CHECKS PASSED"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi