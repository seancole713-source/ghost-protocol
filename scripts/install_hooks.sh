git -C "$REPO_ROOT" config core.hooksPath "$HOOK_DIR"
echo "✅ Git hooks path set to $HOOK_DIR"
#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOK_DIR="$REPO_ROOT/.githooks"
PRE_PUSH="$HOOK_DIR/pre-push"

if [[ ! -f "$PRE_PUSH" ]]; then
	echo "❌ Missing .githooks/pre-push. Create the hook before installing." >&2
	exit 1
fi

git -C "$REPO_ROOT" config core.hooksPath "$HOOK_DIR"
chmod +x "$PRE_PUSH"

echo "✅ Git hooks installed (pre-push → ghost_smoke.sh local)"
